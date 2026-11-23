"""Independent stable-compression references for integer SIMD streams."""

import numpy as np
import pytest

from catlass import tla
from catlass.tla.runtime import from_dlpack

pytestmark = pytest.mark.npu


def _kernel(dtype):
    @tla.kernel
    def kernel(
        values: tla.Tensor,
        markers: tla.Tensor,
        output: tla.Tensor,
        result_count: tla.Tensor,
        length: tla.Int32,
        offset: tla.Int32,
    ):
        capacity = 8192
        guard = 128
        src = tla.make_tensor(
            tla.allocate(capacity, dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(capacity), tla.make_stride(1)),
        )
        pred = tla.make_tensor(
            tla.allocate(capacity, tla.UInt16, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(capacity), tla.make_stride(1)),
        )
        storage = tla.make_tensor(
            tla.allocate(capacity + 2 * guard + 128, dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(
                tla.make_shape(capacity + 2 * guard + 128), tla.make_stride(1)
            ),
        )
        dst = tla.make_tensor(
            storage.ptr + guard + offset,
            tla.make_layout(tla.make_shape(capacity), tla.make_stride(1)),
        )
        count = tla.make_tensor(
            tla.allocate(64, tla.UInt32, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(1), tla.make_stride(1)),
        )
        with tla.vector():
            tla.copy(src, values)
            tla.copy(pred, markers)
            tla.copy(storage, output)
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt16)
                accumulator = tla.full(0, dtype=tla.Int16).bitcast(tla.UInt16)
                ones = tla.full(1, dtype=tla.Int16).bitcast(tla.UInt16)
                zeros = tla.full(0, dtype=tla.Int16).bitcast(tla.UInt16)
                bulk = length // 128 * 128
                for start in tla.range(0, bulk, 128):
                    data = tla.tile_view(
                        src, tla.make_shape(128), tla.make_coord(start // 128)
                    ).load()
                    tags = tla.tile_view(
                        pred, tla.make_shape(128), tla.make_coord(start // 128)
                    ).load()
                    keep = tla.cmp(tags, zeros, mode="ne", mask=full)
                    packed = tla.squeeze(data, keep, record_nbytes=True)
                    dst = dst.store(
                        packed,
                        tla.params.UnalignStoreParams(
                            post_mode=tla.params.PostMode.POST_MODE_UPDATE_NBYTES
                        ),
                    )
                    accumulator = accumulator + tla.where(keep, ones, zeros)
                if bulk < length:
                    tail_data = tla.tile_view(
                        src, tla.make_shape(128), tla.make_coord(bulk // 128)
                    ).load()
                    tail_tags = tla.tile_view(
                        pred, tla.make_shape(128), tla.make_coord(bulk // 128)
                    ).load()
                    tail, _ = tla.update_mask(length - bulk, dtype=tla.UInt16)
                    tail_keep = tla.bitwise_and(
                        tla.cmp(tail_tags, zeros, mode="ne", mask=full), tail
                    )
                    tail_packed = tla.squeeze(tail_data, tail_keep, record_nbytes=True)
                    dst = dst.store(
                        tail_packed,
                        tla.params.UnalignStoreParams(
                            post_mode=tla.params.PostMode.POST_MODE_UPDATE_NBYTES
                        ),
                    )
                    accumulator = accumulator + tla.where(tail_keep, ones, zeros)
                total = accumulator.reduce(tla.ReductionOp.ADD, mask=full)
                count.store(
                    total.to(
                        tla.UInt32,
                        tla.params.CastParams(reg_slot=tla.params.RegSlot.ZERO),
                    )
                )
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(output, storage)
            tla.copy(result_count, count)
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel


@pytest.mark.parametrize(
    "dtype,numpy_dtype", [(tla.Int16, np.int16), (tla.UInt16, np.uint16)]
)
def test_stable_squeeze_stream(device, dtype, numpy_dtype):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    torch.npu.set_op_timeout_ms(30000)
    rng = np.random.default_rng(20261110)
    raw = rng.integers(0, 65536, 8192, dtype=np.uint16).view(numpy_dtype)
    raw[:4] = np.array([0, 0x7FFF, 0x8000, 0xFFFF], dtype=np.uint16).view(numpy_dtype)
    markers = np.zeros(8192, dtype=np.uint16)
    payload = torch.from_numpy(raw.copy()).to(f"npu:{device}")
    tags = torch.from_numpy(markers.copy()).to(f"npu:{device}")
    initial = np.full(8192 + 384, 0x5A5A, dtype=numpy_dtype)
    output = torch.from_numpy(initial.copy()).to(f"npu:{device}")
    count = torch.zeros(1, dtype=torch.int32, device=f"npu:{device}").view(torch.uint32)
    args = tuple(
        from_dlpack(tensor, layout_tag=tla.arch.RowMajor)
        for tensor in (payload, tags, output, count)
    )
    compiled = tla.compile(
        _kernel(dtype), *args, tla.Int32(0), tla.Int32(0), options="--npu-arch 3510"
    )
    stream = torch.npu.Stream(device=f"npu:{device}")
    calls = 0
    for phase in range(3):
        markers[:] = (rng.random(8192) < (1.0, 0.0, 0.37)[phase]).astype(
            np.uint16
        ) * np.uint16(65535)
        tags.copy_(torch.from_numpy(markers.copy()))
        for length, offset in [(n, n % 16) for n in range(129)] + [
            (8192, 0),
            (0, 1),
            (1, 15),
            (127, 3),
            (128, 0),
            (129, 1),
            (255, 15),
            (257, 3),
            (8191, 1),
            (17, 0),
            (0, 15),
            (8192, 0),
        ]:
            output.copy_(torch.from_numpy(initial.copy()))
            stream.wait_stream(torch.npu.current_stream(device))
            with torch.npu.stream(stream):
                compiled(*args, length, offset, block_num=1)
            stream.synchronize()
            expected = raw[:length][markers[:length] != 0]
            got = output.cpu().numpy()
            start = 128 + offset
            assert int(count.cpu().numpy()[0]) == expected.size
            np.testing.assert_array_equal(got[start : start + expected.size], expected)
            np.testing.assert_array_equal(got[:start], initial[:start])
            np.testing.assert_array_equal(
                got[start + expected.size :], initial[start + expected.size :]
            )
            calls += 1
    print(
        f"Stable compression: dtype={dtype}, correctness calls={calls}, reused buffers and non-default stream",
        flush=True,
    )


def _plain_kernel(dtype, record):
    @tla.kernel
    def kernel(values: tla.Tensor, markers: tla.Tensor, output: tla.Tensor):
        source = tla.make_tensor(
            tla.allocate(128, dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(128), tla.make_stride(1)),
        )
        predicate = tla.make_tensor(
            tla.allocate(128, tla.UInt16, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(128), tla.make_stride(1)),
        )
        destination = tla.make_tensor(
            tla.allocate(128, dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(128), tla.make_stride(1)),
        )
        with tla.vector():
            tla.copy(source, values)
            tla.copy(predicate, markers)
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt16)
                zeros = tla.full(0, dtype=tla.Int16).bitcast(tla.UInt16)
                keep = tla.cmp(predicate.load(), zeros, mode="ne", mask=full)
                destination.store(
                    tla.squeeze(source.load(), keep, record_nbytes=record)
                )
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(output, destination)
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel


@pytest.mark.parametrize(
    "dtype,numpy_dtype", [(tla.Int16, np.int16), (tla.UInt16, np.uint16)]
)
@pytest.mark.parametrize("record", [False, True])
def test_squeeze_integer_values(device, dtype, numpy_dtype, record):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    values = np.resize(
        np.array([0, 0x7FFF, 0x8000, 0xFFFF], dtype=np.uint16), 128
    ).view(numpy_dtype)
    payload = torch.from_numpy(values.copy()).to(f"npu:{device}")
    markers = torch.empty(128, dtype=torch.uint16, device=f"npu:{device}")
    output = torch.empty_like(payload)
    args = tuple(
        from_dlpack(tensor, layout_tag=tla.arch.RowMajor)
        for tensor in (payload, markers, output)
    )
    compiled = tla.compile(
        _plain_kernel(dtype, record), *args, options="--npu-arch 3510"
    )
    for step in (1, 2, 3, 129):
        keep = np.arange(128) % step == 0
        if step == 129:
            keep[:] = False
        markers.copy_(torch.from_numpy(keep.astype(np.uint16)))
        compiled(*args, block_num=1)
        torch.npu.synchronize()
        expected = np.zeros(128, dtype=numpy_dtype)
        expected[: np.count_nonzero(keep)] = values[keep]
        np.testing.assert_array_equal(output.cpu().numpy(), expected)


def _existing_type_kernel(dtype, lanes):
    carrier = {
        tla.UInt8: tla.Int8,
        tla.UInt16: tla.Int16,
        tla.UInt32: tla.Int32,
    }.get(dtype, dtype)

    @tla.kernel
    def kernel(values: tla.Tensor, output: tla.Tensor):
        source = tla.make_tensor(
            tla.allocate(lanes, dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(lanes), tla.make_stride(1)),
        )
        storage = tla.make_tensor(
            tla.allocate(4 * lanes, dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(4 * lanes), tla.make_stride(1)),
        )
        destination = tla.make_tensor(
            storage.ptr + lanes + 1,
            tla.make_layout(tla.make_shape(2 * lanes), tla.make_stride(1)),
        )
        with tla.vector():
            tla.copy(source, values)
            tla.copy(storage, output)
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                data = source.load()
                full = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
                if tla.const_expr(dtype.width == 8):
                    keep = tla.create_mask(pattern=tla.mask.M4, dtype=dtype)
                else:
                    zeros = tla.full(0, dtype=carrier).bitcast(dtype)
                    keep = tla.cmp(data, zeros, mode="gt", mask=full)
                for _ in tla.range_constexpr(2):
                    packed = tla.squeeze(data, keep, record_nbytes=True)
                    destination = destination.store(
                        packed,
                        tla.params.UnalignStoreParams(
                            post_mode=tla.params.PostMode.POST_MODE_UPDATE_NBYTES
                        ),
                    )
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(output, storage)
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel


@pytest.mark.parametrize(
    "dtype,numpy_dtype,lanes",
    [
        (tla.UInt8, np.uint8, 256),
        (tla.Float32, np.float32, 64),
        (tla.Float16, np.float16, 128),
    ],
)
def test_squeeze_existing_types_record_byte_count(device, dtype, numpy_dtype, lanes):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    values = np.arange(-lanes // 2, lanes // 2, dtype=np.int32).astype(numpy_dtype)
    initial = np.full(4 * lanes, 91, dtype=numpy_dtype)
    source = torch.from_numpy(values.copy()).to(f"npu:{device}")
    output = torch.from_numpy(initial.copy()).to(f"npu:{device}")
    args = tuple(
        from_dlpack(tensor, layout_tag=tla.arch.RowMajor) for tensor in (source, output)
    )
    compiled = tla.compile(
        _existing_type_kernel(dtype, lanes), *args, options="--npu-arch 3510"
    )
    compiled(*args, block_num=1)
    torch.npu.synchronize()
    expected = initial.copy()
    kept = values[::4] if dtype.width == 8 else values[values > 0]
    packed = np.tile(kept, 2)
    expected[lanes + 1 : lanes + 1 + packed.size] = packed
    np.testing.assert_array_equal(output.cpu().numpy(), expected)


def _squeeze_loop_kernel(dtype, lanes, sparse=False):
    carrier = {
        tla.UInt8: tla.Int8,
        tla.UInt16: tla.Int16,
        tla.UInt32: tla.Int32,
    }.get(dtype, dtype)
    compare_dtype = {8: tla.UInt8, 16: tla.UInt16, 32: tla.UInt32}[dtype.width]

    @tla.kernel
    def kernel(
        values: tla.Tensor, output: tla.Tensor, kept: tla.Int32, trips: tla.Int32
    ):
        source = tla.make_tensor(
            tla.allocate(lanes, dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(lanes), tla.make_stride(1)),
        )
        storage = tla.make_tensor(
            tla.allocate(8 * lanes, dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(8 * lanes), tla.make_stride(1)),
        )
        cursor = tla.make_tensor(
            storage.ptr + lanes + 3,
            tla.make_layout(tla.make_shape(lanes), tla.make_stride(1)),
        )
        with tla.vector():
            tla.copy(source, values)
            tla.copy(storage, output)
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                data = source.load()
                for i in tla.range(trips):
                    # Different lengths across the same loop exercise real
                    # cursor accumulation, including zero-length appends.
                    length = (kept + i * 17) % (lanes + 1)
                    keep, _ = tla.update_mask(length, dtype=dtype)
                    if tla.const_expr(sparse):
                        if tla.const_expr(dtype.width == 8):
                            selected = tla.create_mask(pattern=tla.mask.M4, dtype=dtype)
                        else:
                            zeros = tla.full(0, dtype=carrier).bitcast(compare_dtype)
                            full = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=compare_dtype
                            )
                            selected = tla.cmp(
                                data.bitcast(compare_dtype), zeros, mode="ne", mask=full
                            )
                        keep = tla.bitwise_and(keep, selected)
                    packed = tla.squeeze(data, keep, record_nbytes=True)
                    cursor = cursor.store(
                        packed,
                        tla.params.UnalignStoreParams(
                            post_mode=tla.params.PostMode.POST_MODE_UPDATE_NBYTES
                        ),
                    )
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(output, storage)
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel


@pytest.mark.parametrize(
    "dtype,numpy_dtype,lanes",
    [
        (tla.Int8, np.int8, 256),
        (tla.UInt8, np.uint8, 256),
        (tla.UInt32, np.uint32, 64),
        (tla.Int16, np.int16, 128),
        (tla.UInt16, np.uint16, 128),
        (tla.Int32, np.int32, 64),
        (tla.Float16, np.float16, 128),
        (tla.Float32, np.float32, 64),
    ],
)
def test_compressed_update_appends_variable_counts(device, dtype, numpy_dtype, lanes):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    values = np.arange(lanes, dtype=np.int32).astype(numpy_dtype)
    initial = np.full(8 * lanes, 91, dtype=numpy_dtype)
    source = torch.from_numpy(values.copy()).to(f"npu:{device}")
    output = torch.from_numpy(initial.copy()).to(f"npu:{device}")
    args = tuple(
        from_dlpack(tensor, layout_tag=tla.arch.RowMajor) for tensor in (source, output)
    )
    compiled = tla.compile(
        _squeeze_loop_kernel(dtype, lanes),
        *args,
        tla.Int32(0),
        tla.Int32(0),
        options="--npu-arch 3510",
    )
    for kept, trips in [
        (0, 0),
        (0, 1),
        (1, 3),
        (lanes // 2 - 1, 3),
        (lanes // 2, 3),
        (lanes - 1, 3),
        (lanes, 2),
    ]:
        output.copy_(torch.from_numpy(initial.copy()).to(f"npu:{device}"))
        compiled(*args, tla.Int32(kept), tla.Int32(trips), block_num=1)
        torch.npu.synchronize()
        expected = initial.copy()
        cursor = lanes + 3
        for i in range(trips):
            length = (kept + i * 17) % (lanes + 1)
            expected[cursor : cursor + length] = values[:length]
            cursor += length
        np.testing.assert_array_equal(output.cpu().numpy(), expected)


@pytest.mark.parametrize(
    "dtype,numpy_dtype,lanes",
    [
        (tla.Int8, np.int8, 256),
        (tla.UInt8, np.uint8, 256),
        (tla.Int16, np.int16, 128),
        (tla.UInt16, np.uint16, 128),
        (tla.Int32, np.int32, 64),
        (tla.UInt32, np.uint32, 64),
    ],
)
def test_compressed_update_sparse_masks(device, dtype, numpy_dtype, lanes):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    values = np.arange(lanes, dtype=np.int32).astype(numpy_dtype)
    values[::3] = 0
    initial = np.full(8 * lanes, 91, dtype=numpy_dtype)
    source = torch.from_numpy(values.copy()).to(f"npu:{device}")
    output = torch.from_numpy(initial.copy()).to(f"npu:{device}")
    args = tuple(
        from_dlpack(tensor, layout_tag=tla.arch.RowMajor) for tensor in (source, output)
    )
    compiled = tla.compile(
        _squeeze_loop_kernel(dtype, lanes, sparse=True),
        *args,
        tla.Int32(0),
        tla.Int32(0),
        options="--npu-arch 3510",
    )
    for kept, trips in [(0, 0), (0, 1), (1, 3), (lanes - 1, 3), (lanes, 2)]:
        output.copy_(torch.from_numpy(initial.copy()).to(f"npu:{device}"))
        compiled(*args, tla.Int32(kept), tla.Int32(trips), block_num=1)
        torch.npu.synchronize()
        expected = initial.copy()
        cursor = lanes + 3
        for i in range(trips):
            prefix = values[: (kept + i * 17) % (lanes + 1)]
            packed = prefix[::4] if dtype.width == 8 else prefix[prefix != 0]
            expected[cursor : cursor + packed.size] = packed
            cursor += packed.size
        np.testing.assert_array_equal(output.cpu().numpy(), expected)


def _byte_count_update_kernel(record):
    post_mode = (
        tla.params.PostMode.POST_MODE_UPDATE_NBYTES
        if record
        else tla.params.PostMode.POST_MODE_UPDATE
    )

    @tla.kernel
    def kernel(values: tla.Tensor, output: tla.Tensor, kept: tla.Int32):
        source = tla.make_tensor(
            tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
        )
        storage = tla.make_tensor(
            tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(256), tla.make_stride(1)),
        )
        cursor = tla.make_tensor(
            storage.ptr + 65,
            tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
        )
        with tla.vector():
            tla.copy(source, values)
            tla.copy(storage, output)
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                keep, _ = tla.update_mask(kept, dtype=tla.Float32)
                packed = tla.squeeze(source.load(), keep, record_nbytes=record)
                if tla.const_expr(record):
                    packed = packed + packed
                store_params = tla.params.UnalignStoreParams(post_mode=post_mode)
                cursor = cursor.store(packed, store_params)
                marker_mask = tla.create_mask(pattern=tla.mask.VL1, dtype=tla.Float32)
                marker = tla.squeeze(
                    tla.full(1234, dtype=tla.Float32), marker_mask, record_nbytes=record
                )
                cursor.store(marker, store_params)
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(output, storage)
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel


@pytest.mark.parametrize("record", [False, True])
def test_update_uses_recorded_byte_count(device, record):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    values = np.arange(1, 65, dtype=np.float32)
    initial = np.full(256, 5678, dtype=np.float32)
    source = torch.from_numpy(values.copy()).to(f"npu:{device}")
    output = torch.from_numpy(initial.copy()).to(f"npu:{device}")
    args = tuple(
        from_dlpack(tensor, layout_tag=tla.arch.RowMajor) for tensor in (source, output)
    )
    compiled = tla.compile(
        _byte_count_update_kernel(record),
        *args,
        tla.Int32(0),
        options="--npu-arch 3510",
    )
    for kept in [0, 1, 41, 64]:
        output.copy_(torch.from_numpy(initial.copy()).to(f"npu:{device}"))
        compiled(*args, tla.Int32(kept), block_num=1)
        torch.npu.synchronize()
        expected = initial.copy()
        count = kept if record else 64
        expected[65 : 65 + kept] = values[:kept] * (2 if record else 1)
        if not record:
            expected[65 + kept : 65 + 64] = 0
            expected[65 + 65 : 65 + 128] = 0
        expected[65 + count] = 1234
        np.testing.assert_array_equal(output.cpu().numpy(), expected)
