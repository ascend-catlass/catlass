"""Device regressions for print snapshots while the source UB is reused."""

import pytest

pytestmark = pytest.mark.npu


def _simd_snapshot_kernel():
    global tla
    import catlass.tla as tla

    @tla.kernel
    def simd_print_snapshots(
        source: tla.Tensor, destination: tla.Tensor, repetitions: tla.Int32
    ):
        pointer = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        local = tla.make_tensor_like(pointer, source, tla.arch.RowMajor)
        loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
        stored = tla.flag("stored", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(local, source)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                current = local.load()
                for iteration in tla.range(0, repetitions, 1):
                    tla.print(current, 4)
                    local.store(current)
                    tla.print(local, 4)
                    local.store(current + 4096.0)
                    tla.print(local, 4)
                    current = current + 128.0
            tla.set_flag(stored)
            tla.wait_flag(stored)
            tla.copy(destination, local)
            tla.pipe_barrier(tla.pipes.ALL)

    return simd_print_snapshots


def _simt_snapshot_kernel():
    global tla
    import catlass.tla as tla

    @tla.kernel
    def simt_print_snapshots(
        source: tla.Tensor, destination: tla.Tensor, repetitions: tla.Int32
    ):
        pointer = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        local = tla.make_tensor_like(pointer, source, tla.arch.RowMajor)
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=64):
                thread, _, _ = tla.arch.thread_idx()
                for iteration in tla.range(0, repetitions, 1):
                    local[thread] = source[thread] + iteration.to(tla.Float32) * 128.0
                    tla.arch.sync_threads()
                    if thread == 0:
                        tla.print(local, 8)
                    tla.arch.sync_threads()
                    local[thread] = -local[thread]
                destination[thread] = local[thread]

    return simt_print_snapshots


def _buffers(device):
    import torch
    import torch_npu  # noqa: F401
    import catlass.tla as tla
    from catlass.tla.runtime import make_fake_tensor

    torch.npu.set_device(device)
    source = torch.arange(64, dtype=torch.float32, device=f"npu:{device}")
    guarded = torch.full((96,), -12345.0, device=f"npu:{device}")
    fake = make_fake_tensor(
        tla.Float32, (64,), (1,), origin_shape=(64,), layout_tag=tla.arch.RowMajor
    )
    return source, guarded, fake


def _tensor_records(output):
    records = []
    for line in output.splitlines():
        if line.startswith("DumpTensor:"):
            metadata, values = line.rsplit(" [", 1)
            records.append(
                (
                    metadata,
                    [float(value) for value in values.removesuffix("]").split(",")],
                )
            )
    return records


def _check_guard_and_source(source, guarded):
    import torch

    assert (guarded[:16].cpu() == -12345.0).all()
    assert (guarded[80:].cpu() == -12345.0).all()
    torch.testing.assert_close(
        source.cpu(), torch.arange(64, dtype=torch.float32), rtol=0, atol=0
    )


def test_simd_print_snapshots_survive_ub_reuse(device, capfd):
    import torch
    from catlass.tla.runtime import from_dlpack

    repetitions = 16
    source, guarded, fake = _buffers(device)
    compiled = _simd_snapshot_kernel().compile(
        type_args=(fake, fake, tla.Int32(repetitions)), options="--npu-arch 3510"
    )
    capfd.readouterr()
    compiled(
        from_dlpack(source, layout_tag=tla.arch.RowMajor),
        from_dlpack(guarded[16:80], layout_tag=tla.arch.RowMajor),
        repetitions,
        block_num=1,
    )
    output = capfd.readouterr().out
    records = _tensor_records(output)
    assert len(records) == repetitions * 3, output
    for iteration in range(repetitions):
        expected = [float(lane + iteration * 128) for lane in range(4)]
        for offset, position, shift in ((0, "REG", 0), (1, "UB", 0), (2, "UB", 4096)):
            metadata, values = records[iteration * 3 + offset]
            assert f"position={position}," in metadata, metadata
            assert "block=0," in metadata, metadata
            assert values == [value + shift for value in expected], output
    reference = torch.arange(64, dtype=torch.float32) + (repetitions - 1) * 128 + 4096
    torch.testing.assert_close(guarded[16:80].cpu(), reference, rtol=0, atol=0)
    _check_guard_and_source(source, guarded)
    assert "truncated" not in output and "malformed" not in output, output


def test_simt_print_snapshots_survive_ub_reuse(device, capfd):
    import torch
    from catlass.tla.runtime import from_dlpack

    repetitions = 8
    source, guarded, fake = _buffers(device)
    compiled = _simt_snapshot_kernel().compile(
        type_args=(fake, fake, tla.Int32(repetitions)), options="--npu-arch 3510"
    )
    capfd.readouterr()
    compiled(
        from_dlpack(source, layout_tag=tla.arch.RowMajor),
        from_dlpack(guarded[16:80], layout_tag=tla.arch.RowMajor),
        repetitions,
        block_num=1,
    )
    output = capfd.readouterr().out
    records = _tensor_records(output)
    assert len(records) == repetitions, output
    for iteration, (metadata, values) in enumerate(records):
        assert "block=(0, 0, 0) thread=(0, 0, 0)" in metadata, metadata
        assert values == [float(lane + iteration * 128) for lane in range(8)], output
    reference = -(torch.arange(64, dtype=torch.float32) + (repetitions - 1) * 128)
    torch.testing.assert_close(guarded[16:80].cpu(), reference, rtol=0, atol=0)
    _check_guard_and_source(source, guarded)
    assert "dropped" not in output and "malformed" not in output, output
