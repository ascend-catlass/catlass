from __future__ import annotations

import struct

import pytest

import catlass.tla as tla
from catlass.base_dsl import BaseDSL
from catlass.base_dsl.runtime import ascend_debug_fifo as fifo
from catlass.execution import (
    TlaCompileOption,
    TlaKernelCompileError,
    _validate_vf_debug_environment,
)
from catlass.tla.runtime import make_fake_tensor


def _tensor():
    return make_fake_tensor(
        tla.Float32, (64,), (1,), origin_shape=(64,), layout_tag=tla.arch.RowMajor
    )


def _kernel(kind, length, mode="simd"):
    @tla.kernel
    def kernel(source: tla.Tensor):
        ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        local = tla.make_tensor_like(ptr, source, tla.arch.RowMajor)
        with tla.vector():
            with tla.vec.func(mode=mode):
                if kind == "reg":
                    tla.print(local.load(), length)
                elif kind == "ub":
                    tla.print(local, length)
                elif kind == "gm":
                    tla.print(source, length)
                else:
                    tla.print("value={}", tla.Int32(7))

    return kernel


@pytest.mark.parametrize(
    ("kind", "op"),
    [("reg", "print_vector"), ("ub", "print_tensor"), ("scalar", "debug_print")],
)
def test_vf_print_reuses_public_entry(kind, op):
    ir = _kernel(kind, 4).dump_mlir(type_args=(_tensor(),))
    assert f"tla.{op}" in ir
    assert "tla.vec.func" in ir


@pytest.mark.parametrize("length", [0, -1, 65, True, 1.5])
def test_register_print_rejects_invalid_prefix(length):
    with pytest.raises((tla.TlaCoreAPIError, TypeError, ValueError)):
        _kernel("reg", length).dump_mlir(type_args=(_tensor(),))


def test_vf_print_rejects_gm_access():
    with pytest.raises(tla.TlaCoreAPIError, match="requires UB storage"):
        _kernel("gm", 4).dump_mlir(type_args=(_tensor(),))


@pytest.mark.parametrize("kind", ["scalar", "gm", "ub"])
def test_simt_print_reuses_public_entry(kind):
    ir = _kernel(kind, 4, "simt").dump_mlir(type_args=(_tensor(),))
    assert "tla.debug_print" in ir if kind == "scalar" else "tla.print_tensor" in ir


def _lowered_scalar_print():
    kernel = _kernel("scalar", 4)
    return BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=(_tensor(),),
        location=kernel.decorator_location,
    )


@pytest.mark.parametrize(
    "release_asc,release_stack", [(True, False), (False, True), (True, True)]
)
def test_vf_debug_preserves_both_reserved_ub_regions(release_asc, release_stack):
    lowered = _lowered_scalar_print()
    options = TlaCompileOption(
        cce_disable_asc_reserved_ubuf=release_asc,
        cce_disable_vf_stack_reserved_ubuf=release_stack,
    )
    with pytest.raises(TlaKernelCompileError, match="default reserved UB"):
        _validate_vf_debug_environment(lowered.module, options)


def test_vf_debug_reports_missing_toolkit_support(monkeypatch):
    lowered = _lowered_scalar_print()
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    with pytest.raises(TlaKernelCompileError, match="CANN VF debug headers"):
        _validate_vf_debug_environment(lowered.module, TlaCompileOption())


def _register_record():
    record = bytearray(72 + 32)
    struct.pack_into("<II", record, 0, fifo._FIFO_TENSOR, len(record) - 8)
    struct.pack_into("<I", record, 12, 0)  # f32
    struct.pack_into("<I", record, 16, 0x54500000)
    struct.pack_into("<H", record, 24, 8)  # REG
    struct.pack_into("<I", record, 28, 1)
    struct.pack_into("<I", record, 32, 4)
    struct.pack_into("<I", record, 68, 16)
    struct.pack_into("<4f", record, 72, -7.0, -6.5, -6.0, -5.5)
    return record


@pytest.mark.parametrize("length", [0, -1, 65, True, 1.5])
def test_simt_tensor_print_rejects_invalid_static_prefix(length):
    with pytest.raises((tla.TlaCoreAPIError, TypeError, ValueError)):
        _kernel("gm", length, "simt").dump_mlir(type_args=(_tensor(),))


def test_simt_print_does_not_require_simd_reserved_ub(monkeypatch):
    kernel = _kernel("scalar", 4, "simt")
    lowered = BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=(_tensor(),),
        location=kernel.decorator_location,
    )
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    _validate_vf_debug_environment(
        lowered.module,
        TlaCompileOption(
            cce_disable_asc_reserved_ubuf=True, cce_disable_vf_stack_reserved_ubuf=True
        ),
    )


@pytest.mark.parametrize("mixed,subblocks", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize(
    "position", [fifo._REGISTER_POSITION, fifo._UNIFIED_BUFFER_POSITION]
)
def test_vf_prefix_block_identity_and_bounds(capfd, mixed, subblocks, position):
    capture = fifo._FifoData(
        device_ptr=0,
        region_size=512,
        record_count=1,
        launch_block_count=2,
        mixed_handoff=mixed,
        aiv_subblock_count=subblocks,
        block_length=512,
        ring_buffer_offset=80,
        ring_buffer_bytes=256,
    )
    record = _register_record()
    struct.pack_into("<H", record, 24, position)
    for block in range(2):
        for subblock in range(subblocks):
            struct.pack_into("<H", record, 26, block * subblocks + subblock)
            shape, logical, decoded_subblock = fifo._tensor_record_metadata(
                memoryview(record), len(record), capture, vf_record=True
            )
            assert logical == block
            assert decoded_subblock == (subblock if mixed else -1)
            fifo._render_tensor(
                memoryview(record), shape, logical, subblock=decoded_subblock
            )
            output = capfd.readouterr().out
            assert "shape=[4]" in output
            assert "-6.5" in output
            assert f"block={block}," in output
            if mixed:
                assert f"subblock={subblock}," in output
            else:
                assert "subblock=" not in output
    struct.pack_into("<H", record, 26, 2 * subblocks)
    with pytest.raises(fifo.AscDebugFifoError, match="exceeds launch block_num"):
        fifo._tensor_record_metadata(
            memoryview(record), len(record), capture, vf_record=True
        )


def test_vf_decoder_rejects_legacy_tensor_descriptor():
    record = _register_record()
    struct.pack_into("<I", record, 16, 0x54510000)  # native Tensor subblock 0
    capture = fifo._FifoData(0, 512, 1, 1, True, 512, 80, 256)
    with pytest.raises(fifo.AscDebugFifoError, match="invalid position or descriptor"):
        fifo._tensor_record_metadata(
            memoryview(record), len(record), capture, vf_record=True
        )
