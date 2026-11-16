"""Scatter frontend type, predicate and index contracts."""

import pytest

import catlass.tla as tla
from catlass._mlir import ir
from catlass.base_dsl.op import dsl_user_op
from catlass.core_api import VectorSSA
from catlass.tla.runtime import make_fake_tensor
from catlass.types import TlaVectorSSATypeDescriptor, dtype_size_bytes


@dsl_user_op
def _register(dtype, lanes, *, loc=None):
    desc = TlaVectorSSATypeDescriptor(lanes, dtype.dtype)
    value = ir.Operation.create(
        "builtin.unrealized_conversion_cast", results=[desc.to_mlir_type()], loc=loc
    ).results[0]
    return VectorSSA(value)


@tla.kernel
def _kernel(dst: tla.Tensor, cfg: tla.Constexpr):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src = _register(cfg[0], cfg[2])
            indices = _register(cfg[1], cfg[3])
            mask = None
            if tla.const_expr(cfg[4] is not None):
                mask = tla.create_mask(pattern=tla.mask.M3, dtype=cfg[4])
            result = tla.scatter(dst, src, indices, mask=mask)
            assert result is None


def _capture(compiler, src=tla.Float32, index=tla.UInt32, n=64, ni=64,
             mask=None, dst=None, addr=tla.AddressSpace.ub):
    tensor = make_fake_tensor(
        dst or src, (512,), (1,), addrspace=addr,
        origin_shape=(512,), layout_tag=tla.arch.RowMajor,
    )
    return compiler(_kernel, type_args=(tensor, (src, index, n, ni, mask)))


_PAIRS = (
    [(t, tla.UInt16) for t in (tla.Int8, tla.UInt8, tla.Int16, tla.UInt16,
                               tla.Float16, tla.BFloat16)]
    + [(t, tla.UInt32) for t in (tla.Int32, tla.UInt32, tla.Float32)]
    + [(t, i) for t in (tla.Int64, tla.UInt64) for i in (tla.UInt32, tla.UInt64)]
)


@pytest.mark.parametrize("src,index", _PAIRS)
@pytest.mark.parametrize("masked", [False, True])
def test_native_pairs(compiler_tlair, src, index, masked):
    size = dtype_size_bytes(src.dtype)
    mask = (tla.UInt16 if size == 1 else src) if masked else None
    text = _capture(compiler_tlair, src, index, 256 // size,
                    256 // dtype_size_bytes(index.dtype), mask)
    assert text.count('"tla.scatter"') == 1
    assert f"!tla.mask<{256 // max(size, 2)}>" in text if masked else "!tla.mask<" not in text


@pytest.mark.parametrize("change,message", [
    ({"index": tla.Int32}, "native unsigned"),
    ({"index": tla.UInt16}, "native unsigned"),
    ({"src": tla.Int8, "index": tla.UInt32}, "native unsigned"),
    ({"src": tla.Float8E4M3FN}, "unsupported src element type"),
    ({"dst": tla.Int32}, "src and dst element types must match"),
    ({"addr": tla.AddressSpace.gm}, "expected addrspace ub"),
    ({"addr": tla.AddressSpace.l1}, "expected addrspace ub"),
    ({"mask": tla.UInt16}, "predicate lanes, expected 64"),
    ({"src": tla.Int8, "index": tla.UInt16, "mask": tla.UInt8}, "expected 128"),
    ({"src": tla.Int64, "n": 32, "mask": tla.UInt32}, "expected 32"),
    ({"ni": 63}, "at least 64 scatter indices"),
    ({"src": tla.Int8, "index": tla.UInt16, "n": 3, "ni": 1}, "at least 2 scatter indices"),
    ({"src": tla.Int64, "n": 32, "ni": 31}, "at least 32 scatter indices"),
])
def test_invalid_operands(compiler_tlair, change, message):
    with pytest.raises(tla.TlaCoreAPIError, match=message):
        _capture(compiler_tlair, **change)


@pytest.mark.parametrize("change", [
    {"src": tla.Int8, "index": tla.UInt16, "n": 3, "ni": 2},
    {"n": 17}, {"n": None}, {"ni": None},
])
def test_valid_lane_metadata(compiler_tlair, change):
    assert '"tla.scatter"' in _capture(compiler_tlair, **change)


def test_requires_kernel():
    with pytest.raises(tla.TlaCoreAPIError, match="@tla.kernel"):
        tla.scatter(None, None, None)


def test_mlir_verifier_rejects_signed_indices(compiler_tlair):
    text = _capture(compiler_tlair).replace("64xui32", "64xi32")
    with ir.Context() as context:
        context.load_all_available_dialects()
        with pytest.raises(ir.MLIRError, match="indices must be unsigned"):
            ir.Module.parse(text)


def test_mlir_verifier_accepts_lowered_function():
    with ir.Context() as context:
        context.load_all_available_dialects()
        module = ir.Module.parse("""
!dst = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 32>>
module {
  func.func @lowered(%dst: !dst, %src: !tla.vector<64xf32>, %idx: !tla.vector<64xui32>) {
    tla.scatter %dst, %src, %idx : !dst, !tla.vector<64xf32>, !tla.vector<64xui32>
    return
  }
}
""")
        assert module.operation.verify()
