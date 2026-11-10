from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.types import TlaVectorSSATypeDescriptor


SOURCE = r"""
#include "kernel_operator.h"
extern "C" [aicore] __attribute__((always_inline))
vector_f32 reg_identity(vector_f32 x) { return x; }
extern "C" [aicore] __attribute__((always_inline))
vector_f32 reg_from_scalar(float x) {
    using namespace AscendC::Reg;
    RegTensor<float> result;
    auto mask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(result, x, mask);
    return result.reg[0];
}
extern "C" [aicore] __attribute__((always_inline))
void reg_sink(vector_f32 x) { (void)x; }
"""


@tla.extern(source=SOURCE)
def reg_identity(x: tla.VectorSSA[tla.Float32]) -> tla.VectorSSA[tla.Float32]: ...


@tla.extern(source=SOURCE)
def reg_from_scalar(x: tla.Float32) -> tla.VectorSSA[tla.Float32]: ...


@tla.extern(source=SOURCE)
def reg_sink(x: tla.VectorSSA[tla.Float32]) -> None: ...


@tla.kernel
def register_kernel(dtype: tla.Constexpr, lanes: tla.Constexpr) -> None:
    tensor = tla.make_tensor(
        tla.allocate(lanes, dtype, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(lanes), tla.make_stride(1)),
    )
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = reg_identity(tensor.load())
            reg_sink(value)
            tensor.store(value + value)


@tla.kernel
def outside_vf_kernel(x: tla.Float32) -> None:
    with tla.vector():
        reg_from_scalar(x)


@tla.kernel
def simt_kernel(x: tla.Float32) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=64):
            reg_from_scalar(x)


def test_register_annotations_and_return_value() -> None:
    annotation = tla.VectorSSA[tla.Float32]
    assert annotation == TlaVectorSSATypeDescriptor(64, "f32")
    assert reg_identity.arg_types == (annotation,)
    assert reg_identity.return_type == annotation
    assert reg_sink.return_type is None
    ir = register_kernel.dump_mlir(type_args=(tla.Float32, 64))
    assert "tla.call_extern @reg_identity" in ir
    assert "(!tla.vector<64xf32>) -> !tla.vector<64xf32>" in ir
    assert "tla.call_extern @reg_sink" in ir
    assert "tla.add" in ir


@pytest.mark.parametrize("dtype", [tla.Bool, tla.Float8E4M3FN, tla.Float8E5M2, tla.Numeric])
def test_unsupported_register_annotation(dtype) -> None:
    with pytest.raises(TypeError, match="annotations support"):
        tla.VectorSSA[dtype]


@pytest.mark.parametrize("dtype,lanes", [
    (tla.Float16, 64), (tla.Float16, 128), (tla.BFloat16, 128),
    (tla.Int32, 64), (tla.Float32, 32),
])
def test_register_argument_type_and_valid_lanes(dtype, lanes) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="64 valid lanes"):
        register_kernel.dump_mlir(type_args=(dtype, lanes))


@pytest.mark.parametrize("kernel,diagnostic", [
    (outside_vf_kernel, "must be nested inside tla.vec.func"),
    (simt_kernel, "SIMT tla.vec.func"),
])
def test_register_call_scope(kernel, diagnostic) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=diagnostic):
        kernel.dump_mlir(type_args=(tla.Float32(1),))


def test_extern_rejects_scalar_and_multiple_returns() -> None:
    def scalar_return() -> tla.Float32: ...
    def multiple_returns() -> tuple[tla.VectorSSA[tla.Float32], tla.VectorSSA[tla.Float32]]: ...
    for fn in (scalar_return, multiple_returns):
        with pytest.raises(TypeError, match=r"must be None or VectorSSA\[dtype\]"):
            tla.extern(source=SOURCE)(fn)
