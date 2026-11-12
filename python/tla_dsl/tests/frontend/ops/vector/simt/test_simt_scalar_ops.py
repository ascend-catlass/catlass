"""Frontend dispatch for the SIMT scalar arithmetic ops.

Inside a ``tla.vec.func`` with ``mode="simt"`` the Python operators emit
``tla.simt_*`` ops; outside one they keep emitting plain ``arith``. These tests
pin that split, which is easy to break by touching ``_binary_op`` in
``base_dsl/typing.py`` -- and hard to notice, because both forms lower to the
same arith op in the end and every kernel still passes.

The lowering itself (``tla.simt_* -> arith.*``) is covered by the lit test
``lit/vector/simt/simt-scalar-ops-lowering.mlir``.
"""

from __future__ import annotations

import pytest
from catlass._mlir.dialects import tla as tla_ops_gen

from catlass import tla
from catlass.base_dsl import BaseDSL
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def _simt_arith_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=64):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, 64, nthreads):
                s = src[i] + src[i]
                t = s - src[i]
                u = t * src[i]
                v = u / src[i]
                m = tla.sqrt(abs(v)) + tla.exp(v) + tla.log(v)
                if src[i] > v:
                    m = m + v
                m = m.to(tla.Float16).to(tla.Float32)
                m = tla.where(m > v, m, v)
                m = tla.max(m, v)
                m = tla.min(m, src[i])
                dst[i] = m + v ** tla.Float32(2.0)


@tla.kernel
def _simt_int_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """Integer per-thread ops: `//` must reach tla.simt_div like the rest."""
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=64):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, 64, nthreads):
                v = (src[i] + src[i]) * src[i]
                v = v // src[i]
                v = tla.max(v, tla.Int32(0))
                dst[i] = tla.where(v > src[i], v, src[i])


@tla.kernel
def _simd_arith_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """Same arithmetic, but on the scalar path outside any SIMT region."""
    with tla.vector():
        s = src[0] + src[0]
        t = s - src[0]
        u = t * src[0]
        v = u / src[0]
        dst[0] = v + abs(v) + v ** tla.Float32(2.0)


def _tensors():
    def _fake():
        return make_fake_tensor(
            tla.Float32,
            (64,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(64,),
        )

    return _fake(), _fake()


def test_generated_bindings_exist_for_simt_scalar_ops() -> None:
    for symbol in (
        "simt_add", "simt_sub", "simt_mul", "simt_div", "simt_pow",
        "simt_sqrt", "simt_exp", "simt_abs", "simt_log",
        "simt_max", "simt_min", "simt_cmp", "simt_where", "simt_cast",
    ):
        assert hasattr(tla_ops_gen, symbol), f"missing generated binding: {symbol}"


def test_simt_region_emits_simt_arithmetic_ops() -> None:
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("tla.simt_add", "tla.simt_sub", "tla.simt_mul", "tla.simt_div"):
        assert op_name in mlir, f"{op_name} not emitted inside a SIMT region"


def test_simt_region_emits_simt_math_ops() -> None:
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in (
        "tla.simt_sqrt",
        "tla.simt_exp",
        "tla.simt_abs",
        "tla.simt_log",
        "tla.simt_cmp",
        "tla.simt_where",
        "tla.simt_cast",
        "tla.simt_max",
        "tla.simt_min",
        "tla.simt_pow",
    ):
        assert op_name in mlir, f"{op_name} not emitted inside a SIMT region"


def test_simt_region_does_not_emit_raw_math() -> None:
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("math.sqrt", "math.exp", "math.absf", "math.powf", "math.log"):
        assert op_name not in mlir, f"{op_name} leaked into a SIMT region"


def test_simt_region_does_not_emit_raw_arith() -> None:
    """The point of the ops: a SIMT body is TLA IR, not raw arith."""
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                    "arith.cmpf"):
        assert op_name not in mlir, f"{op_name} leaked into a SIMT region"


def test_simt_region_emits_simt_element_access() -> None:
    src, dst = _tensors()
    mlir = _simt_arith_kernel.dump_mlir(type_args=(src, dst))
    assert "tla.simt_load" in mlir
    assert "tla.simt_store" in mlir
    assert "tla.scalar_load" not in mlir
    assert "tla.scalar_store" not in mlir


def test_outside_a_simt_region_the_operators_stay_arith() -> None:
    """The dispatch is region-scoped: the general scalar path is unchanged."""
    src, dst = _tensors()
    mlir = _simd_arith_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                    "math.absf", "math.powf"):
        assert op_name in mlir, f"{op_name} should still be emitted outside a SIMT region"
    for op_name in ("tla.simt_add", "tla.simt_sub", "tla.simt_mul", "tla.simt_div",
                    "tla.simt_abs", "tla.simt_pow"):
        assert op_name not in mlir, f"{op_name} emitted outside a SIMT region"
    assert "tla.scalar_load" in mlir
    assert "tla.scalar_store" in mlir


def _int_tensors():
    def _fake():
        return make_fake_tensor(
            tla.Int32, (64,), (1,),
            addrspace=tla.AddressSpace.gm, origin_shape=(64,),
        )

    return _fake(), _fake()


def test_simt_region_emits_simt_ops_for_integers() -> None:
    """Integer `//`, max and where go through the TLA ops, not raw arith."""
    src, dst = _int_tensors()
    mlir = _simt_int_kernel.dump_mlir(type_args=(src, dst))
    for op_name in ("tla.simt_add", "tla.simt_mul", "tla.simt_div",
                    "tla.simt_max", "tla.simt_cmp", "tla.simt_where"):
        assert op_name in mlir, f"{op_name} not emitted for an integer SIMT region"
    for op_name in ("arith.divsi", "arith.addi", "arith.muli"):
        assert op_name not in mlir, f"{op_name} leaked into an integer SIMT region"


@tla.kernel
def _simt_integer_signedness_kernel(
    src: tla.Tensor, dst: tla.Tensor, dtype: tla.Constexpr[type]
) -> None:
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simt", thread_block_dim=64):
            tid, _, _ = tla.arch.thread_idx()
            value = dtype(src[tid].ir_value())
            high_bit = dtype(0x80000000 if dtype is tla.UInt32 else -0x80000000)
            bits = (value + dtype(1)) & high_bit
            dst[tid] = bits.to(tla.Int64)
            dst[tid + 64] = (value > dtype(0)).to(tla.Int64)


@pytest.mark.parametrize("dtype", [tla.Int32, tla.UInt32])
def test_simt_integer_signedness_survives_constructor_and_operations(dtype) -> None:
    src, _ = _int_tensors()
    dst = make_fake_tensor(tla.Int64, (128,), (1,))
    kernel = _simt_integer_signedness_kernel
    lowered = BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=(src, dst, dtype),
        location=kernel.decorator_location,
    )
    assert lowered.module.operation.verify()
    mlir = lowered.asm()
    assert "tla.simt_add" in mlir
    assert "arith.andi" in mlir
    assert ("isUnsigned" in mlir) == (dtype is tla.UInt32)
    extension = "extsi" if dtype is tla.Int32 else "extui"
    assert any(
        f'tla.simt_cast "{extension}"' in line and ": i32 to i64" in line
        for line in mlir.splitlines()
    )


@pytest.mark.parametrize("dtype", [tla.UInt8, tla.UInt16, tla.UInt32, tla.UInt64])
def test_simt_unsigned_tensor_boundaries(dtype):
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tid, _, _ = tla.arch.thread_idx()
                dst[tid] = src[tid] + dtype(1)
                dst[tid] = dtype(1)
                dst[tid] = 1

    tensor = make_fake_tensor(dtype, (32,), (1,), layout_tag=tla.arch.RowMajor)
    ir = kernel.dump_mlir(type_args=(tensor, tensor))
    assert "tla.simt_load" in ir
    assert "tla.simt_store" in ir
    assert "builtin.unrealized_conversion_cast" in ir
    assert "tla.simt_add" in ir


def test_simt_unsigned_store_rejects_implicit_signed_conversion():
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                dst[0] = src[0]

    src = make_fake_tensor(tla.Int32, (32,), (1,), layout_tag=tla.arch.RowMajor)
    dst = make_fake_tensor(tla.UInt32, (32,), (1,), layout_tag=tla.arch.RowMajor)
    with pytest.raises(Exception, match="cast explicitly"):
        kernel.dump_mlir(type_args=(src, dst))


@pytest.mark.parametrize("dtype", [tla.UInt8, tla.UInt16, tla.UInt32, tla.UInt64])
def test_simt_unsigned_capture_dominance(dtype):
    @tla.kernel
    def kernel(arg: object, src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():
            captured = src[0]
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tid, _, _ = tla.arch.thread_idx()
                if tid < 16:
                    dst[tid] = captured + dtype(1)
                else:
                    dst[tid] = arg + dtype(1)
            with tla.vec.func(mode="simt", thread_block_dim=32):
                dst[0] = tla.where(arg > dtype(0), captured, arg)
                dst[1] = (captured // dtype(3)) + (arg % dtype(3))
                dst[2] = arg >> dtype(1)
            dst[0] = captured

    tensor = make_fake_tensor(dtype, (32,), (1,), layout_tag=tla.arch.RowMajor)
    ir = kernel.dump_mlir(type_args=(dtype(0), tensor, tensor))
    assert "builtin.module" not in ir  # Invalid IR falls back to generic printing.
    assert "tla.simt_add" in ir
    assert "arith.divui" in ir
    assert "arith.remui" in ir
