from __future__ import annotations

from collections.abc import Callable

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _vector_tensor(dtype: object = tla.Float32) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def vector_scalar_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = tile.load()
            mask = tla.create_mask(pattern=tla.mask.H)
            _ = tla.add(reg, 1.0)
            _ = tla.sub(reg, 2.0)
            _ = tla.mul(3.0, reg)
            _ = tla.max(reg, 5.0)
            _ = tla.min(6.0, reg)
            _ = tla.div(reg, 4.0, mask=mask)


@tla.kernel
def bf16_scalar_literal_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.add(tile.load(), 1.0)


@tla.kernel
def invalid_scalar_lhs_sub_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = tile.load()
            _ = tla.sub(3.0, reg)


@tla.kernel
def invalid_integer_scalar_fraction_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = tile.load()
            _ = tla.add(reg, 1.9)


@tla.kernel
def vector_scalar_numeric_ssa_kernel(
    lhs: tla.Tensor, scalar_buf: tla.Tensor
) -> None:
    """Dynamic Numeric SSA as vector–scalar rhs."""
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = tile.load()
            s = scalar_buf[0]
            _ = tla.add(reg, s)
            _ = tla.mul(s, reg)
            _ = tla.cmp(reg, s, mode="gt")


def test_vector_scalar_accepts_numeric_ssa() -> None:
    scalar_buf = make_fake_tensor(
                     tla.Float32,
                     (1,),
                     (1,),
                     addrspace=tla.AddressSpace.ub,
                     origin_shape=(1,),
                     layout_tag=tla.arch.RowMajor,
                 )
    mlir = vector_scalar_numeric_ssa_kernel.dump_mlir(
        type_args=(_vector_tensor(), scalar_buf)
    )
    assert "tla.adds" in mlir
    assert "tla.muls" in mlir
    assert "tla.cmp" in mlir
    # Dynamic scalar must not be forced through a fresh host arith.constant alone;
    # scalar load feeds the vector–scalar ops.
    assert "tla.scalar_load" in mlir or "memref.load" in mlir or "tla.load" in mlir


@tla.kernel
def vector_scalar_typed_rhs(
    lhs: tla.Tensor,
    scalar_buf: tla.Tensor,
    op: tla.Constexpr[Callable],
    use_literal: tla.Constexpr[bool],
) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            if tla.const_expr(use_literal):
                op(tile.load(), tla.Float16(0.5))
            else:
                op(tile.load(), scalar_buf[0])


@pytest.mark.parametrize("op", [tla.add, tla.sub, tla.mul, tla.div, tla.max, tla.min])
@pytest.mark.parametrize("dtype", [tla.Int32, tla.UInt16, tla.Float32])
@pytest.mark.parametrize("use_literal", [False, True])
def test_vector_scalar_arithmetic_requires_matching_typed_scalar(op, dtype, use_literal):
    with pytest.raises(
        tla.TlaCoreAPIError, match=f"rhs has type f16, expected {dtype.dtype}"
    ):
        vector_scalar_typed_rhs.dump_mlir(
            type_args=(
                _vector_tensor(dtype),
                _vector_tensor(tla.Float16),
                op,
                use_literal,
            )
        )


@pytest.mark.parametrize(
    ("op", "dtype", "scalar_dtype"),
    [
        (tla.add, tla.Int32, tla.Int16),
        (tla.max, tla.Int64, tla.Int16),
        (tla.add, tla.UInt64, tla.UInt32),
        (tla.add, tla.BFloat16, tla.Float32),
    ],
)
def test_vector_scalar_arithmetic_rejects_mismatched_integer_widths_and_bf16(
    op, dtype, scalar_dtype
):
    error = f"rhs has type {scalar_dtype.dtype}, expected {dtype.dtype}"
    with pytest.raises(tla.TlaCoreAPIError, match=error):
        vector_scalar_typed_rhs.dump_mlir(
            type_args=(
                _vector_tensor(dtype),
                _vector_tensor(scalar_dtype),
                op,
                False,
            )
        )


def test_vector_scalar_symbols_are_exported() -> None:
    for symbol in ("add", "sub", "mul", "max", "min", "div"):
        assert callable(getattr(tla, symbol))
    for symbol in (
        "adds",
        "subs",
        "muls",
        "maxs",
        "mins",
        "divs",
        "vadd",
        "vsub",
        "vmul",
        "vdiv",
        "vadds",
        "vsubs",
        "vmuls",
        "vdivs",
    ):
        assert not hasattr(tla, symbol)


def test_vector_scalar_dispatch_and_masking() -> None:
    mlir = vector_scalar_kernel.dump_mlir(type_args=(_vector_tensor(),))

    for op_name in ("adds", "subs", "muls", "maxs", "mins", "divs"):
        assert mlir.count(f"tla.{op_name}") == 1
    divs_line = next(line for line in mlir.splitlines() if "tla.divs" in line)
    assert "mask" in divs_line
    assert "pass_thru" not in mlir


def test_bf16_scalar_literal_uses_bf16_constant(
    compiler_tlair: Callable[..., str],
) -> None:
    mlir = compiler_tlair(bf16_scalar_literal_kernel, type_args=(_vector_tensor(tla.BFloat16),))

    assert any("arith.constant" in line and "bf16" in line for line in mlir.splitlines())
    assert "tla.adds" in mlir


def test_noncommutative_vector_scalar_rejects_scalar_lhs() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="invalid argument 'lhs'"):
        invalid_scalar_lhs_sub_kernel.dump_mlir(type_args=(_vector_tensor(),))


def test_integer_vector_scalar_rejects_fractional_literal() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="expected integer scalar"):
        invalid_integer_scalar_fraction_kernel.dump_mlir(
            type_args=(_vector_tensor(tla.Int32),)
        )


# --- unsigned max/min signed-twin guard (u16 65535 vs 1) -------------------
# Device behavior is covered by lit/vector/simd/unsigned-casts.mlir
# (ave.hir.vumax / vumin). Keep a card-free frontend check here: ui16 must
# survive dump_mlir so lowering can pick the unsigned AVE ops.


_U16_N = 128


@tla.kernel
def u16_vec_max_min_kernel(a: tla.Tensor, b: tla.Tensor, out: tla.Tensor) -> None:
    ta = tla.tile_view(a, tla.make_shape(_U16_N), tla.make_coord(0))
    tb = tla.tile_view(b, tla.make_shape(_U16_N), tla.make_coord(0))
    to = tla.tile_view(out, tla.make_shape(_U16_N), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mx = tla.max(ta.load(), tb.load())
            to.store(tla.min(mx, tb.load()))


def test_u16_max_min_keeps_unsigned_vector_type() -> None:
    """Signed twin would flip max(65535,1); frontend must keep ui16 on max/min."""
    tensor = make_fake_tensor(
        tla.UInt16,
        (_U16_N,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(_U16_N,),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = u16_vec_max_min_kernel.dump_mlir(type_args=(tensor, tensor, tensor))
    assert "tla.max" in mlir
    assert "tla.min" in mlir
    assert "!tla.vector<128xui16>" in mlir
