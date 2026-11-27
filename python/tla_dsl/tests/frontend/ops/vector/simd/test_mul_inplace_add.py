from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor
from catlass.core_api import VectorSSA


@tla.kernel
def _compute(
    accumulator: tla.Tensor,
    left: tla.Tensor,
    right: tla.Tensor,
    lanes: tla.Constexpr[int],
    dtype: tla.Constexpr[type],
    masked: tla.Constexpr[bool],
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            acc = tla.tile_view(
                accumulator, tla.make_shape(lanes), tla.make_coord(0)
            ).load()
            lhs = tla.tile_view(left, tla.make_shape(lanes), tla.make_coord(0)).load()
            rhs = tla.tile_view(right, tla.make_shape(lanes), tla.make_coord(0)).load()
            if tla.const_expr(masked):
                active = tla.create_mask(pattern=tla.mask.H, dtype=dtype)
                result = tla.arch.mul_inplace_add(acc, lhs, rhs, mask=active)
            else:
                result = tla.arch.mul_inplace_add(acc, lhs, rhs)
            assert isinstance(result, VectorSSA)


def _tensor(dtype, lanes=64):
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        tla.Float16,
        tla.BFloat16,
        tla.Float32,
        tla.Int16,
        tla.UInt16,
        tla.Int32,
        tla.UInt32,
        tla.Int64,
        tla.UInt64,
    ],
)
@pytest.mark.parametrize("masked", [False, True])
def test_emits_mul_inplace_add(dtype, masked):
    lanes = min(64, 2048 // dtype.width)
    source = _tensor(dtype, lanes)
    text = _compute.dump_mlir(type_args=(source, source, source, lanes, dtype, masked))
    (line,) = [line for line in text.splitlines() if "tla.mul_inplace_add" in line]
    assert (" mask " in line) is masked
    assert "tla.mul " not in text and "tla.add " not in text


@pytest.mark.parametrize("dtype", [tla.Int8, tla.Float8E4M3FN])
def test_rejects_unsupported_types(dtype):
    source = _tensor(dtype, 32)
    message = (
        rf"unsupported mul_inplace_add element type {dtype.dtype}; "
        r"supported element types are bf16, f16, f32, i16, i32, i64, u16, u32, u64"
    )
    with pytest.raises(tla.TlaCoreAPIError, match=message):
        _ = _compute.dump_mlir(type_args=(source, source, source, 32, dtype, False))


def test_rejects_mismatched_types():
    with pytest.raises(tla.TlaCoreAPIError, match="same type and valid lane count"):
        _ = _compute.dump_mlir(
            type_args=(
                _tensor(tla.Float32),
                _tensor(tla.Int32),
                _tensor(tla.Float32),
                64,
                tla.Float32,
                False,
            )
        )


@tla.kernel
def _bad_lanes(source: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            full = tla.tile_view(source, tla.make_shape(64), tla.make_coord(0)).load()
            short = tla.tile_view(source, tla.make_shape(13), tla.make_coord(0)).load()
            _ = tla.arch.mul_inplace_add(full, short, full)


def test_rejects_mismatched_valid_lanes():
    with pytest.raises(tla.TlaCoreAPIError, match="same type and valid lane count"):
        _ = _bad_lanes.dump_mlir(type_args=(_tensor(tla.Float32),))


def test_rejects_wrong_mask_width():
    source = _tensor(tla.Float32)
    with pytest.raises(tla.TlaCoreAPIError, match="mask"):
        _ = _compute.dump_mlir(
            type_args=(source, source, source, 64, tla.Float16, True)
        )


@tla.kernel
def _bad_scalar(mode: tla.Constexpr) -> None:
    with tla.vector():
        with tla.vec.func(mode=mode):
            _ = tla.arch.mul_inplace_add(tla.Float32(1), tla.Float32(2), tla.Float32(3))


@pytest.mark.parametrize(
    ("mode", "error"), [("simt", "simd"), ("simd", "expected vector_ssa")]
)
def test_rejects_scalar_inputs(mode, error):
    with pytest.raises(
        tla.TlaCoreAPIError, match=rf"tla\.arch\.mul_inplace_add.*{error}"
    ):
        _ = _bad_scalar.dump_mlir(type_args=(mode,))
