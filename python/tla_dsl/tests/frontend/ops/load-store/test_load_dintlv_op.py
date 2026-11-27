from typing import Any

import pytest

import catlass.tla as tla
from catlass.core_api import VectorSSA
from catlass.execution_lowering import TlaLoweringError, UnsupportedExecutionLowering
from catlass.params import LoadDist, MaskLoadParams, NormalLoadParams, PostMode
from catlass.tla.runtime import make_fake_tensor


def _ub_tensor(dtype: type[tla.Numeric], extent: int) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def load_dintlv_b16(source: tla.Tensor, even: tla.Tensor, odd: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            # The view supplies the address. The unmasked transfer reads
            # elements [128, 384), within the 512-element backing tensor.
            source_tile = tla.tile_view(source, tla.make_shape(128), tla.make_coord(1))
            even_register, odd_register = source_tile.load(
                NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B16)
            )
            even.store(even_register)
            odd.store(odd_register)


@pytest.mark.parametrize(
    ("dtype", "element"),
    [
        (tla.Float16, "f16"),
        (tla.BFloat16, "bf16"),
        (tla.Int16, "i16"),
        (tla.UInt16, "ui16"),
    ],
)
def test_load_dintlv_b16_returns_two_full_vectors(
    compiler_tlair: Any, dtype: type[tla.Numeric], element: str
):
    ir = compiler_tlair(
        load_dintlv_b16,
        type_args=(
            _ub_tensor(dtype, 512),
            _ub_tensor(dtype, 128),
            _ub_tensor(dtype, 128),
        ),
    )
    (load_line,) = [line for line in ir.splitlines() if "tla.load" in line]
    assert "#tla.load_dist<dintlv_b16>" in load_line
    assert load_line.count(f"!tla.vector<128x{element}>") == 2
    assert ir.count('"tla.store"') == 2


@pytest.mark.parametrize("dtype", [tla.Int8, tla.Int32, tla.Float32])
def test_load_dintlv_b16_rejects_other_element_widths(
    compiler_tlair: Any, dtype: type[tla.Numeric]
):
    with pytest.raises(TlaLoweringError, match="DIST_DINTLV_B16 requires a 2-byte"):
        compiler_tlair(
            load_dintlv_b16,
            type_args=(
                _ub_tensor(dtype, 512),
                _ub_tensor(dtype, 128),
                _ub_tensor(dtype, 128),
            ),
        )


def _load_dintlv_b8(source_view: int):
    @tla.kernel
    def dual(source: tla.Tensor, even: tla.Tensor, odd: tla.Tensor):
        source_tile = tla.tile_view(
            source, tla.make_shape(source_view), tla.make_coord(0)
        )
        with tla.vector():
            with tla.vec.func(mode="simd"):
                pair = source_tile.load(
                    NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B8)
                )
                assert isinstance(pair, tuple)
                even_register, odd_register = pair
                assert isinstance(even_register, VectorSSA)
                assert isinstance(odd_register, VectorSSA)
                even.store(even_register)
                odd.store(odd_register)

    return dual


@pytest.mark.parametrize(
    ("dtype", "element"),
    [
        (tla.Bool, "i1"),
        (tla.Int8, "i8"),
        (tla.UInt8, "ui8"),
        (tla.UInt16, "ui16"),
        (tla.Float8E4M3FN, "f8E4M3FN"),
        (tla.Float8E5M2, "f8E5M2"),
        (tla.HiFloat8, "!tla.hif8"),
        (tla.Float8E8M0, "!tla.f8e8m0"),
        (tla.Float4E2M1x2, "!tla.f4e2m1x2"),
        (tla.Float4E1M2x2, "!tla.f4e1m2x2"),
    ],
)
@pytest.mark.parametrize(
    "whole_tile", [False, True], ids=["single-element", "full-tile"]
)
def test_load_dintlv_b8_returns_two_full_vectors(
    compiler_tlair: Any, dtype: type[tla.Numeric], element: str, whole_tile: bool
):
    lanes = 128 if dtype is tla.UInt16 else 256
    ir = compiler_tlair(
        _load_dintlv_b8(2 * lanes if whole_tile else 1),
        type_args=(
            _ub_tensor(dtype, 2 * lanes),
            _ub_tensor(dtype, lanes),
            _ub_tensor(dtype, lanes),
        ),
    )
    (load_line,) = [line for line in ir.splitlines() if '"tla.load"' in line]
    assert "#tla.load_dist<dintlv_b8>" in load_line
    assert load_line.count(f"!tla.vector<{lanes}x{element}>") == 2
    assert ir.count('"tla.store"') == 2


@pytest.mark.parametrize(
    "dtype",
    [
        tla.Float4E2M1,
        tla.Float4E1M2,
        tla.Int16,
        tla.Float16,
        tla.UInt32,
        tla.Float32,
    ],
)
def test_load_dintlv_b8_rejects_unsupported_types(
    compiler_tlair: Any, dtype: type[tla.Numeric]
):
    with pytest.raises(TlaLoweringError, match="DIST_DINTLV_B8 requires"):
        compiler_tlair(
            _load_dintlv_b8(1),
            type_args=(
                _ub_tensor(dtype, 512),
                _ub_tensor(dtype, 256),
                _ub_tensor(dtype, 256),
            ),
        )


def test_e8m0_vector_norm_load_emits(compiler_tlair: Any):
    """E8M0 is AscendC b8 move-only; DIST_NORM (and DINTLV_B8) are both transport."""

    @tla.kernel
    def ordinary(source: tla.Tensor):
        with tla.vector():  # noqa: SIM117 - vector regions require nested syntax.
            with tla.vec.func(mode="simd"):
                source.load()

    mlir = compiler_tlair(ordinary, type_args=(_ub_tensor(tla.Float8E8M0, 256),))
    assert '"tla.load"' in mlir


def test_load_dintlv_b8_requires_ub(compiler_tlair: Any):
    source = make_fake_tensor(
        tla.Int8,
        (512,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(512,),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(tla.TlaCoreAPIError, match="expected addrspace ub, got gm"):
        compiler_tlair(
            _load_dintlv_b8(512),
            type_args=(source, _ub_tensor(tla.Int8, 256), _ub_tensor(tla.Int8, 256)),
        )


@pytest.mark.parametrize(
    ("dtype", "stride", "error"),
    [
        (tla.Int8, 0, None),
        (tla.Int8, 32, None),
        (tla.Int8, 31, "32-byte aligned"),
        (tla.UInt16, 16, None),
        (tla.UInt16, 8, "32-byte aligned"),
    ],
)
def test_load_dintlv_b8_post_update(
    compiler_tlair: Any, dtype: type[tla.Numeric], stride: int, error: str | None
):
    @tla.kernel
    def update(source: tla.Tensor, dest: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                pair, _next_source = source.load(
                    NormalLoadParams(
                        load_dist=LoadDist.DIST_DINTLV_B8,
                        post_mode=PostMode.POST_MODE_UPDATE,
                        post_update_stride=stride,
                    )
                )
                _, odd = pair
                dest.store(odd)

    lanes = 128 if dtype is tla.UInt16 else 256
    type_args = (_ub_tensor(dtype, 2 * lanes), _ub_tensor(dtype, lanes))
    if error is not None:
        with pytest.raises(TlaLoweringError, match=error):
            compiler_tlair(update, type_args=type_args)
    else:
        mlir = compiler_tlair(update, type_args=type_args)
        assert '"tla.load_update"' in mlir
        assert "#tla.load_dist<dintlv_b8>" in mlir
        element = "ui16" if dtype is tla.UInt16 else "i8"
        vector = f"!tla.vector<{lanes}x{element}>"
        assert f"-> ({vector}, {vector}, !tla.ptr<{element}, ub" in mlir


def test_load_dintlv_b8_is_not_a_mask_distribution(compiler_tlair: Any):
    @tla.kernel
    def mask_load(source: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                source.load(MaskLoadParams(load_dist=LoadDist.DIST_DINTLV_B8))

    with pytest.raises(UnsupportedExecutionLowering, match="unsupported MaskLoadDist"):
        compiler_tlair(mask_load, type_args=(_ub_tensor(tla.Int8, 512),))
