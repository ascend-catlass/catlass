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
        (tla.Int8, "i8"),
        (tla.UInt8, "ui8"),
        (tla.Float8E4M3FN, "f8E4M3FN"),
        (tla.Float8E5M2, "f8E5M2"),
        (tla.HiFloat8, "!tla.hif8"),
        (tla.Float8E8M0, "!tla.f8e8m0"),
        (tla.Float4E2M1x2, "!tla.f4e2m1x2"),
        (tla.Float4E1M2x2, "!tla.f4e1m2x2"),
    ],
)
@pytest.mark.parametrize("source_view", [1, 512])
def test_load_dintlv_b8_returns_two_full_vectors(
    compiler_tlair: Any, dtype: type[tla.Numeric], element: str, source_view: int
):
    ir = compiler_tlair(
        _load_dintlv_b8(source_view),
        type_args=(
            _ub_tensor(dtype, 512),
            _ub_tensor(dtype, 256),
            _ub_tensor(dtype, 256),
        ),
    )
    (load_line,) = [line for line in ir.splitlines() if '"tla.load"' in line]
    assert "#tla.load_dist<dintlv_b8>" in load_line
    assert load_line.count(f"!tla.vector<256x{element}>") == 2
    assert ir.count('"tla.store"') == 2


@pytest.mark.parametrize(
    "dtype",
    [
        tla.Bool,
        tla.Float4E2M1,
        tla.Float4E1M2,
        tla.Int16,
        tla.UInt16,
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


def test_e8m0_vector_load_requires_dintlv_b8(compiler_tlair: Any):
    @tla.kernel
    def ordinary(source: tla.Tensor):
        with tla.vector():  # noqa: SIM117 - vector regions require nested syntax.
            with tla.vec.func(mode="simd"):
                source.load()

    with pytest.raises(
        TlaLoweringError, match="E8M0 vector load requires DIST_DINTLV_B8"
    ):
        compiler_tlair(ordinary, type_args=(_ub_tensor(tla.Float8E8M0, 256),))


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


# compiler_tlair uses execution lowering: NotImplementedError is wrapped as
# UnsupportedExecutionLowering; TlaLoweringError has a separate catch branch.
def test_load_dintlv_b8_rejects_post_update(compiler_tlair: Any):
    @tla.kernel
    def update(source: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                source.load(
                    NormalLoadParams(
                        load_dist=LoadDist.DIST_DINTLV_B8,
                        post_mode=PostMode.POST_MODE_UPDATE,
                    )
                )

    with pytest.raises(
        UnsupportedExecutionLowering,
        match="post_mode=update with load_dist='dintlv_b8' is not supported yet",
    ):
        compiler_tlair(update, type_args=(_ub_tensor(tla.Int8, 512),))


def test_load_dintlv_b8_is_not_a_mask_distribution(compiler_tlair: Any):
    @tla.kernel
    def mask_load(source: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                source.load(MaskLoadParams(load_dist=LoadDist.DIST_DINTLV_B8))

    with pytest.raises(UnsupportedExecutionLowering, match="unsupported MaskLoadDist"):
        compiler_tlair(mask_load, type_args=(_ub_tensor(tla.Int8, 512),))
