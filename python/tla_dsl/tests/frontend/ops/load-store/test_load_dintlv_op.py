from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError
from catlass.params import LoadDist, NormalLoadParams
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
