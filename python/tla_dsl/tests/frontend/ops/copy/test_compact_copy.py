import pytest

from catlass import tla
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def copy_compact(
    dst: tla.Tensor, src: tla.Tensor, params: tla.Constexpr[tla.params.CopyParams]
) -> None:
    with tla.vector():
        tla.copy(dst, src, params)


def _tiles(dtype, layout):
    if layout == tla.arch.RowMajor:
        shape, ub_stride, gm_stride = (3, 8), (8, 1), (16, 1)
    else:
        shape, ub_stride, gm_stride = (24,), (1,), (1,)
    ub = make_fake_tensor(
        dtype,
        shape,
        ub_stride,
        addrspace=tla.AddressSpace.ub,
        origin_shape=shape,
        layout_tag=layout,
    )
    gm = make_fake_tensor(
        dtype,
        shape,
        gm_stride,
        addrspace=tla.AddressSpace.gm,
        origin_shape=shape,
        layout_tag=layout,
    )
    return ub, gm


@pytest.mark.parametrize("compact", (False, True))
@pytest.mark.parametrize("dtype", (tla.UInt8, tla.Int32, tla.BFloat16))
@pytest.mark.parametrize("direction", ("gm_to_ub", "ub_to_gm"))
def test_compact_copy_flag(dtype, compact, direction) -> None:
    ub, gm = _tiles(dtype, tla.arch.RowMajor)
    if direction == "gm_to_ub":
        dst, src = ub, gm
        params = tla.params.CopyGmToUbParams(compact=compact)
    else:
        dst, src = gm, ub
        params = tla.params.CopyUbToGmParams(compact=compact)
    ir = copy_compact.dump_mlir(type_args=(dst, src, params))
    copy = next(line for line in ir.splitlines() if "tla.copy" in line)
    assert ("compact" in copy) is compact


@pytest.mark.parametrize("compact", (False, True))
@pytest.mark.parametrize(
    ("params_type", "source", "destination"),
    (
        (tla.params.CopyGmToUbParams, "GM", "UB"),
        (tla.params.CopyUbToGmParams, "UB", "GM"),
    ),
)
def test_directional_copy_params_reject_reversed_route(
    compact, params_type, source, destination
) -> None:
    ub, gm = _tiles(tla.Int32, tla.arch.RowMajor)
    tiles = {"UB": ub, "GM": gm}
    params = params_type(compact=compact)
    message = f"{params_type.__name__} requires {source} -> {destination}"
    with pytest.raises(tla.TlaCoreAPIError, match=message):
        copy_compact.dump_mlir(type_args=(tiles[source], tiles[destination], params))


def test_gm_to_ub_compact_requires_row_major() -> None:
    ub, gm = _tiles(tla.Int32, tla.arch.VectorLayout)
    params = tla.params.CopyGmToUbParams(compact=True)
    with pytest.raises(tla.TlaCoreAPIError, match="compact requires RowMajor src and dst"):
        copy_compact.dump_mlir(type_args=(ub, gm, params))
