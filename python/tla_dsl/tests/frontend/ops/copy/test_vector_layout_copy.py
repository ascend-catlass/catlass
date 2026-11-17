"""Frontend guard for the VectorLayout layout tag.

``VectorLayout`` is the one-dimensional, unit-stride tile shape: it is the tag that
matches ``tla::detail::isVector`` on the device side, so the VectorLayout
TileCopyTla specializations serve it. The flat GM -> L1 and L1 -> biasbuf copy
routes are keyed on it, which is why a 2-D tile under this tag is refused rather
than silently projected onto its leading index.
"""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _vector_tensor(dtype, shape, stride, addrspace):
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        addrspace=addrspace,
        origin_shape=shape,
        layout_tag=tla.arch.VectorLayout,
    )


def _gm(dtype, shape, stride):
    return _vector_tensor(dtype, shape, stride, tla.AddressSpace.gm)


def _l1(dtype, shape, stride):
    return _vector_tensor(dtype, shape, stride, tla.AddressSpace.l1)


def _bias(dtype, shape, stride):
    return _vector_tensor(dtype, shape, stride, tla.AddressSpace.biasbuf)


@tla.kernel
def gm_to_l1_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


@tla.kernel
def l1_to_bias_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


# --- GM VectorLayout -> L1 VectorLayout ------------------------------------


def test_gm_vector_to_l1_vector_is_accepted() -> None:
    src = _gm(tla.Float32, (256,), (1,))
    dst = _l1(tla.Float32, (256,), (1,))
    mlir = gm_to_l1_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.copy" in mlir
    assert "VectorLayout" in mlir


def test_gm_vector_to_l1_vector_accepts_int32() -> None:
    src = _gm(tla.Int32, (256,), (1,))
    dst = _l1(tla.Int32, (256,), (1,))
    mlir = gm_to_l1_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.copy" in mlir


@pytest.mark.parametrize(
    "dtype",
    [tla.UInt64, tla.Int64, tla.Float64],
    ids=["u64", "i64", "f64"],
)
def test_gm_vector_to_l1_vector_accepts_b64(dtype) -> None:
    """Flat VectorLayout GM→L1 stages 64-bit elements (u64/i64/f64)."""
    src = _gm(dtype, (256,), (1,))
    dst = _l1(dtype, (256,), (1,))
    mlir = gm_to_l1_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.copy" in mlir
    assert "VectorLayout" in mlir


@pytest.mark.parametrize(
    "dtype",
    [tla.Int8, tla.Float16, tla.BFloat16],
    ids=["i8", "f16", "bf16"],
)
def test_gm_vector_to_l1_vector_accepts_narrow_continuous(dtype) -> None:
    """Flat VectorLayout GM→L1 also stages i8 / f16 / bf16 continuous runs."""
    src = _gm(dtype, (256,), (1,))
    dst = _l1(dtype, (256,), (1,))
    mlir = gm_to_l1_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.copy" in mlir
    assert "VectorLayout" in mlir


def test_gm_vector_to_l1_vector_refuses_two_dimensions() -> None:
    """VectorLayout is one-dimensional by definition.

    A 2-D VectorLayout tile never reaches the copy: the host tensor cannot be built,
    because ``VectorLayout`` has no 2-D form to derive a coord from.
    """
    with pytest.raises(Exception, match="coord|cannot derive"):
        _gm(tla.Float32, (1, 256), (256, 1))


# --- L1 VectorLayout -> biasbuf VectorLayout ------------------------------


def test_l1_vector_to_biasbuf_vector_is_accepted() -> None:
    src = _l1(tla.Float32, (256,), (1,))
    dst = _bias(tla.Float32, (256,), (1,))
    mlir = l1_to_bias_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.copy" in mlir
    assert "VectorLayout" in mlir


def test_l1_vector_to_biasbuf_vector_accepts_int32() -> None:
    src = _l1(tla.Int32, (256,), (1,))
    dst = _bias(tla.Int32, (256,), (1,))
    mlir = l1_to_bias_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.copy" in mlir


# --- The retired spelling --------------------------------------------------


def test_flat_row_major_copy_route_is_gone() -> None:
    """A 1-D RowMajor GM->L1 copy no longer has a route.

    The continuous-run copy is keyed on VectorLayout now, so the old RowMajor spelling
    must be refused instead of silently reaching a symbol that no longer exists.
    """
    src = make_fake_tensor(
        tla.Float32,
        (256,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(256,),
        layout_tag=tla.arch.RowMajor,
    )
    dst = make_fake_tensor(
        tla.Float32,
        (256,),
        (1,),
        addrspace=tla.AddressSpace.l1,
        origin_shape=(256,),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(Exception, match="unsupported copy route"):
        gm_to_l1_kernel.dump_mlir(type_args=(dst, src))
