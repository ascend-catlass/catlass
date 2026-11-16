"""Frontend surface of the ``tla.arch.shuffle_sync*`` family.

The four interfaces share one op, ``tla.arch.shuffle``; what distinguishes them
is the ``mode`` attribute, so these tests pin that mapping and the argument
checks. The lowering onto the shfl intrinsics is covered by the lit test
``lit/vector/simt/simt-warp-shuffle-lowering.mlir``, and the lane-by-lane
semantics by the ``warp_shuffle`` example.
"""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

THREADS = 64


def _tensors(dtype=None):
    def _fake():
        return make_fake_tensor(
            dtype or tla.Float32,
            (THREADS,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(THREADS,),
        )

    return _fake(), _fake()


def _dump(api: str, index, width=32, dtype=None):
    """Build a one-shuffle SIMT kernel and return its MLIR.

    The interface is resolved out here, not in the kernel body: a traced body
    may not call ``getattr``, and a Python ``if`` inside the region would be
    traced as control flow rather than folded away.
    """
    fn = getattr(tla.arch, api)

    @tla.kernel
    def _kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=THREADS):
                tid, _, _ = tla.arch.thread_idx()
                dst[tid] = fn(src[tid], index, width)

    src, dst = _tensors(dtype)
    return _kernel.dump_mlir(type_args=(src, dst))


def _rejected(api: str, index, width=32, dtype=None) -> None:
    """Trace a kernel the frontend must reject.

    Returns nothing: it either propagates the rejection or fails with the MLIR
    that should never have been produced.
    """
    mlir = _dump(api, index, width, dtype)
    raise AssertionError(f"expected a rejection, got:\n{mlir}")


@pytest.mark.parametrize(
    "api,mode",
    [
        ("shuffle_sync", "idx"),
        ("shuffle_sync_up", "up"),
        ("shuffle_sync_down", "down"),
        ("shuffle_sync_bfly", "bfly"),
    ],
)
def test_each_interface_emits_its_mode(api: str, mode: str) -> None:
    mlir = _dump(api, 1)
    assert f"tla.arch.shuffle <{mode}>" in mlir


def test_width_defaults_to_the_whole_warp() -> None:
    """Called with two arguments the width is 32, which the printer elides."""

    @tla.kernel
    def _kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=THREADS):
                tid, _, _ = tla.arch.thread_idx()
                dst[tid] = tla.arch.shuffle_sync(src[tid], 0)

    src, dst = _tensors()
    mlir = _kernel.dump_mlir(type_args=(src, dst))
    assert "tla.arch.shuffle <idx>" in mlir
    assert "width" not in mlir


def test_width_is_carried_through() -> None:
    """A non-default width is printed, so it cannot be silently dropped."""
    assert "width = 16" in _dump("shuffle_sync", 0, 16)


@pytest.mark.parametrize("width", [0, 1, 3, 64])
def test_width_must_be_a_power_of_two_in_range(width: int) -> None:
    with pytest.raises(Exception, match="power of two"):
        _rejected("shuffle_sync", 0, width)


def test_negative_index_is_rejected() -> None:
    with pytest.raises(Exception, match="non-negative"):
        _rejected("shuffle_sync_up", -1)


def test_rejected_outside_a_simt_region() -> None:
    @tla.kernel
    def _kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            dst[0] = tla.arch.shuffle_sync(src[0], 0)

    src, dst = _tensors()
    with pytest.raises(Exception, match="simt"):
        _kernel.dump_mlir(type_args=(src, dst))


# An unsigned value presents itself as a signless integer of its own width
# inside a SIMT region, which is exactly what the op accepts, so all four
# unsigned types shuffle and keep their Python type on the way out. The op's
# own operand type is signless either way: signedness is not a property of a
# bit pattern being moved between registers.
@pytest.mark.parametrize(
    "api", ["shuffle_sync", "shuffle_sync_up", "shuffle_sync_down", "shuffle_sync_bfly"]
)
@pytest.mark.parametrize(
    "dtype,ir_type",
    [(tla.UInt8, "i8"), (tla.UInt16, "i16"), (tla.UInt32, "i32"), (tla.UInt64, "i64")],
)
def test_unsigned_values_shuffle_as_signless(api: str, dtype, ir_type: str) -> None:
    mlir = _dump(api, 1, 32, dtype)
    assert "tla.arch.shuffle" in mlir
    assert f": {ir_type}" in mlir


# `Bool` is the one integer type left out: it is i1, and there is no 1-bit
# shuffle to widen it to.
@pytest.mark.parametrize(
    "api", ["shuffle_sync", "shuffle_sync_up", "shuffle_sync_down", "shuffle_sync_bfly"]
)
def test_bool_values_are_rejected(api: str) -> None:
    with pytest.raises(Exception, match="Bool"):
        _rejected(api, 1, 32, tla.Bool)


# "up"/"down" take a distance and "bfly" a lane mask, and the 5-bit control
# field carries any of them exactly up to 31. From 32 the value stops meaning
# what the caller wrote -- 33 survives the mask as 1 and reads a neighbour --
# so that is where the rejection belongs, whatever `width` is.
@pytest.mark.parametrize(
    "api", ["shuffle_sync_up", "shuffle_sync_down", "shuffle_sync_bfly"]
)
@pytest.mark.parametrize("index,width", [(32, 32), (33, 32), (64, 32), (32, 16)])
def test_index_at_or_above_the_warp_size_is_rejected(
    api: str, index: int, width: int
) -> None:
    with pytest.raises(Exception, match="must be less than 32"):
        _rejected(api, index, width)


# An index that reaches outside the lane's group is a legal request: the
# hardware answers it by leaving that lane's own value in place, which is what
# makes a group-confined prefix sum or butterfly expressible. It must not be
# confused with the wrap above -- below 32 the control field loses nothing.
@pytest.mark.parametrize(
    "api", ["shuffle_sync_up", "shuffle_sync_down", "shuffle_sync_bfly"]
)
@pytest.mark.parametrize("index", [16, 20, 31])
def test_index_outside_the_group_is_accepted(api: str, index: int) -> None:
    assert "tla.arch.shuffle" in _dump(api, index, 16)


# "idx" is the exception: there the index is a lane number and the documented
# behaviour is src_lane % width, so wrapping is the contract, not a bug.
@pytest.mark.parametrize("index", [32, 33, 64])
def test_idx_wraps_a_lane_number_instead_of_rejecting_it(index: int) -> None:
    assert "tla.arch.shuffle <idx>" in _dump("shuffle_sync", index)
