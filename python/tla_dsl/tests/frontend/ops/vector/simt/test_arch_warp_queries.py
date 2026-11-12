"""Frontend scope and scalar types for the A1 warp queries."""

from __future__ import annotations

import pytest

from catlass import tla
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def _warp_queries(out: tla.Tensor) -> None:
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simt", thread_block_dim=(4, 4, 3)):
            tid, _, _ = tla.arch.thread_idx()
            out[tid] = tla.arch.lane_idx()
            out[tid + 64] = tla.arch.warpsize()


@tla.kernel
def _lane_in_vector() -> None:
    with tla.vector():
        tla.arch.lane_idx()


@tla.kernel
def _size_in_vector() -> None:
    with tla.vector():
        tla.arch.warpsize()


@tla.kernel
def _lane_in_simd() -> None:
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simd"):
            tla.arch.lane_idx()


@tla.kernel
def _size_in_simd() -> None:
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simd"):
            tla.arch.warpsize()


@tla.kernel
def _lane_in_default_simd() -> None:
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func():
            tla.arch.lane_idx()


@tla.kernel
def _size_in_default_simd() -> None:
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func():
            tla.arch.warpsize()


@tla.kernel
def _lane_outside_vector() -> None:
    tla.arch.lane_idx()


@tla.kernel
def _size_outside_vector() -> None:
    tla.arch.warpsize()


def test_warp_queries_emit_i32_lane_and_constant_size() -> None:
    out = make_fake_tensor(tla.Int32, (128,), (1,))
    mlir = _warp_queries.dump_mlir(type_args=(out,))
    assert "tla.arch.lane_idx" in mlir
    assert "tla.arch.lane_idx -> i32" in mlir
    assert "arith.constant 32 : i32" in mlir
    assert "tla.arch.warpsize" not in mlir


@pytest.mark.parametrize(
    "kernel, name",
    [
        (_lane_in_vector, "lane_idx"),
        (_size_in_vector, "warpsize"),
        (_lane_in_simd, "lane_idx"),
        (_size_in_simd, "warpsize"),
        (_lane_in_default_simd, "lane_idx"),
        (_size_in_default_simd, "warpsize"),
        (_lane_outside_vector, "lane_idx"),
        (_size_outside_vector, "warpsize"),
    ],
)
def test_warp_queries_require_simt_region(kernel, name) -> None:
    with pytest.raises(Exception, match=rf"arch\.{name}.*mode='simt'"):
        kernel.dump_mlir()


@pytest.mark.parametrize("name", ["lane_idx", "warpsize"])
def test_warp_queries_reject_host_calls(name) -> None:
    with pytest.raises(Exception, match=rf"arch\.{name}"):
        getattr(tla.arch, name)()
