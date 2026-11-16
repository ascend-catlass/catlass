# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile and run tla.arch.shuffle_sync* cases, checking each lane's source.

Every case writes one output slot per thread, so the host can check the result
lane by lane against a reference derived from the warp/group geometry alone.
That is what pins the ctrl word's lane-mask field: at ``width=16`` a lane must
never see a value from the other half of its warp.

Intra-warp register exchange with ``tla.arch.shuffle_sync*``.

The four interfaces differ only in how a lane picks its source:

    shuffle_sync(v, src_lane)    lane `src_lane` of the group   -- broadcast
    shuffle_sync_up(v, delta)    lane `laneid - delta`          -- prefix scans
    shuffle_sync_down(v, delta)  lane `laneid + delta`          -- reductions
    shuffle_sync_bfly(v, mask)   lane `laneid ^ mask`           -- all-reduce

None of them touches UB and none needs ``tla.arch.sync_threads()``: the value
moves between registers inside one warp.

Every kernel here writes one output slot per thread, so the host can check the
result lane by lane against a reference computed from the warp geometry.
"""

from __future__ import annotations

import argparse
import math

import catlass.tla as tla

THREADS = 128  # four warps
WARP = 32
_DTYPE = tla.Float32

NPU_ARCH = "--npu-arch 3510"


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@tla.kernel
def warp_shuffle_modes(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    """All four modes over a whole warp, one output block per mode."""
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            v = gm_in[tid]
            gm_out[tid] = tla.arch.shuffle_sync(v, 0)
            gm_out[THREADS + tid] = tla.arch.shuffle_sync_up(v, 1)
            gm_out[2 * THREADS + tid] = tla.arch.shuffle_sync_down(v, 1)
            gm_out[3 * THREADS + tid] = tla.arch.shuffle_sync_bfly(v, 1)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def warp_shuffle_width16(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    """The same four modes confined to 16-lane groups.

    This is what pins the `width` encoding: with width=16 a lane must never see
    a value from the other half of its warp.
    """
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            v = gm_in[tid]
            gm_out[tid] = tla.arch.shuffle_sync(v, 0, 16)
            gm_out[THREADS + tid] = tla.arch.shuffle_sync_up(v, 1, 16)
            gm_out[2 * THREADS + tid] = tla.arch.shuffle_sync_down(v, 1, 16)
            gm_out[3 * THREADS + tid] = tla.arch.shuffle_sync_bfly(v, 1, 16)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def warp_shuffle_idx_above_width(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    """`shuffle_sync` with a lane number past the end of its 16-lane group.

    `shuffle_sync` documents the source as `src_lane % width`, so lane 20 of a
    16-lane group must read group lane 4 -- not the group's last lane, which is
    what a clamping control field would give. This is the only mode where an
    index at or above `width` is legal; the others reject it.
    """
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = tla.arch.shuffle_sync(gm_in[tid], 20, 16)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def warp_shuffle_outgroup(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    """`up`/`down` with a delta that reaches outside the lane's group.

    A delta below 32 is carried by the control field exactly, so a source lane
    outside the group is a request the hardware answers by leaving that lane its
    own value -- it is legal, not an error. The two blocks separate the two
    things that could go wrong: at `width=32` only lanes 20 and above can read,
    which fails if delta=20 were masked to something smaller, and at `width=16`
    no lane can read at all, which fails if the group boundary were ignored.
    """
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            v = gm_in[tid]
            gm_out[tid] = tla.arch.shuffle_sync_up(v, 20)
            gm_out[THREADS + tid] = tla.arch.shuffle_sync_down(v, 20)
            gm_out[2 * THREADS + tid] = tla.arch.shuffle_sync_up(v, 20, 16)
            gm_out[3 * THREADS + tid] = tla.arch.shuffle_sync_down(v, 20, 16)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def warp_shuffle_bfly_across(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    """`bfly` at `width=16` with a mask that crosses the group boundary.

    Unlike `up` and `down`, a butterfly can leave its group in either
    direction, and only the upper edge is checked: a source past the group's
    last lane keeps the lane its own value, while a source in a lower-numbered
    group is read normally. So `lane_mask=16` at `width=16` splits the warp in
    half -- lanes 0-15 keep their own values, lanes 16-31 read the lower half.

    The second block is the in-group control: `lane_mask=8` stays inside the
    group, so a regression that broke the butterfly outright would fail it
    rather than being absorbed by the asymmetry the first block expects.
    """
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            v = gm_in[tid]
            gm_out[tid] = tla.arch.shuffle_sync_bfly(v, 16, 16)
            gm_out[THREADS + tid] = tla.arch.shuffle_sync_bfly(v, 8, 16)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def warp_allreduce(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    """Butterfly all-reduce: every lane ends up with its warp's sum."""
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            total = gm_in[tid]
            for mask in (16, 8, 4, 2, 1):
                total = total + tla.arch.shuffle_sync_bfly(total, mask)
            gm_out[tid] = total
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def warp_shuffle_int32(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    """The i32 intrinsic family, to prove the dispatch is not f32-only."""
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            v = gm_in[tid]
            gm_out[tid] = tla.arch.shuffle_sync_bfly(v, 1)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def warp_shuffle_uint8(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    """An unsigned operand, which rides the i8 widening path.

    Signedness is not a property of a bit pattern being moved, so an unsigned
    value shuffles as a signless integer of its own width and keeps its Python
    type. u8 also has no intrinsic of its own, so this is the extend/truncate
    path as well.
    """
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = tla.arch.shuffle_sync_bfly(gm_in[tid], 1)
        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _reference_modes(values, width):
    """What each lane must read, derived from the warp/group geometry alone."""
    out = []
    for mode in ("idx", "up", "down", "bfly"):
        for tid in range(THREADS):
            lane = tid % WARP
            base = tid - lane + (lane // width) * width
            glane = lane % width
            if mode == "idx":
                src = base
            elif mode == "up":
                src = base + glane - 1 if glane >= 1 else tid
            elif mode == "down":
                src = base + glane + 1 if glane + 1 < width else tid
            else:
                src = base + (glane ^ 1)
            out.append(values[src])
    return out


def _reference_idx_above_width(values, src_lane=20, width=16):
    """Where lane `src_lane % width` of each group lives, in whole-tensor terms."""
    out = []
    for tid in range(THREADS):
        lane = tid % WARP
        base = tid - lane + (lane // width) * width
        out.append(values[base + src_lane % width])
    return out


def _reference_outgroup(values, delta=20):
    """Lane by lane, for `up` then `down` at width 32 and then at width 16.

    `up` keeps a lane's own value once `laneid - delta` falls below the start of
    its group, and `down` once `laneid + delta` passes the end of it.
    """
    out = []
    for mode, width in (("up", 32), ("down", 32), ("up", 16), ("down", 16)):
        for tid in range(THREADS):
            lane = tid % WARP
            base = tid - lane + (lane // width) * width
            glane = lane % width
            if mode == "up":
                src = base + glane - delta if glane >= delta else tid
            else:
                src = base + glane + delta if glane + delta < width else tid
            out.append(values[src])
    return out


def _reference_bfly_across(values, width=16):
    """Lane by lane, for `lane_mask=16` then `lane_mask=8`, both at `width=16`.

    Only the group's last lane bounds a butterfly, so a source above it leaves
    the lane its own value while a source below the group's first lane is read
    as it stands.
    """
    out = []
    for mask in (16, 8):
        for tid in range(THREADS):
            lane = tid % WARP
            base = tid - lane
            min_lane = (lane // width) * width
            max_lane = min_lane + 31 % width
            src = lane ^ mask
            out.append(values[tid] if src > max_lane else values[base + src])
    return out


def _reference_allreduce(values):
    out = []
    for tid in range(THREADS):
        base = tid - (tid % WARP)
        out.append(sum(values[base : base + WARP]))
    return out


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


CASES = (
    "modes",
    "width16",
    "idxwide",
    "outgroup",
    "bflyacross",
    "allreduce",
    "int32",
    "uint8",
)


def mismatches(got, reference, atol):
    """Return the (slot, got, want) triples where `got` is wrong.

    A NaN makes every comparison false, so `abs(got - want) > atol` on its own
    would let a kernel that writes nothing but NaNs report PASS. No reference
    value here is non-finite, so a non-finite output is a mismatch in its own
    right and is reported as one.
    """
    return [
        (i, got[i], reference[i])
        for i in range(len(reference))
        if not math.isfinite(got[i]) or abs(got[i] - reference[i]) > atol
    ]


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    def tla_tensor(buf):
        return from_dlpack(buf.contiguous(), layout_tag=tla.arch.RowMajor)

    torch.npu.set_device(args.device)
    atol = float(args.atol)
    failures = []

    # Distinct, exactly representable per-lane values, so a wrong source lane
    # cannot coincidentally produce the right answer.
    host_values = [float(i + 1) for i in range(THREADS)]

    def check(name, kernel, values, reference, dtype=torch.float32) -> None:
        """Run one case and record its verdict in ``failures``.

        Returns nothing on purpose: the verdict is already recorded, so no
        caller has to read one back.
        """
        src = torch.tensor(values, dtype=dtype, device="npu")
        n_out = len(reference)
        dst = torch.full((n_out,), -999, dtype=dtype, device="npu")
        a, b = tla_tensor(src), tla_tensor(dst)
        artifact = tla.compile(kernel, a, b, options=NPU_ARCH)
        artifact(a, b, block_num=1)
        torch.npu.synchronize()
        got = dst.tolist()
        bad = mismatches(got, reference, atol)
        ok = not bad
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({n_out} lanes checked)")
        if not ok:
            for i, g, w in bad[:8]:
                print(f"    lane {i % THREADS} (slot {i}): got {g}, want {w}")
            print(f"    ... {len(bad)} mismatches total")
            failures.append(name)

    selected = CASES if args.all_cases else (args.case,)
    print(f"--- warp shuffle, {THREADS} threads = {THREADS // WARP} warps ---")

    if "modes" in selected:
        check(
            "shuffle_sync / _up / _down / _bfly, width=32",
            warp_shuffle_modes,
            host_values,
            _reference_modes(host_values, 32),
        )
    if "width16" in selected:
        check(
            "the same four, width=16",
            warp_shuffle_width16,
            host_values,
            _reference_modes(host_values, 16),
        )
    if "idxwide" in selected:
        check(
            "shuffle_sync with src_lane past the group (20 of width 16)",
            warp_shuffle_idx_above_width,
            host_values,
            _reference_idx_above_width(host_values),
        )
    if "outgroup" in selected:
        check(
            "up/down with delta=20 reaching outside the group",
            warp_shuffle_outgroup,
            host_values,
            _reference_outgroup(host_values),
        )
    if "bflyacross" in selected:
        check(
            "bfly with lane_mask=16 crossing the width=16 group",
            warp_shuffle_bfly_across,
            host_values,
            _reference_bfly_across(host_values),
        )
    if "allreduce" in selected:
        check(
            "butterfly all-reduce over the warp",
            warp_allreduce,
            host_values,
            _reference_allreduce(host_values),
        )
    if "int32" in selected:
        int_values = [i + 1 for i in range(THREADS)]
        check(
            "shuffle_sync_bfly on Int32",
            warp_shuffle_int32,
            int_values,
            _reference_modes(int_values, 32)[3 * THREADS : 4 * THREADS],
            dtype=torch.int32,
        )

    if "uint8" in selected:
        u8_values = [i + 1 for i in range(THREADS)]
        check(
            "shuffle_sync_bfly on UInt8",
            warp_shuffle_uint8,
            u8_values,
            _reference_modes(u8_values, 32)[3 * THREADS : 4 * THREADS],
            dtype=torch.uint8,
        )

    passed = not failures
    print(f"SUMMARY passed={passed}" + ("" if passed else f" failed={failures}"))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Warp shuffle checks.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--case", choices=CASES, default="modes")
    parser.add_argument("--all-cases", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
