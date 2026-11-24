# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E: the AR special register on the AIV.

AR counts the bytes the compressing vector stores have written. Two operations
reach it:

    tla.arch.clear_ar()   set AR to zero, so a sequence counts from the start
    count = tla.arch.get_ar()   read the accumulated byte count

Both run on the AIV. The clear is written inside the SIMD ``tla.vec.func`` that
holds the sequence -- the placement the native API documents -- or directly in
``tla.vector()``; the read is written in ``tla.vector()``, outside the vector
function, after the sequence has completed.

Each case writes the count it read to its own GM word and the host reads that
word back. What the cases establish:

    clear_in_vec_func   the in-VF clear and the scalar read run and agree on 0
    clear_in_scalar     the scalar placement of the clear does too
    clear_then_squeeze  a masked squeeze records the number of kept bytes
                        and preserves the compacted tile values
    two_clears          two clears and two reads, to show neither pair folds

Each kernel runs on one AIV and reports the value observed after its own
register operations.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import catlass.tla as tla

if TYPE_CHECKING:
    import torch

# One VL of f32 on Ascend 950.
LANES = 64
# M4 keeps every fourth lane, so a squeeze of one full register yields 16.
KEPT_LANES = LANES // 4


# --- kernels ---------------------------------------------------------------


@tla.kernel
def clear_in_vec_func(count_out: tla.Tensor) -> None:
    """The documented placement: the clear inside the SIMD vector function."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.arch.clear_ar()
        count_out[0] = tla.arch.get_ar()


@tla.kernel
def clear_in_scalar(count_out: tla.Tensor) -> None:
    """The scalar placement: the clear directly in the tla.vector() region.

    The helper behind it brings the vector function the instruction needs, so
    the source says nothing about one.
    """
    with tla.vector():
        tla.arch.clear_ar()
        count_out[0] = tla.arch.get_ar()


@tla.kernel
def clear_then_squeeze(src: tla.Tensor, dst: tla.Tensor, count_out: tla.Tensor) -> None:
    """Record the byte count of a masked squeeze and read it after completion."""
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("stored", tla.arch.VECTOR, tla.arch.MTE3)
    count_ready = tla.flag("count_ready", tla.arch.VECTOR, tla.arch.SCALAR)

    src_gm = tla.tile_view(src, tla.make_shape(LANES), tla.make_coord(0))
    dst_gm = tla.tile_view(dst, tla.make_shape(LANES), tla.make_coord(0))
    src_ub = tla.make_tensor_like(
        tla.allocate(LANES, tla.Float32, tla.AddressSpace.ub, 256),
        src_gm,
        tla.arch.RowMajor,
    )
    dst_ub = tla.make_tensor_like(
        tla.allocate(LANES, tla.Float32, tla.AddressSpace.ub, 256),
        dst_gm,
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(src_ub, src_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)

        with tla.vec.func(mode="simd"):
            tla.arch.clear_ar()
            src_tile = tla.tile_view(src_ub, tla.make_shape(LANES), tla.make_coord(0))
            dst_tile = tla.tile_view(dst_ub, tla.make_shape(LANES), tla.make_coord(0))
            every_fourth = tla.create_mask(pattern=tla.mask.M4, dtype=tla.Float32)
            packed = tla.squeeze(src_tile.load(), every_fourth, record_nbytes=True)
            dst_tile.store(
                packed,
                tla.params.UnalignStoreParams(
                    post_mode=tla.params.PostMode.POST_MODE_UPDATE_NBYTES
                ),
            )

        tla.set_flag(stored)
        tla.wait_flag(stored)

        tla.set_flag(count_ready)
        tla.wait_flag(count_ready)
        count_out[0] = tla.arch.get_ar()
        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def two_clears(count_out: tla.Tensor) -> None:
    """Two clears and two reads: neither pair may be folded into one.

    A folded read would report the count from before the second clear, and a
    dropped clear would leave the register as the previous sequence left it.
    Both words must read 0, and the IR behind them must still hold two of each,
    which the lit test checks.
    """
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.arch.clear_ar()
        count_out[0] = tla.arch.get_ar()
        tla.arch.clear_ar()
        count_out[1] = tla.arch.get_ar()


# --- harness ---------------------------------------------------------------


def _require_torch_npu(device: int):
    import torch

    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("torch_npu is required for this example") from exc
    torch_npu.npu.set_device(device)
    return torch


def _gm(tensor_1d) -> tla.Tensor:
    return tla.from_dlpack(tensor_1d.contiguous(), layout_tag=tla.arch.RowMajor)


# A sentinel the kernel must overwrite: a count word left at this value means
# the read never ran, which a check against 0 alone could not tell apart from a
# read that returned 0.
UNWRITTEN = -1


def _run_count_only(torch, device: str, name: str, kernel, words: int) -> int:
    counts = torch.full((max(words, 2),), UNWRITTEN, dtype=torch.int64, device=device)
    counts_t = _gm(counts)

    tla.compile(kernel, counts_t)(counts_t, block_num=1)
    torch.npu.synchronize()

    read = [int(counts[i].item()) for i in range(words)]
    if any(value != 0 for value in read):
        print(f"{name}_failed expected=0 per word actual={read}")
        return 1
    print(f"{name}_ok=True counts={read}")
    return 0


def _run_clear_then_squeeze(torch, device: str) -> int:
    src = torch.arange(1, LANES + 1, dtype=torch.float32, device=device)
    dst = torch.zeros(LANES, dtype=torch.float32, device=device)
    counts = torch.full((2,), UNWRITTEN, dtype=torch.int64, device=device)
    src_t, dst_t, counts_t = _gm(src), _gm(dst), _gm(counts)

    tla.compile(clear_then_squeeze, src_t, dst_t, counts_t)(
        src_t, dst_t, counts_t, block_num=1
    )
    torch.npu.synchronize()

    count = int(counts[0].item())
    expected = src[0::4]
    actual = dst[:KEPT_LANES]
    expected_bytes = KEPT_LANES * src.element_size()
    if count != expected_bytes:
        print(
            f"clear_then_squeeze_failed count expected={expected_bytes} actual={count}"
        )
        return 1
    if not torch.equal(actual, expected):
        print(
            f"clear_then_squeeze_failed tile expected={expected.tolist()} "
            f"actual={actual.tolist()}"
        )
        return 1
    print(f"clear_then_squeeze_ok=True count={count} kept={KEPT_LANES}")
    return 0


def run(args: argparse.Namespace) -> int:
    torch = _require_torch_npu(args.device)
    device = f"npu:{args.device}"

    failures = 0
    failures += _run_count_only(
        torch, device, "clear_in_vec_func", clear_in_vec_func, 1
    )
    failures += _run_count_only(torch, device, "clear_in_scalar", clear_in_scalar, 1)
    failures += _run_clear_then_squeeze(torch, device)
    failures += _run_count_only(torch, device, "two_clears", two_clears, 2)

    if failures:
        print(f"verification_failed cases={failures}")
        return 1
    print("verification_ok=True")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AR special register: tla.arch.clear_ar / tla.arch.get_ar."
    )
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
