"""Exact lane-mask checks; run while holding the project's NPU lock."""

from __future__ import annotations

import argparse
import json
import math
import operator

from catlass import tla


def _query(mode):
    return getattr(tla.arch, "lanemask_" + mode)()


@tla.kernel
def warp_lane_masks(
    output: tla.Tensor,
    masks: tla.Tensor,
    geometry: tla.Constexpr[tuple],
    mode: tla.Constexpr[str],
) -> None:
    block = tla.arch.block_idx()
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=geometry):
            tx, ty, tz = tla.arch.thread_idx()
            tid = tx + ty * geometry[0] + tz * geometry[0] * geometry[1]
            threads = geometry[0] * geometry[1] * geometry[2]
            offset = 16 + (block * threads + tid) * 5
            value = _query(mode)
            masks[16 + block * threads + tid] = value
            value = masks[16 + block * threads + tid] + tla.UInt32(0)
            output[offset] = value.to(tla.Int32).to(tla.Int64)
            output[offset + 1] = value.to(tla.UInt64).to(tla.Int64)
            output[offset + 2] = (value > tla.UInt32(0x7FFFFFFF)).to(tla.Int64)
            if tid % 2 == 0:
                output[offset + 3] = _query(mode).to(tla.UInt64).to(tla.Int64)
            else:
                output[offset + 3] = (_query(mode) ^ tla.UInt32(1)).to(tla.Int64)
            if tid % 32 < 16:
                output[offset + 4] = _query(mode).to(tla.UInt64).to(tla.Int64)
            else:
                output[offset + 4] = (_query(mode) ^ tla.UInt32(2)).to(tla.Int64)
        tla.pipe_barrier(tla.pipes.ALL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    for geometry, blocks in (
        ((32, 1, 1), 1),
        ((48, 1, 1), 1),
        ((8, 4, 1), 1),
        ((4, 4, 3), 3),
    ):
        threads = math.prod(geometry)
        count = threads * blocks
        lanes = torch.arange(threads, dtype=torch.int64).repeat(blocks) % 32
        for mode in ("eq", "lt", "le", "gt", "ge"):
            predicate = getattr(operator, mode)
            table = torch.tensor(
                [
                    sum(1 << bit for bit in range(32) if predicate(bit, lane))
                    for lane in range(32)
                ],
                dtype=torch.int64,
            )
            mask = table[lanes]
            expected = torch.stack(
                (
                    torch.where(mask >= 1 << 31, mask - (1 << 32), mask),
                    mask,
                    (mask > 0x7FFFFFFF).to(torch.int64),
                    mask ^ (lanes % 2),
                    mask ^ ((lanes >= 16).to(torch.int64) * 2),
                ),
                dim=1,
            )
            output = torch.full((count * 5 + 32,), -13, dtype=torch.int64, device="npu")
            tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
            storage = torch.full((count + 32,), -13, dtype=torch.int32, device="npu")
            masks = from_dlpack(
                storage.view(torch.uint32), layout_tag=tla.arch.RowMajor
            )
            artifact = tla.compile(
                warp_lane_masks,
                tensor,
                masks,
                geometry,
                mode,
                options="--npu-arch 3510",
            )
            for _ in range(2):
                output.fill_(-13)
                storage.fill_(-13)
                artifact(tensor, masks, block_num=blocks)
                torch.npu.synchronize()
                actual = output.cpu()
                stored = storage.cpu()
                if not torch.all(stored[:16] == -13) or not torch.all(
                    stored[-16:] == -13
                ):
                    raise AssertionError("Unsigned output guard was modified")
                torch.testing.assert_close(
                    stored[16:-16].to(torch.int64), expected[:, 0], rtol=0, atol=0
                )
                if not torch.all(actual[:16] == -13) or not torch.all(
                    actual[-16:] == -13
                ):
                    raise AssertionError("Output guard was modified")
                torch.testing.assert_close(
                    actual[16:-16].reshape(count, 5), expected, rtol=0, atol=0
                )
            print(
                json.dumps(
                    {
                        "geometry": geometry,
                        "blocks": blocks,
                        "mode": mode,
                        "threads_checked": count,
                        "repeats": 2,
                        "passed": True,
                        "binary": str(artifact.kernel_binary_path),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
