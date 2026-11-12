"""Exact A1 warp-query checks; run only after acquiring an admitted NPU.

Usage: python warp_lane_queries.py --device 0
The runner must hold the project's device lock for the whole process.
"""

from __future__ import annotations

import argparse
import json
import math

from catlass import tla


@tla.kernel
def warp_lane_queries(
    lanes: tla.Tensor,
    sizes: tla.Tensor,
    branches: tla.Tensor,
    geometry: tla.Constexpr[tuple],
) -> None:
    block = tla.arch.block_idx()
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=geometry):
            tx, ty, tz = tla.arch.thread_idx()
            tid = tx + ty * geometry[0] + tz * geometry[0] * geometry[1]
            offset = block * (geometry[0] * geometry[1] * geometry[2]) + tid
            lanes[offset] = tla.arch.lane_idx()
            sizes[offset] = tla.arch.warpsize()
            if tid % 2 == 0:
                branches[offset] = tla.arch.lane_idx()
            else:
                branches[offset] = tla.arch.lane_idx() + tla.arch.warpsize()
        tla.pipe_barrier(tla.pipes.ALL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    geometries = (
        (32, 1, 1),
        (48, 1, 1),
        (64, 1, 1),
        (8, 4, 1),
        (4, 4, 3),
        (32, 2, 2),
        (1, 1, 1),
        (31, 1, 1),
        (33, 1, 1),
        (1, 4, 9),
        (4, 1, 9),
        (5, 7, 2),
        (32, 8, 8),
    )
    for geometry in geometries:
        threads = math.prod(geometry)
        for blocks in (1, 3):
            count = threads * blocks
            buffers = [
                torch.full((count,), -1, dtype=torch.int32, device="npu")
                for _ in range(3)
            ]
            tensors = [from_dlpack(b, layout_tag=tla.arch.RowMajor) for b in buffers]
            artifact = tla.compile(
                warp_lane_queries, *tensors, geometry, options="--npu-arch 3510"
            )
            artifact(*tensors, block_num=blocks)
            torch.npu.synchronize()
            tid = torch.arange(threads, dtype=torch.int32).repeat(blocks)
            expected = (
                tid % 32,
                torch.full((count,), 32, dtype=torch.int32),
                tid % 32 + (tid % 2) * 32,
            )
            for name, actual, reference in zip(
                ("lane_idx", "warpsize", "branch_lane_idx"), buffers, expected
            ):
                torch.testing.assert_close(
                    actual.cpu(),
                    reference,
                    rtol=0,
                    atol=0,
                    msg=f"{name}, geometry={geometry}, blocks={blocks}",
                )
            print(
                json.dumps(
                    {
                        "geometry": geometry,
                        "blocks": blocks,
                        "threads_checked": count,
                        "passed": True,
                        "cache_key": artifact.cache_key,
                        "binary": str(artifact.kernel_binary_path),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
