"""Read lane IDs and warp size in one 48-thread SIMT block.

Threads 0-31 form a full warp; threads 32-47 form a partial warp.
Lane IDs restart at zero in the second warp, while warp size stays 32.

Usage: python warp_lane_queries.py --device 0
"""

from __future__ import annotations

import argparse

from catlass import tla

THREADS = 48


@tla.kernel
def warp_lane_queries(lanes: tla.Tensor, sizes: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            lanes[tid] = tla.arch.lane_idx()
            sizes[tid] = tla.arch.warpsize()
        tla.pipe_barrier(tla.pipes.ALL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch_npu

    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    lanes = torch.empty(THREADS, dtype=torch.int32, device="npu")
    sizes = torch.empty_like(lanes)
    lane_tensor = from_dlpack(lanes, layout_tag=tla.arch.RowMajor)
    size_tensor = from_dlpack(sizes, layout_tag=tla.arch.RowMajor)
    kernel = tla.compile(
        warp_lane_queries, lane_tensor, size_tensor, options="--npu-arch 3510"
    )
    kernel(lane_tensor, size_tensor, block_num=1)
    torch.npu.synchronize()

    print("Warp 0 lane IDs:", lanes[:32].cpu().tolist())
    print("Warp 1 lane IDs:", lanes[32:].cpu().tolist())
    print("Warp size:", sizes[0].item())


if __name__ == "__main__":
    main()
