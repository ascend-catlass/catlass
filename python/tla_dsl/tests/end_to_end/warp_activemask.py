"""Exact active masks for full/partial warps, divergence, loops and blocks."""

import argparse
import json
import math

from catlass import tla


@tla.kernel
def warp_activemask(output: tla.Tensor, geometry: tla.Constexpr[tuple]):
    block = tla.arch.block_idx()
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=geometry):
            x, y, z = tla.arch.thread_idx()
            tid = x + y * geometry[0] + z * geometry[0] * geometry[1]
            offset = 16 + (block * geometry[0] * geometry[1] * geometry[2] + tid) * 12
            value = tla.arch.activemask()
            output[offset] = value.to(tla.UInt64).to(tla.Int64)
            output[offset + 1] = value.to(tla.Int32).to(tla.Int64)
            if tid % 2 == 0:
                output[offset + 2] = tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
            else:
                # Distinct effects keep the backend from reconverging identical arms.
                output[offset + 2] = (
                    (tla.arch.activemask() ^ tla.UInt32(1)).to(tla.UInt64).to(tla.Int64)
                )
            if tid % 32 < 16:
                output[offset + 3] = tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
            else:
                output[offset + 3] = (
                    (tla.arch.activemask() ^ tla.UInt32(2)).to(tla.UInt64).to(tla.Int64)
                )
            output[offset + 4] = tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
            for step in tla.range(2):
                if tid % 32 >= step * 8:
                    output[offset + 5 + step] = (
                        tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
                    )
            if tid % 2 == 0:
                if tid % 32 < 16:
                    output[offset + 7] = (
                        tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
                    )
                output[offset + 8] = tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
            if tid % 32 == 31:
                output[offset + 9] = tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
            if tid % 32 == 0 or tid % 32 == 31:
                output[offset + 10] = tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
            output[offset + 11] = tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
        tla.pipe_barrier(tla.pipes.ALL)


def _check_guards(actual):
    if not ((actual[:16] == -13).all() and (actual[-16:] == -13).all()):
        raise AssertionError("Output guard was modified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    geometries = (
        (1, 1, 1),
        (31, 1, 1),
        (32, 1, 1),
        (33, 1, 1),
        (48, 1, 1),
        (64, 1, 1),
        (8, 4, 1),
        (4, 4, 3),
        (1, 4, 9),
        (4, 1, 9),
        (5, 7, 2),
        (32, 2, 2),
    )
    for geometry in geometries:
        for blocks in (1, 3):
            threads = math.prod(geometry)
            rows = []
            for _ in range(blocks):
                for tid in range(threads):
                    lane = tid % 32
                    active = range(min(32, threads - tid // 32 * 32))
                    full = sum(1 << j for j in active)
                    parity = sum(1 << j for j in active if j % 2 == lane % 2)
                    half = sum(1 << j for j in active if (j < 16) == (lane < 16))
                    tail = sum(1 << j for j in active if j >= 8) if lane >= 8 else -13
                    rows.append(
                        [
                            full,
                            full if full < 1 << 31 else full - (1 << 32),
                            parity ^ (lane % 2),
                            half ^ (2 if lane >= 16 else 0),
                            full,
                            full,
                            tail,
                            sum(1 << j for j in active if j % 2 == 0 and j < 16)
                            if lane % 2 == 0 and lane < 16
                            else -13,
                            parity if lane % 2 == 0 else -13,
                            (1 << 31) if lane == 31 else -13,
                            sum(1 << j for j in active if j in (0, 31))
                            if lane in (0, 31)
                            else -13,
                            full,
                        ]
                    )
            expected = torch.tensor(rows, dtype=torch.int64)
            output = torch.full(
                (threads * blocks * 12 + 32,), -13, dtype=torch.int64, device="npu"
            )
            tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
            artifact = tla.compile(
                warp_activemask, tensor, geometry, options="--npu-arch 3510"
            )
            for _ in range(2):
                output.fill_(-13)
                artifact(tensor, block_num=blocks)
                torch.npu.synchronize()
                actual = output.cpu()
                _check_guards(actual)
                torch.testing.assert_close(
                    actual[16:-16].reshape(-1, 12), expected, rtol=0, atol=0
                )
            print(
                json.dumps(
                    {
                        "geometry": geometry,
                        "blocks": blocks,
                        "passed": True,
                        "binary": str(artifact.kernel_binary_path),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
