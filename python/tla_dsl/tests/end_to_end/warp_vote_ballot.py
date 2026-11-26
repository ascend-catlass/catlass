"""Exact SIMT ballot checks for predicates, active sets, geometry, and cache reuse."""

import argparse
import json
import math

from catlass import tla


@tla.kernel
def ballot_geometry(out: tla.Tensor, masks: tla.Tensor, geometry: tla.Constexpr):
    block = tla.arch.block_idx()
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=geometry):
            x, y, z = tla.arch.thread_idx()
            tid = x + y * geometry[0] + z * geometry[0] * geometry[1]
            threads = geometry[0] * geometry[1] * geometry[2]
            index = block * threads + tid
            offset = 16 + index * 13
            lane = tla.arch.lane_idx()
            pred = (tid + block) % 3 == 0
            out[offset] = tla.arch.vote_ballot_sync(False).to(tla.UInt64).to(tla.Int64)
            out[offset + 1] = (
                tla.arch.vote_ballot_sync(True).to(tla.UInt64).to(tla.Int64)
            )
            out[offset + 2] = (
                tla.arch.vote_ballot_sync(pred).to(tla.UInt64).to(tla.Int64)
            )
            if lane % 2 == 0:
                out[offset + 3] = (
                    tla.arch.vote_ballot_sync(pred).to(tla.UInt64).to(tla.Int64)
                )
            if lane < 16:
                out[offset + 4] = (
                    tla.arch.vote_ballot_sync(True).to(tla.UInt64).to(tla.Int64)
                )
            else:
                out[offset + 4] = (
                    (tla.arch.vote_ballot_sync(True) ^ tla.UInt32(1))
                    .to(tla.UInt64)
                    .to(tla.Int64)
                )
            out[offset + 5] = (
                tla.arch.vote_ballot_sync(pred).to(tla.UInt64).to(tla.Int64)
            )
            if lane % 2 == 0:
                if lane < 16:
                    out[offset + 6] = (
                        tla.arch.vote_ballot_sync(pred).to(tla.UInt64).to(tla.Int64)
                    )
            if lane == 31:
                out[offset + 7] = (
                    tla.arch.vote_ballot_sync(True).to(tla.UInt64).to(tla.Int64)
                )
            if lane == 0 or lane == 31:
                out[offset + 8] = (
                    tla.arch.vote_ballot_sync(True).to(tla.UInt64).to(tla.Int64)
                )
            for step in tla.range(2):
                if lane >= step * 8:
                    out[offset + 9 + step] = (
                        tla.arch.vote_ballot_sync(pred).to(tla.UInt64).to(tla.Int64)
                    )
            out[offset + 11] = (
                tla.arch.vote_ballot_sync(True).to(tla.Int32).to(tla.Int64)
            )
            masks[16 + index] = tla.arch.vote_ballot_sync(pred)
            out[offset + 12] = masks[16 + index].to(tla.UInt64).to(tla.Int64)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def ballot_integer(src: tla.Tensor, out: tla.Tensor, dtype: tla.Constexpr):
    block = tla.arch.block_idx()
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=48):
            tid, _, _ = tla.arch.thread_idx()
            index = block * 48 + tid
            value = tla.arch.vote_ballot_sync(src[index].to(dtype))
            out[16 + index * 2] = value.to(tla.UInt64).to(tla.Int64)
            out[17 + index * 2] = value.to(tla.Int32).to(tla.Int64)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def ballot_literal(out: tla.Tensor, predicate: tla.Constexpr):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=48):
            tid, _, _ = tla.arch.thread_idx()
            out[16 + tid] = (
                tla.arch.vote_ballot_sync(predicate).to(tla.UInt64).to(tla.Int64)
            )
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def constant_ballot(out: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=32):
            tid, _, _ = tla.arch.thread_idx()
            out[16 + tid * 3] = (
                tla.arch.vote_ballot_sync(True).to(tla.UInt64).to(tla.Int64)
            )
            if tid % 2 == 0:
                out[17 + tid * 3] = (
                    tla.arch.vote_ballot_sync(True).to(tla.UInt64).to(tla.Int64)
                )
            out[18 + tid * 3] = (
                tla.arch.vote_ballot_sync(True).to(tla.UInt64).to(tla.Int64)
            )
        tla.pipe_barrier(tla.pipes.ALL)


def check_output(actual, expected, torch):
    if not ((actual[:16] == -13).all() and (actual[-16:] == -13).all()):
        raise AssertionError("Output guard was modified")
    torch.testing.assert_close(
        actual[16:-16].reshape(expected.shape), expected, rtol=0, atol=0
    )


def run_case(
    kernel,
    output,
    expected,
    tensor,
    args,
    blocks,
    info,
    torch,
    masks=None,
    runtime_args=(),
):
    artifact = tla.compile(kernel, tensor, *args, options="--npu-arch 3510")
    for _ in range(2):
        output.fill_(-13)
        if masks is not None:
            masks.fill_(-13)
        artifact(tensor, *runtime_args, block_num=blocks)
        torch.npu.synchronize()
        check_output(output.cpu(), expected, torch)
        if masks is not None:
            stored = masks.cpu().to(torch.int64)
            check_output(stored, expected[:, 2].to(torch.int32).to(torch.int64), torch)
    print(
        json.dumps(
            {**info, "passed": True, "binary": str(artifact.kernel_binary_path)}
        ),
        flush=True,
    )


def check_cse(torch, from_dlpack):
    import os
    from unittest.mock import patch
    from catlass import execution
    from catlass._mlir import passmanager

    lower = execution.lower_tlair_module_to_mlir

    def lower_with_cse(module, **kwargs):
        with module.context:
            passmanager.PassManager.parse("builtin.module(cse)").run(module.operation)
        return lower(module, **kwargs)

    expected = torch.full((32, 3), -13, dtype=torch.int64)
    expected[:, 0] = 0xFFFFFFFF
    expected[::2, 1] = 0x55555555
    expected[:, 2] = 0xFFFFFFFF
    output = torch.full((32 * 3 + 32,), -13, dtype=torch.int64, device="npu")
    tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
    for cse in (False, True):
        with (
            patch.dict(os.environ, {"CATLASS_DSL_FORCE_RECOMPILE": "1"}),
            patch.object(
                execution,
                "lower_tlair_module_to_mlir",
                lower_with_cse if cse else lower,
            ),
        ):
            artifact = tla.compile(constant_ballot, tensor, options="--npu-arch 3510")
        for _ in range(2):
            output.fill_(-13)
            artifact(tensor, block_num=1)
            torch.npu.synchronize()
            check_output(output.cpu(), expected, torch)
        print(json.dumps({"pre_lowering_cse": cse, "passed": True}), flush=True)


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
            for block in range(blocks):
                for tid in range(threads):
                    lane = tid % 32
                    start = tid // 32 * 32
                    lanes = range(min(32, threads - start))
                    full = sum(1 << j for j in lanes)
                    voting = [j for j in lanes if (start + j + block) % 3 == 0]
                    ballot = sum(1 << j for j in voting)
                    half = sum(1 << j for j in lanes if (j < 16) == (lane < 16))
                    rows.append(
                        [
                            0,
                            full,
                            ballot,
                            sum(1 << j for j in voting if j % 2 == 0)
                            if lane % 2 == 0
                            else -13,
                            half ^ (1 if lane >= 16 else 0),
                            ballot,
                            sum(1 << j for j in voting if j % 2 == 0 and j < 16)
                            if lane % 2 == 0 and lane < 16
                            else -13,
                            (1 << 31) if lane == 31 else -13,
                            sum(1 << j for j in lanes if j in (0, 31))
                            if lane in (0, 31)
                            else -13,
                            ballot,
                            sum(1 << j for j in voting if j >= 8) if lane >= 8 else -13,
                            full if full < 1 << 31 else full - (1 << 32),
                            ballot,
                        ]
                    )
            expected = torch.tensor(rows, dtype=torch.int64)
            output = torch.full(
                (threads * blocks * 13 + 32,), -13, dtype=torch.int64, device="npu"
            )
            tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
            storage = torch.full(
                (threads * blocks + 32,), -13, dtype=torch.int32, device="npu"
            )
            masks = from_dlpack(
                storage.view(torch.uint32), layout_tag=tla.arch.RowMajor
            )
            run_case(
                ballot_geometry,
                output,
                expected,
                tensor,
                (masks, geometry),
                blocks,
                {"case": "geometry", "geometry": geometry, "blocks": blocks},
                torch,
                masks=storage,
                runtime_args=(masks,),
            )
    pattern = [0, -1, 2, 0, -(1 << 63), 1 << 40, 1 << 32, 1 << 31, 255, 256, -128, 0]
    values = (pattern * 12)[:144]
    source = torch.tensor(values, dtype=torch.int64, device="npu")
    src = from_dlpack(source, layout_tag=tla.arch.RowMajor)
    for dtype in (
        tla.Int8,
        tla.Int16,
        tla.Int32,
        tla.Int64,
        tla.UInt8,
        tla.UInt16,
        tla.UInt32,
        tla.UInt64,
    ):
        rows = []
        for block in range(3):
            for tid in range(48):
                start = block * 48 + tid // 32 * 32
                mask = sum(
                    1 << j
                    for j in range(min(32, 48 - tid // 32 * 32))
                    if (values[start + j] & ((1 << dtype.width) - 1)) != 0
                )
                rows.append([mask, mask if mask < 1 << 31 else mask - (1 << 32)])
        expected = torch.tensor(rows, dtype=torch.int64)
        output = torch.full((144 * 2 + 32,), -13, dtype=torch.int64, device="npu")
        tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
        artifact = tla.compile(
            ballot_integer, src, tensor, dtype, options="--npu-arch 3510"
        )
        for _ in range(2):
            output.fill_(-13)
            artifact(src, tensor, block_num=3)
            torch.npu.synchronize()
            check_output(output.cpu(), expected, torch)
        print(
            json.dumps(
                {
                    "case": "integer",
                    "dtype": dtype.__name__,
                    "passed": True,
                    "binary": str(artifact.kernel_binary_path),
                }
            ),
            flush=True,
        )
    for predicate in (False, True, 0, -7, 1 << 100):
        expected = torch.tensor(
            [0xFFFFFFFF if predicate else 0] * 32 + [0xFFFF if predicate else 0] * 16,
            dtype=torch.int64,
        )
        output = torch.full((80,), -13, dtype=torch.int64, device="npu")
        tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
        run_case(
            ballot_literal,
            output,
            expected,
            tensor,
            (predicate,),
            1,
            {"case": "literal", "predicate": str(predicate)},
            torch,
        )
    check_cse(torch, from_dlpack)


if __name__ == "__main__":
    main()
