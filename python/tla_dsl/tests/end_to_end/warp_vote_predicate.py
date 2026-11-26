"""Exact Warp all/any checks over active sets and integer predicate widths."""

import argparse
import json
import math
from contextlib import nullcontext
from unittest.mock import patch

from catlass import tla
from warp_vote_ballot import check_output


@tla.kernel
def vote_geometry(
    src: tla.Tensor, out: tla.Tensor, geometry: tla.Constexpr, dtype: tla.Constexpr
):
    block = tla.arch.block_idx()
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=geometry):
            x, y, z = tla.arch.thread_idx()
            tid = x + y * geometry[0] + z * geometry[0] * geometry[1]
            threads = geometry[0] * geometry[1] * geometry[2]
            index = block * threads + tid
            offset = 16 + index * 20
            lane = tla.arch.lane_idx()
            pred = src[index].to(dtype)
            out[offset] = tla.arch.vote_all_sync(False).to(tla.Int64)
            out[offset + 1] = tla.arch.vote_all_sync(True).to(tla.Int64)
            out[offset + 2] = tla.arch.vote_any_sync(False).to(tla.Int64)
            out[offset + 3] = tla.arch.vote_any_sync(True).to(tla.Int64)
            out[offset + 4] = tla.arch.vote_all_sync(pred).to(tla.Int64)
            out[offset + 5] = tla.arch.vote_any_sync(pred).to(tla.Int64)
            even = lane % 2 == 0
            odd = lane % 2 != 0
            out[offset + 18] = tla.arch.vote_all_sync(even).to(tla.Int64)
            out[offset + 19] = tla.arch.vote_any_sync(odd).to(tla.Int64)
            if lane % 2 == 0:
                out[offset + 6] = tla.arch.vote_all_sync(even).to(tla.Int64)
                out[offset + 7] = tla.arch.vote_any_sync(odd).to(tla.Int64)
                if lane < 16:
                    out[offset + 8] = tla.arch.vote_all_sync(pred).to(tla.Int64)
                    out[offset + 9] = tla.arch.vote_any_sync(pred).to(tla.Int64)
            out[offset + 10] = tla.arch.vote_all_sync(pred).to(tla.Int64)
            out[offset + 11] = tla.arch.vote_any_sync(pred).to(tla.Int64)
            for step in tla.range(2):
                if lane >= step * 8:
                    out[offset + 12 + step * 2] = tla.arch.vote_all_sync(pred).to(
                        tla.Int64
                    )
                    out[offset + 13 + step * 2] = tla.arch.vote_any_sync(pred).to(
                        tla.Int64
                    )
            if lane == 31:
                out[offset + 16] = tla.arch.vote_all_sync(pred).to(tla.Int64)
                out[offset + 17] = tla.arch.vote_any_sync(pred).to(tla.Int64)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def vote_literal(out: tla.Tensor, predicate: tla.Constexpr):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=48):
            tid, _, _ = tla.arch.thread_idx()
            out[16 + tid * 2] = tla.arch.vote_all_sync(predicate).to(tla.Int64)
            out[17 + tid * 2] = tla.arch.vote_any_sync(predicate).to(tla.Int64)
        tla.pipe_barrier(tla.pipes.ALL)


def reference(values, threads, blocks, torch):
    expected = torch.full((threads * blocks, 20), -13, dtype=torch.int64)
    for block in range(blocks):
        for tid in range(threads):
            index = block * threads + tid
            lane = tid % 32
            start = tid // 32 * 32
            active = range(start, min(start + 32, threads))

            def pair(lanes):
                preds = [values[block * threads + j] for j in lanes]
                return [int(all(preds)), int(any(preds))]

            expected[index, :4] = torch.tensor([0, 1, 0, 1])
            expected[index, 4:6] = torch.tensor(pair(active))
            expected[index, 18:20] = torch.tensor(
                [
                    int(all(j % 2 == 0 for j in active)),
                    int(any(j % 2 != 0 for j in active)),
                ]
            )
            if lane % 2 == 0:
                expected[index, 6:8] = torch.tensor([1, 0])
                if lane < 16:
                    expected[index, 8:10] = torch.tensor(
                        pair(j for j in active if j % 2 == 0 and j % 32 < 16)
                    )
            expected[index, 10:12] = torch.tensor(pair(active))
            for step in range(2):
                if lane >= step * 8:
                    expected[index, 12 + step * 2 : 14 + step * 2] = torch.tensor(
                        pair(j for j in active if j % 32 >= step * 8)
                    )
            if lane == 31:
                expected[index, 16:18] = torch.tensor(pair([tid]))
    return expected


def launch(
    kernel,
    tensors,
    compile_args,
    output,
    expected,
    blocks,
    info,
    torch,
    pre_lowering_cse=False,
):
    from catlass import execution
    from catlass._mlir import passmanager

    lower = execution.lower_tlair_module_to_mlir

    def lower_with_cse(module, **kwargs):
        with module.context:
            passmanager.PassManager.parse("builtin.module(cse)").run(module.operation)
        return lower(module, **kwargs)

    context = (
        patch.object(execution, "lower_tlair_module_to_mlir", lower_with_cse)
        if pre_lowering_cse
        else nullcontext()
    )
    with context:
        artifact = tla.compile(
            kernel, *tensors, *compile_args, options="--npu-arch 3510"
        )
    for _ in range(2):
        output.fill_(-13)
        artifact(*tensors, block_num=blocks)
        torch.npu.synchronize()
        check_output(output.cpu(), expected, torch)
    print(
        json.dumps(
            {**info, "passed": True, "binary": str(artifact.kernel_binary_path)}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pre-lowering-cse", action="store_true")
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    geometries = [
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
    ]
    cases = []
    for geometry in geometries:
        for blocks in (1, 3):
            cases.append((geometry, blocks, tla.Int32))
    cases += [
        ((48, 1, 1), 3, dtype)
        for dtype in (
            tla.Int8,
            tla.Int16,
            tla.Int64,
            tla.UInt8,
            tla.UInt16,
            tla.UInt32,
            tla.UInt64,
        )
    ]
    if args.pre_lowering_cse:
        cases = [((48, 1, 1), 3, tla.Int64)]
    pattern = [0, -1, 2, 0, -(1 << 63), 1 << 40, 1 << 32, 1 << 31, 255, 256, -128, 0]
    for geometry, blocks, dtype in cases:
        threads = math.prod(geometry)
        values = (pattern * (threads * blocks // len(pattern) + 1))[: threads * blocks]
        normalized = [bool(value & ((1 << dtype.width) - 1)) for value in values]
        expected = reference(normalized, threads, blocks, torch)
        src = torch.tensor(values, dtype=torch.int64, device="npu")
        output = torch.full(
            (threads * blocks * 20 + 32,), -13, dtype=torch.int64, device="npu"
        )
        tensors = tuple(
            from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in (src, output)
        )
        launch(
            vote_geometry,
            tensors,
            (geometry, dtype),
            output,
            expected,
            blocks,
            {
                "geometry": geometry,
                "blocks": blocks,
                "dtype": dtype.__name__,
                "pre_lowering_cse": args.pre_lowering_cse,
            },
            torch,
            args.pre_lowering_cse,
        )
    for predicate in () if args.pre_lowering_cse else (False, True, 0, -7, 1 << 100):
        output = torch.full((48 * 2 + 32,), -13, dtype=torch.int64, device="npu")
        tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
        expected = torch.full((48, 2), int(bool(predicate)), dtype=torch.int64)
        launch(
            vote_literal,
            (tensor,),
            (predicate,),
            output,
            expected,
            1,
            {"literal": str(predicate)},
            torch,
        )


if __name__ == "__main__":
    main()
