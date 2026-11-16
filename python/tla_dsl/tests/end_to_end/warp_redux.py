"""Warp reduction correctness; run while holding the project's NPU lock."""

import argparse
import json
import math

from catlass import tla


@tla.kernel
def warp_redux(
    src: tla.Tensor,
    dst: tla.Tensor,
    dtype: tla.Constexpr,
    kind: tla.Constexpr,
    geometry: tla.Constexpr,
):
    block = tla.arch.block_idx()
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=geometry):
            tx, ty, tz = tla.arch.thread_idx()
            tid = tx + ty * geometry[0] + tz * geometry[0] * geometry[1]
            offset = block * (geometry[0] * geometry[1] * geometry[2]) + tid
            lane = tla.arch.lane_idx()
            value = src[offset].to(dtype)
            dst[16 + offset * 5] = tla.arch.warp_redux_sync(value, kind).to(
                dst.element_type
            )
            if lane % 2 == 0:
                dst[17 + offset * 5] = tla.arch.warp_redux_sync(value, kind).to(
                    dst.element_type
                )
            if lane < 16:
                dst[18 + offset * 5] = tla.arch.warp_redux_sync(value, kind).to(
                    dst.element_type
                )
            if lane == 0:
                dst[19 + offset * 5] = tla.arch.warp_redux_sync(value, kind).to(
                    dst.element_type
                )
            dst[20 + offset * 5] = tla.arch.warp_redux_sync(value, kind).to(
                dst.element_type
            )
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def constant_redux(dst: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=32):
            tid, _, _ = tla.arch.thread_idx()
            one = tla.Int32(1)
            dst[16 + tid] = tla.arch.warp_redux_sync(one)
            if tid % 2 == 0:
                dst[48 + tid] = tla.arch.warp_redux_sync(one)
            dst[80 + tid] = tla.arch.warp_redux_sync(one)
        tla.pipe_barrier(tla.pipes.ALL)


def check_constant_redux(torch, from_dlpack):
    import os
    from unittest.mock import patch

    from catlass import execution
    from catlass._mlir import passmanager

    lower = execution.lower_tlair_module_to_mlir

    def lower_with_cse(module, **kwargs):
        with module.context:
            passmanager.PassManager.parse("builtin.module(cse)").run(module.operation)
        return lower(module, **kwargs)

    output = torch.full((128,), -13, dtype=torch.int32, device="npu")
    tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
    expected = torch.full((128,), -13, dtype=torch.int32)
    expected[16:48] = 32
    expected[48:80:2] = 16
    expected[80:112] = 32
    for use_cse in (False, True):
        with (
            patch.dict(os.environ, {"CATLASS_DSL_FORCE_RECOMPILE": "1"}),
            patch.object(
                execution,
                "lower_tlair_module_to_mlir",
                lower_with_cse if use_cse else lower,
            ),
        ):
            artifact = tla.compile(constant_redux, tensor, options="--npu-arch 3510")
        for _ in range(3):
            output.fill_(-13)
            artifact(tensor, block_num=1)
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        print(
            json.dumps(
                {"constant_redux": True, "pre_lowering_cse": use_cse, "passed": True}
            ),
            flush=True,
        )


def reference(values, threads, kind, dtype, torch):
    floating = dtype in (tla.Float16, tla.Float32)
    bits = dtype.width
    rows = torch.full((len(values), 5), -13, dtype=values.dtype)
    for start in range(0, len(values), threads):
        for warp in range(start, start + threads, 32):
            lanes = list(range(warp, min(warp + 32, start + threads)))
            groups = (lanes, lanes[::2], lanes[:16], lanes[:1], lanes)
            for column, group in enumerate(groups):
                data = values[group]
                if floating:
                    data = data.to(
                        torch.float16 if dtype is tla.Float16 else torch.float32
                    )
                    result = getattr(
                        data, {"add": "sum", "max": "max", "min": "min"}[kind]
                    )().item()
                else:
                    mask = (1 << bits) - 1
                    numbers = [int(x) & mask for x in data]
                    if dtype.signed:
                        numbers = [
                            x - (1 << bits) if x >= 1 << (bits - 1) else x
                            for x in numbers
                        ]
                    result = {"add": sum, "max": max, "min": min}[kind](numbers) & mask
                    if dtype.signed and result >= 1 << (bits - 1):
                        result -= 1 << bits
                rows[group, column] = result
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    check_constant_redux(torch, from_dlpack)
    types = (
        tla.Int8,
        tla.Int16,
        tla.Int32,
        tla.UInt8,
        tla.UInt16,
        tla.UInt32,
        tla.Float16,
        tla.Float32,
    )
    for dtype in types:
        floating = dtype in (tla.Float16, tla.Float32)
        for geometry in ((32, 1, 1), (48, 1, 1), (4, 4, 3)):
            threads = math.prod(geometry)
            count = threads * 3
            if floating:
                values = (torch.arange(count, dtype=torch.float32) % 17 - 8) / 8
            else:
                high = 1 << (dtype.width - 1)
                seeds = (-1, 0, 1, high - 1, high, (1 << dtype.width) - 1)
                values = torch.tensor(
                    [seeds[i % len(seeds)] for i in range(count)], dtype=torch.int64
                )
            source = values.npu()
            output = torch.full(
                (count * 5 + 32,), -13, dtype=values.dtype, device="npu"
            )
            tensors = [
                from_dlpack(source, layout_tag=tla.arch.RowMajor),
                from_dlpack(output, layout_tag=tla.arch.RowMajor),
            ]
            for kind in ("add", "max", "min"):
                artifact = tla.compile(
                    warp_redux,
                    *tensors,
                    dtype,
                    kind,
                    geometry,
                    options="--npu-arch 3510",
                )
                datasets = [values]
                if floating and geometry == (32, 1, 1):
                    # Defined exceptional cases; NaN min/max policy is checked
                    # separately against native Ascend C during device validation.
                    datasets.append(
                        torch.tensor([float("inf"), -float("inf")] * (count // 2))
                    )
                    if kind == "add":
                        datasets.append(torch.full_like(values, float("nan")))
                for data in datasets:
                    source.copy_(data)
                    expected = reference(data, threads, kind, dtype, torch)
                    for _ in range(2):
                        output.fill_(-13)
                        artifact(*tensors, block_num=3)
                        torch.npu.synchronize()
                        actual = output.cpu()
                        if not torch.all(actual[:16] == -13) or not torch.all(
                            actual[-16:] == -13
                        ):
                            raise AssertionError("Output guard was modified")
                        torch.testing.assert_close(
                            actual[16:-16].reshape(count, 5),
                            expected,
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )
                print(
                    json.dumps(
                        {
                            "type": dtype.__name__,
                            "kind": kind,
                            "geometry": geometry,
                            "passed": True,
                            "binary": str(artifact.kernel_binary_path),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
