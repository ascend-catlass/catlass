# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end checks for packed pairwise sums and source masking."""

from __future__ import annotations

import argparse
from typing import Any

import catlass.tla as tla

_CASES = ("all", "single", "odd-tail")
_VECTOR_ELE = 64
_DTYPE: Any = tla.Float32
_CASE = _CASES[0]
_BATCH_RUN = False


@tla.kernel
def pair_wise_sum_op(mem_x: tla.Tensor, mem_y: tla.Tensor) -> None:
    """Store packed adjacent-lane sums and the zero-filled upper half."""
    block_idx = tla.arch.block_idx()
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    x_gm = tla.tile_view(mem_x, tla.make_shape(_VECTOR_ELE), tla.make_coord(block_idx))
    y_gm = tla.tile_view(mem_y, tla.make_shape(_VECTOR_ELE), tla.make_coord(block_idx))
    x_ptr = tla.allocate(_VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256)
    y_ptr = tla.allocate(_VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ptr, y_gm, tla.arch.RowMajor)

    def _store_pair_wise(x_tile: Any, y_tile: Any, case: str) -> None:
        """Emit one mask variant; shared by the single and multi-AIV paths."""
        source = x_tile.load()
        if tla.const_expr(case == "all"):
            selected = tla.create_mask(pattern=tla.mask.ALL, dtype=_DTYPE)
        elif tla.const_expr(case == "single"):
            selected = tla.create_mask(pattern=tla.mask.VL1, dtype=_DTYPE)
        else:
            selected, _ = tla.update_mask(_VECTOR_ELE - 1, dtype=_DTYPE)
        result = source.reduce(
            tla.ReductionOp.ADD,
            mode=tla.ReduceMode.PAIR_WISE,
            mask=selected,
        )
        # Do not reuse the source mask for the packed destination.
        y_tile.store(result)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        # Seed UB as well as GM so an unwritten lane cannot pass as zero.
        tla.copy(y_ub, y_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(x_ub, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
            y_tile = tla.tile_view(y_ub, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
            if tla.const_expr(_BATCH_RUN):
                if block_idx == 0:
                    _store_pair_wise(x_tile, y_tile, _CASES[0])
                elif block_idx == 1:
                    _store_pair_wise(x_tile, y_tile, _CASES[1])
                else:
                    _store_pair_wise(x_tile, y_tile, _CASES[2])
            else:
                _store_pair_wise(x_tile, y_tile, _CASE)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(y_gm, y_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _set_config(dtype: str, case: str, batch_run: bool) -> None:
    global _VECTOR_ELE, _DTYPE, _CASE, _BATCH_RUN
    _DTYPE = {"f16": tla.Float16, "f32": tla.Float32}[dtype]
    _VECTOR_ELE = 128 if dtype == "f16" else 64
    _CASE = case
    _BATCH_RUN = batch_run


def _input_data(dtype: str) -> Any:
    """Build deterministic, nonzero signed inputs on the CPU."""
    import torch

    count = 128 if dtype == "f16" else 64
    lanes = torch.arange(count, dtype=torch.float32)
    values = (lanes + 1.0) * 0.25
    values[1::4] *= -1
    # Include different magnitudes and f16 rounding, as well as cancellation.
    values[:8] = torch.tensor(
        [16.0, 0.0078125, -16.0, -0.0078125, 3.5, -3.5, -2.0, 0.25]
    )
    return values.to({"f16": torch.float16, "f32": torch.float32}[dtype])


def _pair_wise_sum_expected(x: Any, case: str) -> Any:
    """CPU reference: execute sum pairs, and zero the upper half."""
    import torch

    values = x.to(torch.float32).clone()
    if case == "single":
        values[1:] = 0
    elif case == "odd-tail":
        values[-1] = 0
    elif case != "all":
        raise ValueError(f"unknown pair-wise case: {case}")
    expected = torch.zeros_like(x)
    expected[: x.numel() // 2] = (values[0::2] + values[1::2]).to(x.dtype)
    return expected


def _run_cases(dtype: str, cases: tuple[str, ...], batch_run: bool) -> int:
    """Compile once per dtype and assign one independent GM tile to each AIV."""
    import torch

    _set_config(dtype, cases[0], batch_run)
    base = _input_data(dtype)
    # Distinct data per task exposes accidental reuse of block zero's GM tile.
    inputs = [base * (index + 1) for index in range(len(cases))]
    expected = [
        _pair_wise_sum_expected(values, case)
        for values, case in zip(inputs, cases)
    ]
    x = torch.cat(inputs).to("npu")
    y = torch.full_like(x, 123.0)
    tla_x = tla.from_dlpack(x, layout_tag=tla.arch.RowMajor)
    tla_y = tla.from_dlpack(y, layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(pair_wise_sum_op, tla_x, tla_y, options="--npu-arch 3510")
    artifact(tla_x, tla_y, block_num=len(cases))
    torch.npu.synchronize()

    actual = y.cpu().reshape(len(cases), _VECTOR_ELE)
    failed = 0
    for index, case in enumerate(cases):
        # One addition per pair: exact equality also checks all zeroed lanes.
        ok = torch.equal(actual[index], expected[index])
        failed += not ok
        print(f"op=pair_wise dtype={dtype} case={case} block={index} passed={ok}")
        if not ok:
            print(f"result   = {actual[index].tolist()}")
            print(f"expected = {expected[index].tolist()}")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print(f"kernel_ok={failed == 0} blocks={len(cases)}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run one dtype/mask case")
    mode.add_argument(
        "--batch-run", action="store_true",
        help="run all masks on separate AIV blocks; both dtypes unless --dtype is set",
    )
    parser.add_argument("--dtype", choices=("f16", "f32"))
    parser.add_argument("--case", choices=_CASES)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.batch_run and args.case is not None:
        parser.error("--case cannot be combined with --batch-run")

    import torch
    import torch_npu  # noqa: F401 -- registers the NPU backend

    torch.npu.set_device(args.device)
    if args.batch_run:
        dtypes = (args.dtype,) if args.dtype else ("f16", "f32")
        failed = 0
        for dtype in dtypes:
            failed += _run_cases(dtype, _CASES, batch_run=True)
        total = len(dtypes) * len(_CASES)
        print(
            f"batch summary: passed={total - failed} failed={failed} "
            f"batches={len(dtypes)} total={total}"
        )
        return 0 if failed == 0 else 1
    failed = _run_cases(args.dtype or "f32", (args.case or _CASES[0],), batch_run=False)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
