# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
from typing import Any

import catlass.tla as tla

_REDUCE_OP = tla.ReductionOp.ADD
# Blocks are 32 bytes: 8 lanes per block for 32-bit types, 16 for 16-bit
# types. The operand is one full 256-byte register vector: 64 lanes for
# 32-bit types, 128 for 16-bit types; both hold 8 blocks.
_BLOCK_LANES = {"f32": 8, "i16": 16}
_VECTOR_ELE = 64
_DTYPE: Any = tla.Float32
# ADD on 16-bit sources returns the widened 32-bit instruction dst: an i32
# vector with half the lanes.
_OUT_DTYPE: Any = tla.Float32
_OUT_ELE = _VECTOR_ELE


@tla.kernel
def reduce_intra_block_op(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """Reduce each 32-byte block, packed result in the least significant lanes.

    reduce_intra_block mirrors the Vcg group-reduction instruction: the result
    has the same shape as the operand, with each block's reduction result laid
    out contiguously in the least-significant lanes and the remaining lanes
    zero. The whole packed vector is stored so the host can check both parts.
    """
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    x_gm = tla.tile_view(mem_x, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(_OUT_ELE), tla.make_coord(0))
    x_ptr = tla.allocate(_VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256)
    z_ptr = tla.allocate(_OUT_ELE, _OUT_DTYPE, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(x_ub, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(_OUT_ELE), tla.make_coord(0))
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=_DTYPE)
            z_tile.store(x_tile.load().reduce(_REDUCE_OP, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask))
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _set_op(op: str) -> None:
    global _REDUCE_OP
    _REDUCE_OP = {
        "add": tla.ReductionOp.ADD,
        "max": tla.ReductionOp.MAX,
        "min": tla.ReductionOp.MIN,
    }[op]


def _set_dtype(dtype: str) -> None:
    global _VECTOR_ELE, _DTYPE, _OUT_DTYPE, _OUT_ELE
    _DTYPE = {"f32": tla.Float32, "i16": tla.Int16}[dtype]
    _VECTOR_ELE = 64 if _DTYPE is tla.Float32 else 128
    if dtype == "i16" and _REDUCE_OP is tla.ReductionOp.ADD:
        # Widened i32 dst: half the lanes of the i16 operand.
        _OUT_DTYPE = tla.Int32
        _OUT_ELE = _VECTOR_ELE // 2
    else:
        _OUT_DTYPE = _DTYPE
        _OUT_ELE = _VECTOR_ELE


def _runtime_tensor(dev_buf: Any) -> Any:
    return tla.from_dlpack(
        dev_buf.contiguous(),
        layout_tag=tla.arch.RowMajor,
    )


def _input_data(dtype: str, device: Any) -> Any:
    import torch

    if dtype == "f32":
        return torch.linspace(-40.0, 87.0, 64, dtype=torch.float32, device=device)
    # i16: blocks 0..6 stay in range; block 7 sums past the i16 max to show
    # that the widened i32 dst keeps the exact sum. (arange has no int16 NPU
    # kernel, so build on host and move.)
    in_range = torch.arange(1, 113, dtype=torch.int16)
    overflow = torch.full((16,), 4100, dtype=torch.int16)
    return torch.cat((in_range, overflow)).to(device)


def _packed_expected(op: str, x: Any, block_lanes: int, out_ele: int) -> Any:
    """Host reference: per-block result in the low lanes, zero elsewhere.

    For i16 ADD the instruction widens the block sum to 32 bits and the DSL
    returns that i32 dst as-is, so sums beyond the i16 range (the last block)
    stay exact instead of wrapping.
    """
    import torch

    blocks = x.reshape(-1, block_lanes)
    if op == "add":
        if x.dtype == torch.int16:
            block_result = blocks.sum(dim=1, dtype=torch.int32)
        else:
            block_result = blocks.sum(dim=1)
    elif op == "max":
        # The int32 detour is only needed for i16 (avoiding dtype promotion in
        # torch.max); on f32 it would truncate the values before comparing.
        if x.dtype == torch.int16:
            block_result = blocks.to(torch.int32).max(dim=1).values.to(x.dtype)
        else:
            block_result = blocks.max(dim=1).values
    else:
        if x.dtype == torch.int16:
            block_result = blocks.to(torch.int32).min(dim=1).values.to(x.dtype)
        else:
            block_result = blocks.min(dim=1).values
    out_dtype = torch.int32 if x.dtype == torch.int16 and op == "add" else x.dtype
    packed = torch.zeros(out_ele, dtype=out_dtype, device=x.device)
    packed[: block_result.numel()] = block_result
    return packed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=("add", "max", "min"), nargs="?")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dtype", choices=("f32", "i16"), default="f32")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.op is None:
        raise SystemExit("a positional op is required")
    _set_op(args.op)
    _set_dtype(args.dtype)
    if not args.run:
        raise SystemExit("pass --run")

    import torch
    import torch_npu

    torch.npu.set_device(args.device)

    block_lanes = _BLOCK_LANES[args.dtype]
    n_lanes = _VECTOR_ELE // block_lanes
    x = _input_data(args.dtype, "npu")
    z = torch.zeros(
        _OUT_ELE, dtype=x.dtype if _OUT_DTYPE is _DTYPE else torch.int32, device="npu"
    )
    tla_x = _runtime_tensor(x)
    tla_z = _runtime_tensor(z)
    artifact = tla.compile(
        reduce_intra_block_op,
        tla_x,
        tla_z,
        options="--npu-arch 3510",
    )
    artifact(tla_x, tla_z, block_num=1)
    torch.npu.synchronize()

    expected = _packed_expected(args.op, x, block_lanes, _OUT_ELE).cpu()
    z_host = z.cpu()
    if z_host.dtype.is_floating_point:
        # f32 block sums may add up in a different order on hardware than in
        # the host reference, so compare with a small absolute tolerance
        # instead of exact equality.
        ok = bool(torch.isclose(z_host, expected, rtol=0.0, atol=1e-4).all())
    else:
        ok = bool(torch.equal(z_host, expected))

    print(f"op={args.op} dtype={args.dtype} block_lanes={block_lanes} blocks={n_lanes}")
    print(f"result lanes  = {z[:n_lanes].cpu().tolist()}")
    print(f"expected      = {expected[:n_lanes].cpu().tolist()}")
    print(f"packed output equals expected block reduction? {ok}")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
