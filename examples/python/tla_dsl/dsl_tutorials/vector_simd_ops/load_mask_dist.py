# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end MaskLoadParams(DIST_US / DIST_DS) coverage.

Modes (positional ``op``):

* ``ds`` — bit-downsample ``plds`` dist=2 → ``tla.where``
* ``us`` — bit-upsample / PACK ``plds`` dist=1 → ``tla.where``
  (f32 companion: byte ``i`` gates lane ``4*i``)

``--mask-dtype`` selects AscendC ``LoadAlign<T, MaskDist>`` address typing
(``i8`` / ``i16`` / ``i32``; IR also allows b64 but GM↔UB copy does not).
The first 64 mask **bytes** are kept identical across dtypes so ``plds``
results match (``T`` does not change the bit window). Companion vector is
always f32 VL=64. Inactive lanes are zeroed.

One kernel body; module state is refreshed before each compile.
"""

from __future__ import annotations

import argparse
from typing import Any

import catlass.tla as tla
from catlass.params import MaskLoadDist, MaskLoadParams

VECTOR_ELE = 64
# Bytes consumed by DIST_DS (VL/4) / enough for DIST_US (VL/16) on 3510.
_MASK_PATTERN_BYTES = 64

_MASK_LOAD_PARAMS = MaskLoadParams(load_dist=MaskLoadDist.DIST_DS)
_MASK_DTYPE: type = tla.Int8

# make_tensor_like / GM↔UB copy support i8/i16/i32 (not i64) today.
# MaskLoadDist IR still accepts b64; e2e covers the copy-capable set.
_MASK_DTYPE_MAP: dict[str, type] = {
    "i8": tla.Int8,
    "i16": tla.Int16,
    "i32": tla.Int32,
}

_TORCH_DTYPE_MAP: dict[str, str] = {
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
}


@tla.kernel
def load_mask_dist_kernel(
    mem_src: tla.Tensor, mem_mask: tla.Tensor, mem_dst: tla.Tensor
) -> None:
    src_loaded = tla.flag("src_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    mask_loaded = tla.flag("mask_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    mask_gm = tla.tile_view(mem_mask, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    src_ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    mask_ptr = tla.allocate(VECTOR_ELE, _MASK_DTYPE, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)

    src_ub = tla.make_tensor_like(src_ptr, src_gm, tla.arch.RowMajor)
    mask_ub = tla.make_tensor_like(mask_ptr, mask_gm, tla.arch.RowMajor)
    dst_ub = tla.make_tensor_like(dst_ptr, dst_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(src_ub, src_gm)
        tla.copy(mask_ub, mask_gm)

        tla.set_flag(src_loaded)
        tla.wait_flag(src_loaded)
        tla.set_flag(mask_loaded)
        tla.wait_flag(mask_loaded)

        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(
                src_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )
            mask_tile = tla.tile_view(
                mask_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )
            dst_tile = tla.tile_view(
                dst_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )

            src_reg = src_tile.load()
            mask_reg = mask_tile.load(_MASK_LOAD_PARAMS)
            zero = tla.sub(src_reg, src_reg)
            dst_tile.store(tla.where(mask_reg, src_reg, zero))

        tla.set_flag(done)
        tla.wait_flag(done)

        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _set_case(op: str, mask_dtype: str) -> None:
    global _MASK_LOAD_PARAMS, _MASK_DTYPE
    dist = MaskLoadDist.DIST_DS if op == "ds" else MaskLoadDist.DIST_US
    _MASK_LOAD_PARAMS = MaskLoadParams(load_dist=dist)
    _MASK_DTYPE = _MASK_DTYPE_MAP[mask_dtype]


def _runtime_tensor(dev_buf: Any) -> Any:
    return tla.from_dlpack(
        dev_buf.contiguous(),
        layout_tag=tla.arch.RowMajor,
    )


def _mask_tensor_with_byte_pattern(
    torch: Any, *, op: str, mask_dtype: str, device: str
) -> tuple[Any, Any]:
    """Build a VL=64 mask tile whose first 64 **bytes** match the i8 reference."""
    import numpy as np

    torch_name = _TORCH_DTYPE_MAP[mask_dtype]
    np_dtype = getattr(np, torch_name)
    host = np.zeros(VECTOR_ELE, dtype=np_dtype)
    host_bytes = host.view(np.uint8)
    pattern = np.zeros(_MASK_PATTERN_BYTES, dtype=np.uint8)
    if op == "ds":
        pattern[: VECTOR_ELE // 2] = 1
    else:
        pattern[:16] = 1
    n = min(_MASK_PATTERN_BYTES, host_bytes.size)
    host_bytes[:n] = pattern[:n]
    mask = torch.from_numpy(host).to(device=device)

    src_ref = torch.linspace(-17.0, 46.0, VECTOR_ELE, dtype=torch.float32)
    if op == "ds":
        # Same byte layout as historical i8 DS e2e → lane i ↔ byte i.
        byte_mask = torch.from_numpy(pattern.astype(np.int8))
        expected = torch.where(
            byte_mask.to(torch.bool), src_ref, torch.zeros_like(src_ref)
        )
    else:
        expected = torch.zeros_like(src_ref)
        for i in range(16):
            if int(pattern[i]) != 0:
                expected[4 * i] = src_ref[4 * i]
    return mask, expected


def _run_case(
    *,
    op: str,
    mask_dtype: str,
    device: int,
    atol: float,
) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    _set_case(op, mask_dtype)

    src = torch.linspace(-17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu")
    dst = torch.full((VECTOR_ELE,), -999.0, dtype=torch.float32, device="npu")
    mask, expected_cpu = _mask_tensor_with_byte_pattern(
        torch, op=op, mask_dtype=mask_dtype, device="npu"
    )
    expected = expected_cpu.to(device="npu")

    op_tag = f"mask_{op}"
    dtype_tag = f"f32+{mask_dtype}"
    layout_tag = "us_pack" if op == "us" else "row"

    tla_src = _runtime_tensor(src)
    tla_mask = _runtime_tensor(mask)
    tla_dst = _runtime_tensor(dst)

    artifact = tla.compile(
        load_mask_dist_kernel,
        tla_src,
        tla_mask,
        tla_dst,
        options="--npu-arch 3510",
    )
    artifact(tla_src, tla_mask, tla_dst, block_num=1)
    torch.npu.synchronize()

    ok = bool(torch.isclose(dst, expected, rtol=0.0, atol=atol).all())
    mismatch = torch.isclose(dst, expected, rtol=0.0, atol=atol).logical_not()
    first = None
    if mismatch.any():
        index = int(mismatch.nonzero(as_tuple=False)[0].item())
        first = {
            "index": index,
            "actual": dst[index].item(),
            "expected": expected[index].item(),
        }

    print(
        f"compile_ok=True host=torch_npu op={op_tag} "
        f"dtype={dtype_tag} layout={layout_tag}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"output equals expected {op_tag}? {ok}")
    print(f"first mismatch={first}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "MaskLoadParams DIST_US/DIST_DS e2e: AscendC-aligned b8/b16/b32/b64 "
            "mask UB + tla.where on f32 companion."
        )
    )
    parser.add_argument(
        "op",
        choices=("ds", "us"),
        help="ds | us",
    )
    parser.add_argument(
        "--mask-dtype",
        choices=tuple(_MASK_DTYPE_MAP.keys()),
        default="i8",
        help="Mask UB element type (AscendC LoadAlign<T>); default i8",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    return _run_case(
        op=args.op,
        mask_dtype=args.mask_dtype,
        device=args.device,
        atol=args.atol,
    )


if __name__ == "__main__":
    raise SystemExit(main())
