# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scatter register values into UB using sparse element indices.

Each dtype checks full, M3, and zero masks in separate nonzero-offset subviews.
The whole allocation is compared byte-for-byte, including guard windows,
sparse holes, and masked slots. For 8-bit data, scatter reads src[2*j].
"""

import argparse
from typing import Any

import catlass.tla as tla

# (TLA dtype, torch dtype name, element bytes, destination sentinel).
_DTYPES = {
    "f32": (tla.Float32, "float32", 4, -999.0),
    "f16": (tla.Float16, "float16", 2, -999.0),
    "bf16": (tla.BFloat16, "bfloat16", 2, -999.0),
    "i8": (tla.Int8, "int8", 1, -128),
    "u8": (tla.UInt8, "uint8", 1, 255),
    "i16": (tla.Int16, "int16", 2, -32768),
    "u16": (tla.UInt16, "uint16", 2, 65535),
    "i32": (tla.Int32, "int32", 4, -2147483648),
    "u32": (tla.UInt32, "uint32", 4, 4294967295),
}

# Refreshed before each compile, including repeated battery invocations.
_DATA_DTYPE = tla.Float32
_INDEX_COPY_DTYPE = tla.Int32
_INDEX_DTYPE = tla.UInt32
_MASK_DTYPE = tla.Float32
SRC_ELE = 64
INDEX_ELE = 64
SCATTER_LANES = 64
WINDOW_ELE = 128
DST_ELE = 7 * WINDOW_ELE


@tla.kernel
def scatter_op(mem_src: tla.Tensor, mem_idx: tla.Tensor, mem_dst: tla.Tensor) -> None:
    loaded = tla.flag("scatter_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("scatter_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_ub = tla.make_tensor_like(
        tla.allocate(SRC_ELE, _DATA_DTYPE, tla.AddressSpace.ub, 256),
        mem_src,
        tla.arch.RowMajor,
    )
    idx_ub = tla.make_tensor_like(
        tla.allocate(INDEX_ELE, _INDEX_COPY_DTYPE, tla.AddressSpace.ub, 256),
        mem_idx,
        tla.arch.RowMajor,
    )
    dst_ub = tla.make_tensor_like(
        tla.allocate(DST_ELE, _DATA_DTYPE, tla.AddressSpace.ub, 256),
        mem_dst,
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(src_ub, mem_src)
        tla.copy(idx_ub, mem_idx)
        tla.copy(dst_ub, mem_dst)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)

        with tla.vec.func(mode="simd"):
            values = src_ub.load()
            indices = idx_ub.load().bitcast(_INDEX_DTYPE)
            full_dst = tla.tile_view(
                dst_ub, tla.make_shape(WINDOW_ELE), tla.make_coord(1)
            )
            partial_dst = tla.tile_view(
                dst_ub, tla.make_shape(WINDOW_ELE), tla.make_coord(3)
            )
            zero_dst = tla.tile_view(
                dst_ub, tla.make_shape(WINDOW_ELE), tla.make_coord(5)
            )
            tla.scatter(full_dst, values, indices)
            partial = tla.create_mask(pattern=tla.mask.M3, dtype=_MASK_DTYPE)
            tla.scatter(partial_dst, values, indices, mask=partial)
            zero = tla.create_mask(pattern=tla.mask.ALLF, dtype=_MASK_DTYPE)
            tla.scatter(zero_dst, values, indices, mask=zero)

        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(mem_dst, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _configure(dtype_name: str) -> tuple[str, int, float | int]:
    global _DATA_DTYPE, _INDEX_COPY_DTYPE, _INDEX_DTYPE, _MASK_DTYPE
    global SRC_ELE, INDEX_ELE, SCATTER_LANES, WINDOW_ELE, DST_ELE
    _DATA_DTYPE, torch_name, element_bytes, sentinel = _DTYPES[dtype_name]
    SRC_ELE = 256 // element_bytes
    SCATTER_LANES = 128 if element_bytes == 1 else SRC_ELE
    INDEX_ELE = 128 if element_bytes <= 2 else 64
    _INDEX_COPY_DTYPE = tla.Int16 if element_bytes <= 2 else tla.Int32
    _INDEX_DTYPE = tla.UInt16 if element_bytes <= 2 else tla.UInt32
    _MASK_DTYPE = tla.UInt16 if element_bytes == 1 else _DATA_DTYPE
    WINDOW_ELE = 2 * SCATTER_LANES
    DST_ELE = 7 * WINDOW_ELE
    assert WINDOW_ELE * element_bytes % 32 == 0
    return torch_name, element_bytes, sentinel


def _host_inputs(dtype_name: str, torch: Any) -> tuple[Any, Any, Any, Any]:
    torch_name, element_bytes, sentinel = _configure(dtype_name)
    dtype = getattr(torch, torch_name)
    lanes = torch.arange(SRC_ELE, dtype=torch.int64)
    if dtype_name == "i8":
        values = (29 * lanes + 7) % 211 - 105
    elif dtype_name == "u8":
        values = (29 * lanes + 7) % 211 + 1
    elif dtype_name == "i16":
        values = -30000 + 113 * lanes
    elif dtype_name == "u16":
        values = 40000 + 113 * lanes
    elif dtype_name == "i32":
        values = -1700000000 + 1000003 * lanes
    elif dtype_name == "u32":
        values = 3000000000 + 1000003 * lanes
    else:
        values = lanes - SRC_ELE // 2 + 1
    src = values.to(dtype=dtype)
    j = torch.arange(SCATTER_LANES, dtype=torch.int64)
    # A permutation of odd positions leaves every even slot untouched.
    indices = 2 * ((5 * j + 3) % SCATTER_LANES) + 1
    index_dtype = torch.int16 if element_bytes <= 2 else torch.int32
    idx = indices.to(dtype=index_dtype)
    dst = torch.full((DST_ELE,), sentinel, dtype=dtype)

    source_bytes = src.view(torch.uint8).reshape(SRC_ELE, element_bytes)
    expected_bytes = dst.view(torch.uint8).reshape(DST_ELE, element_bytes).clone()
    source_lanes = 2 * j if element_bytes == 1 else j
    expected_bytes[WINDOW_ELE + indices] = source_bytes[source_lanes]
    active = j % 3 == 0
    expected_bytes[3 * WINDOW_ELE + indices[active]] = source_bytes[
        source_lanes[active]
    ]
    return src, idx, dst, expected_bytes.flatten()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dtype", choices=("all", *_DTYPES), default="f32")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run to compile and execute on NPU")

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    names = tuple(_DTYPES) if args.dtype == "all" else (args.dtype,)
    for dtype_name in names:
        src, idx, dst, expected_bytes = _host_inputs(dtype_name, torch)
        dev_src, dev_idx, dev_dst = (t.to(device="npu") for t in (src, idx, dst))
        runtime_args = tuple(
            tla.from_dlpack(t, layout_tag=tla.arch.RowMajor)
            for t in (dev_src, dev_idx, dev_dst)
        )
        artifact = tla.compile(scatter_op, *runtime_args, options="--npu-arch 3510")
        artifact(*runtime_args, block_num=1)
        torch.npu.synchronize()
        actual_bytes = dev_dst.cpu().view(torch.uint8).flatten()
        torch.testing.assert_close(actual_bytes, expected_bytes, rtol=0, atol=0)
        native_index = "u16" if _DTYPES[dtype_name][2] <= 2 else "u32"
        print(
            f"compile_ok=True host=torch_npu op=scatter dtype={dtype_name} "
            f"index={native_index} lanes={SCATTER_LANES} masks=full,M3,zero layout=row"
        )
        print(f"kernel.o path={artifact.kernel_binary_path}")
        print("launch_ok=True output_bitwise_equal=True sentinel_preserved=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
