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
import re

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def vf_print_mixed(source: tla.Tensor, destination: tla.Tensor):
    source_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    result_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.cube():
        tla.print("cube_block={}", tla.arch.block_idx())
        tla.print(source, 4)
    with tla.vector():
        tile_index = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
        source_tile = tla.tile_view(
            source, tla.make_shape(64), tla.make_coord(tile_index)
        )
        result_tile = tla.tile_view(
            destination, tla.make_shape(64), tla.make_coord(tile_index)
        )
        source_ub = tla.make_tensor_like(source_ptr, source_tile, tla.arch.RowMajor)
        result_ub = tla.make_tensor_like(result_ptr, result_tile, tla.arch.RowMajor)
        tla.copy(source_ub, source_tile)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.print("vector_before={}", tile_index)
        tla.print(source_tile, 4)
        with tla.vec.func(mode="simd"):
            result = source_ub.load() * 2.0 + 1.0
            tla.print(result, 4)
            result_ub.store(result)
            tla.print(result_ub, 4)
        tla.print(result_ub, 4)
        tla.print("vector_after={}", tile_index)
        tla.set_flag(stored)
        tla.wait_flag(stored)
        tla.copy(result_tile, result_ub)
        with tla.vec.func(mode="simt", thread_block_dim=32):
            thread, _, _ = tla.arch.thread_idx()
            if thread == 0:
                tla.print("simt_tile={} value={}", tile_index, source_tile[0])
                tla.print(source_tile, 4)
        tla.pipe_barrier(tla.pipes.ALL)


def _verify_mixed_print_output(
    output: str,
    *,
    block_num: int,
    source_values: list[float],
    expected_values: list[float],
) -> None:
    record = re.compile(
        r"DumpTensor: call=([01]), block=(\d+), subblock=([01]), data_type=float32, position=(REG|UB), shape=\[(4|64)\] dump_size=4 \[(.*?)\]"
    )
    matches = record.findall(output)
    if len(matches) != block_num * 4:
        raise RuntimeError(f"Unexpected VF record count: {output!r}")
    seen = set()
    for call, block, subblock, position, shape, values in matches:
        if int(call) != (0 if position == "REG" else 1):
            raise RuntimeError(f"Unexpected VF call identifier: {output!r}")
        key = (int(block), int(subblock), position)
        if key in seen or not 0 <= key[0] < block_num:
            raise RuntimeError(f"Duplicate or invalid VF record: {output!r}")
        seen.add(key)
        if shape != ("4" if position == "REG" else "64"):
            raise RuntimeError(f"Unexpected VF shape: {output!r}")
        start = (key[0] * 2 + key[1]) * 64
        if [float(value) for value in values.split(",")] != expected_values[
            start : start + 4
        ]:
            raise RuntimeError(f"Unexpected VF values: {output!r}")
    for block in range(block_num):
        if output.count(f"cube_block={block}\n") != 1:
            raise RuntimeError(f"Unexpected Cube scalar output: {output!r}")
    for tile in range(block_num * 2):
        expected_value = source_values[tile * 64]
        if output.count(f"simt_tile={tile} value={expected_value:f}\n") != 1:
            raise RuntimeError(f"Unexpected SIMT scalar output: {output!r}")
    # Native AIV GM records must retain subblock identity even though AIC
    # uses the same dtype and storage in this kernel.
    native_gm = re.findall(
        r"DumpTensor: call=2, block=(\d+), subblock=([01]), "
        r"data_type=float32, position=GM, shape=\[64\] dump_size=4 \[(.*?)\]",
        output,
    )
    if len(native_gm) != block_num * 2:
        raise RuntimeError(f"Unexpected AIV GM record count: {output!r}")
    for block, subblock, values in native_gm:
        start = (int(block) * 2 + int(subblock)) * 64
        if [float(value) for value in values.split(",")] != source_values[
            start : start + 4
        ]:
            raise RuntimeError(f"Unexpected AIV GM values: {output!r}")
    if output.count("position=GM") != block_num * 3:
        raise RuntimeError(f"Unexpected GM record count: {output!r}")
    if output.count("DumpTensor: call=3,") != block_num * 2:
        raise RuntimeError(f"Unexpected outer UB record count: {output!r}")
    for tile in range(block_num * 2):
        if output.count(f"vector_before={tile}\n") != 1:
            raise RuntimeError(f"Unexpected Vector entry output: {output!r}")
        if output.count(f"vector_after={tile}\n") != 1:
            raise RuntimeError(f"Unexpected Vector exit output: {output!r}")
    if output.count("DumpTensor: block=(") != block_num * 2:
        raise RuntimeError(f"Unexpected SIMT Tensor record count: {output!r}")
    if "malformed" in output or "no records captured" in output:
        raise RuntimeError(f"Invalid print records: {output!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print both AIV subblocks of mixed Cube+Vector kernels."
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--block-num", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.run and args.device is None:
        parser.error("--run requires --device")
    elements = args.block_num * 2 * 64
    tensor = make_fake_tensor(
        tla.Float32,
        (elements,),
        (1,),
        origin_shape=(elements,),
        layout_tag=tla.arch.RowMajor,
    )
    compiled = vf_print_mixed.compile(
        type_args=(tensor, tensor), options="--npu-arch 3510"
    )
    print(f"kernel.o path={compiled.kernel_binary_path}")
    if not args.run:
        return
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    try:
        from .debug_print import _capture_c_stdout
    except ImportError:
        from debug_print import _capture_c_stdout

    torch.npu.set_device(args.device)
    for update in (0, 1):
        source = (
            torch.arange(elements, dtype=torch.float32, device=f"npu:{args.device}")
            + update
        )
        destination = torch.empty_like(source)
        output = _capture_c_stdout(
            lambda: compiled(
                from_dlpack(source, layout_tag=tla.arch.RowMajor),
                from_dlpack(destination, layout_tag=tla.arch.RowMajor),
                block_num=args.block_num,
            )
        )
        source_host = source.cpu()
        expected = source_host * 2 + 1
        torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
        _verify_mixed_print_output(
            output,
            block_num=args.block_num,
            source_values=source_host.tolist(),
            expected_values=expected.tolist(),
        )
        print(output, end="")
    print("output_ok=True")


if __name__ == "__main__":
    main()
