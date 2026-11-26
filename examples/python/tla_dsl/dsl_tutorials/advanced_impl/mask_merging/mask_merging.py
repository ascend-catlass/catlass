# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Merge selected results with the previous destination vector."""

import argparse

import catlass.tla as tla


LANES = 64
ROUND_EVEN = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_RINT)
OPERATIONS = ("add", "max", "min", "full", "move", "to_i32", "to_f32")


def ub_like(tensor):
    ptr = tla.allocate(LANES, tensor.element_type, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, tensor, tla.arch.RowMajor)


def selected_value(op, a, b, mask):
    if op == "add":
        return tla.add(a, b, mask=mask)
    if op == "max":
        return tla.max(a, b, mask=mask)
    if op == "min":
        return tla.min(a, b, mask=mask)
    if op == "to_i32":
        return a.to(tla.Int32, ROUND_EVEN, mask=mask)
    if op == "to_f32":
        return a.to(tla.Float32, ROUND_EVEN, mask=mask)
    return a  # Move selects the source directly.


@tla.kernel
def mask_merging(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_initial: tla.Tensor,
    mem_out: tla.Tensor,
    op: tla.Constexpr,
) -> None:
    ub_a = ub_like(mem_a)
    ub_b = ub_like(mem_b)
    output_buffer = ub_like(mem_out)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(ub_a, mem_a)
        tla.copy(ub_b, mem_b)
        tla.copy(output_buffer, mem_initial)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            b = ub_b.load()
            mask = tla.cmp(b, 0, "gt")
            previous = output_buffer.load()
            result = selected_value(op, ub_a.load(), b, mask)
            output_buffer.store(tla.where(mask, result, previous))
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(mem_out, output_buffer)


@tla.kernel
def mask_merging_full(
    mem_b: tla.Tensor,
    mem_initial: tla.Tensor,
    mem_out: tla.Tensor,
) -> None:
    ub_b = ub_like(mem_b)
    output_buffer = ub_like(mem_out)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(ub_b, mem_b)
        tla.copy(output_buffer, mem_initial)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            b = ub_b.load()
            mask = tla.cmp(b, 0, "gt")
            previous = output_buffer.load()
            result = tla.full(7.0, dtype=tla.Float32, mask=mask)
            output_buffer.store(tla.where(mask, result, previous))
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(mem_out, output_buffer)


def run(op, torch):
    lane = torch.arange(LANES)
    src_dtype = torch.int32 if op == "to_f32" else torch.float32
    dst_dtype = torch.int32 if op == "to_i32" else torch.float32
    a = (lane % 23 - 11).to(src_dtype)
    if src_dtype == torch.float32:
        a += 0.5
    initial_values = torch.arange(90, 90 + LANES, dtype=dst_dtype)
    if dst_dtype == torch.float32:
        initial_values[2] = -0.0
        initial_values.view(torch.int32)[3] = 0x7FC01234

    device_b = torch.empty_like(a, device="npu")
    device_out = torch.empty_like(initial_values, device="npu")
    buffers = (device_b, initial_values.to("npu"), device_out)
    tensors = [tla.from_dlpack(v, layout_tag=tla.arch.RowMajor) for v in buffers]
    if op == "full":
        kernel = tla.compile(mask_merging_full, *tensors, options="--npu-arch 3510")
    else:
        device_a = a.to("npu")
        tensors.insert(0, tla.from_dlpack(device_a, layout_tag=tla.arch.RowMajor))
        kernel = tla.compile(mask_merging, *tensors, op, options="--npu-arch 3510")

    for active in (lane >= 0, lane < 0, lane % 2 == 0, lane < 2):
        b = torch.where(active, 4, -4).to(src_dtype)
        device_b.copy_(b)
        device_out.fill_(-999)
        kernel(*tensors, block_num=1)
        torch.npu.synchronize()

        if op == "add":
            candidate = a + b
        elif op == "max":
            candidate = torch.maximum(a, b)
        elif op == "min":
            candidate = torch.minimum(a, b)
        elif op == "full":
            candidate = torch.full_like(a, 7)
        elif op == "to_i32":
            candidate = a.round().to(dst_dtype)
        else:
            candidate = a.to(dst_dtype)
        expected_bits = torch.where(
            active, candidate.view(torch.int32), initial_values.view(torch.int32)
        )
        torch.testing.assert_close(
            device_out.cpu().view(torch.int32), expected_bits, rtol=0, atol=0
        )
    print(f"{op}: all four masks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    for op in OPERATIONS:
        run(op, torch)


if __name__ == "__main__":
    main()
