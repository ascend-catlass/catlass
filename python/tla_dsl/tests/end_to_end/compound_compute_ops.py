# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check SIMD Axpy, SubAbs, and MulsCast on Ascend 950."""

import argparse

import catlass.tla as tla


@tla.jit
def _ub_tensor(lanes, dtype):
    ptr = tla.allocate(lanes, dtype, tla.AddressSpace.ub, 256)
    layout = tla.make_layout(tla.make_shape(lanes), tla.make_stride(1))
    return tla.make_tensor(ptr, layout)


@tla.kernel
def _binary_kernel(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
    dtype: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
    op: tla.Constexpr[str],
    alpha: tla.Constexpr,
    masked: tla.Constexpr[bool],
) -> None:
    lhs_ub = _ub_tensor(lanes, dtype)
    rhs_ub = _ub_tensor(lanes, dtype)
    out_ub = _ub_tensor(lanes, dtype)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    computed = tla.flag("computed", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(lhs_ub, lhs)
        tla.copy(rhs_ub, rhs)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            x = lhs_ub.load()
            y = rhs_ub.load()
            mask = None
            if tla.const_expr(masked):
                if tla.const_expr(dtype in (tla.Int64, tla.UInt64)):
                    mask = tla.create_mask(pattern=tla.mask.VL16, dtype=dtype)
                else:
                    mask = tla.create_mask(pattern=tla.mask.M4, dtype=dtype)
            if tla.const_expr(op == "axpy"):
                result = tla.arch.axpy(x, y, alpha, mask=mask)
            else:
                result = tla.arch.sub_abs(x, y, mask=mask)
            # Store the entire result so mask handling is observable.
            out_ub.store(result)
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(out, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def _muls_cast_kernel(
    src: tla.Tensor,
    even: tla.Tensor,
    odd: tla.Tensor,
    scalar: tla.Constexpr[float],
    masked: tla.Constexpr[bool],
) -> None:
    src_ub = _ub_tensor(64, tla.Float32)
    even_ub = _ub_tensor(128, tla.Float16)
    odd_ub = _ub_tensor(128, tla.Float16)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    computed = tla.flag("computed", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(src_ub, src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            x = src_ub.load()
            mask = None
            if tla.const_expr(masked):
                mask = tla.create_mask(pattern=tla.mask.M4, dtype=tla.Float32)
            even_ub.store(tla.arch.muls_cast(x, scalar, mask=mask))
            odd_ub.store(
                tla.arch.muls_cast(x, scalar, reg_slot=tla.params.RegSlot.ONE, mask=mask)
            )
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(even, even_ub)
        tla.copy(odd, odd_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _run(torch, kernel, buffers, *params):
    from catlass.tla.runtime import from_dlpack

    inputs = tuple(from_dlpack(buf, layout_tag=tla.arch.RowMajor) for buf in buffers)
    artifact = tla.compile(kernel, *inputs, *params, options="--npu-arch 3510")
    artifact(*inputs, block_num=1)
    torch.npu.synchronize()


def _check_binary(torch, op):
    cases = [
        (tla.Float16, torch.float16, 128, -0.5),
        (tla.Float32, torch.float32, 64, -0.5),
        (tla.Int64, torch.int64, 32, -3),
    ]
    if op == "axpy":
        cases.append((tla.UInt64, torch.int64, 32, 3))

    for dtype, torch_dtype, lanes, alpha in cases:
        idx = torch.arange(lanes)
        if torch_dtype == torch.int64:
            # Exercise both 32-bit words and carries. The UInt64 case uses
            # int64 views for host arithmetic and transfers.
            lhs = ((idx % 17) - 8) * (1 << 33) + 0xFFFF_FFF7
            rhs = ((idx % 13) + 1) * (1 << 33) + 0x8000_0003
        else:
            lhs = (((idx % 17) - 8) * 0.5).to(torch_dtype)
            rhs = (((idx % 13) + 1) * 0.5 + 0.125).to(torch_dtype)
        expected = lhs * alpha + rhs if op == "axpy" else (lhs - rhs).abs()
        active = idx < 16 if torch_dtype == torch.int64 else idx % 4 == 0
        dst = torch.full_like(lhs, -177).npu()
        buffers = (lhs.npu(), rhs.npu(), dst)
        if dtype is tla.UInt64:
            buffers = tuple(buf.view(torch.uint64) for buf in buffers)

        for masked in (False, True):
            _run(torch, _binary_kernel, buffers, dtype, lanes, op, alpha, masked)
            actual = dst.cpu()
            reference = expected.masked_fill(~active, 0) if masked else expected
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            if (
                masked
                and actual.is_floating_point()
                and torch.signbit(actual[~active]).any()
            ):
                raise AssertionError(f"{op} masked lanes must contain positive zero")
            print(f"PASS {op} {dtype.__name__} masked={masked}")


def _check_muls_cast(torch):
    idx = torch.arange(64)
    src = (idx.to(torch.float32) - 31.25) / 13.0
    scalar = -1.25
    expected = (src * scalar).to(torch.float16)
    even = torch.full((128,), -177, dtype=torch.float16, device="npu")
    odd = torch.full_like(even, -177)
    buffers = (src.npu(), even, odd)

    for masked in (False, True):
        _run(torch, _muls_cast_kernel, buffers, scalar, masked)
        active = idx % 4 == 0 if masked else slice(None)
        # Only the selected slot of each active input lane is specified.
        torch.testing.assert_close(
            even.cpu()[0::2][active], expected[active], rtol=0, atol=0
        )
        torch.testing.assert_close(
            odd.cpu()[1::2][active], expected[active], rtol=0, atol=0
        )
        print(f"PASS muls_cast (both slots) masked={masked}")


def main():
    ops = ("axpy", "sub-abs", "muls-cast")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--op", choices=ops)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    for op in ops if args.op is None else (args.op,):
        if op == "muls-cast":
            _check_muls_cast(torch)
        else:
            _check_binary(torch, op)


if __name__ == "__main__":
    main()
