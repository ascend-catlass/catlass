# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check ReLU, PReLU, Leaky ReLU, and Int64 maximum on Ascend 950."""

import argparse

import catlass.tla as tla


@tla.jit
def _ub_tensor(size, dtype):
    ptr = tla.allocate(size, dtype, tla.AddressSpace.ub, 256)
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(size), tla.make_stride(1))
    )


@tla.kernel
def _relu(
    src: tla.Tensor,
    out: tla.Tensor,
    dtype: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
    masked: tla.Constexpr[bool],
) -> None:
    buf = _ub_tensor(lanes, dtype)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    computed = tla.flag("computed", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(buf, src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            x = buf.load()
            if tla.const_expr(masked):
                mask = tla.create_mask(pattern=tla.mask.VL16, dtype=dtype)
                buf.store(tla.relu(x, mask=mask))
            else:
                buf.store(tla.relu(x))
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(out, buf)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def _max_int64(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    vector_out: tla.Tensor,
    scalar_out: tla.Tensor,
    masked: tla.Constexpr[bool],
) -> None:
    lhs_ub = _ub_tensor(32, tla.Int64)
    rhs_ub = _ub_tensor(32, tla.Int64)
    vector_ub = _ub_tensor(32, tla.Int64)
    scalar_ub = _ub_tensor(32, tla.Int64)
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
            threshold = rhs[0]
            if tla.const_expr(masked):
                mask = tla.create_mask(pattern=tla.mask.VL16, dtype=tla.Int64)
                vector_ub.store(tla.max(x, y, mask=mask))
                scalar_ub.store(tla.max(x, threshold, mask=mask))
            else:
                vector_ub.store(tla.max(x, y))
                scalar_ub.store(tla.max(x, threshold))
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(vector_out, vector_ub)
        tla.copy(scalar_out, scalar_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def _relu_kernel(
    src: tla.Tensor,
    slopes: tla.Tensor,
    prelu_out: tla.Tensor,
    leaky_out: tla.Tensor,
    dtype: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
    masked: tla.Constexpr[bool],
) -> None:
    src_ub = _ub_tensor(lanes, dtype)
    slope_ub = _ub_tensor(lanes, dtype)
    prelu_ub = _ub_tensor(lanes, dtype)
    leaky_ub = _ub_tensor(lanes, dtype)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    computed = tla.flag("computed", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(src_ub, src)
        tla.copy(slope_ub, slopes)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            x = src_ub.load()
            slope = slope_ub.load()
            if tla.const_expr(masked):
                mask = tla.create_mask(pattern=tla.mask.VL16, dtype=dtype)
                prelu_ub.store(tla.prelu(x, slope, mask=mask))
                leaky_ub.store(tla.leaky_relu(x, slopes[0], mask=mask))
            else:
                prelu_ub.store(tla.prelu(x, slope))
                leaky_ub.store(tla.leaky_relu(x, 0.25))
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(prelu_out, prelu_ub)
        tla.copy(leaky_out, leaky_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def main():
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    cases = ((tla.Float16, torch.float16, 128), (tla.Float32, torch.float32, 64))
    for dtype, torch_dtype, lanes in (
        *cases,
        (tla.Int32, torch.int32, 64),
        (tla.Int64, torch.int64, 32),
    ):
        src = (torch.arange(lanes) % 9 - 4).to(torch_dtype)
        if dtype is tla.Int64:
            src *= 1 << 40
            # Cover both 32-bit words and the sign at the edges of the range.
            src[:6] = torch.tensor(
                [-(1 << 63), -1, 0, 1, (1 << 32) - 1, (1 << 63) - 1],
                dtype=torch_dtype,
            )
        out = torch.empty_like(src).npu()
        inputs = tuple(
            from_dlpack(buf, layout_tag=tla.arch.RowMajor) for buf in (src.npu(), out)
        )
        for masked in (False, True):
            artifact = tla.compile(
                _relu, *inputs, dtype, lanes, masked, options="--npu-arch 3510"
            )
            artifact(*inputs, block_num=1)
            torch.npu.synchronize()
            expected = src.clamp_min(0)
            if masked:
                expected[16:] = 0
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            print(f"PASS ReLU {torch_dtype} masked={masked}")

    # The first pair has equal high words and low words straddling the signed
    # 32-bit boundary. rhs[0] also supplies a runtime scalar with two nonzero words.
    lhs = torch.tensor(
        [0x1_7FFFFFFF, 0x1_80000000, 0x2_00000000, 0x0_FFFFFFFF, -1, -(1 << 63), 9, -7]
        * 4,
        dtype=torch.int64,
    )
    rhs = torch.tensor(
        [0x1_80000000, 0x1_7FFFFFFF, 0x1_FFFFFFFF, 0x1_00000000, -9, -11, 7, -3] * 4,
        dtype=torch.int64,
    )
    vector_out, scalar_out = torch.empty_like(lhs).npu(), torch.empty_like(lhs).npu()
    inputs = tuple(
        from_dlpack(buf, layout_tag=tla.arch.RowMajor)
        for buf in (lhs.npu(), rhs.npu(), vector_out, scalar_out)
    )
    for masked in (False, True):
        artifact = tla.compile(_max_int64, *inputs, masked, options="--npu-arch 3510")
        artifact(*inputs, block_num=1)
        torch.npu.synchronize()
        for out, other in ((vector_out, rhs), (scalar_out, rhs[0])):
            expected = torch.maximum(lhs, other)
            if masked:
                expected[16:] = 0
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        print(f"PASS Int64 vector and scalar maximum masked={masked}")

    for dtype, torch_dtype, lanes in cases:
        indices = torch.arange(lanes)
        src = (indices % 9 - 4).to(torch_dtype)
        src[:2] = torch.tensor([0.0, -0.0], dtype=torch_dtype)
        keep_source = (src > 0) | ((src == 0) & ~torch.signbit(src))
        slopes = ((indices % 7 - 2) * 0.125).to(torch_dtype)
        for masked in (False, True):
            buffers = (
                src.npu(),
                slopes.npu(),
                torch.empty_like(src).npu(),
                torch.empty_like(src).npu(),
            )
            inputs = tuple(
                from_dlpack(buf, layout_tag=tla.arch.RowMajor) for buf in buffers
            )
            artifact = tla.compile(
                _relu_kernel, *inputs, dtype, lanes, masked, options="--npu-arch 3510"
            )
            artifact(*inputs, block_num=1)
            torch.npu.synchronize()

            scalar = slopes[0] if masked else 0.25
            for out, slope in zip(buffers[2:], (slopes, scalar)):
                expected = torch.where(keep_source, src, src * slope)
                if masked:
                    expected[16:] = 0
                actual = out.cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if masked and torch.signbit(actual[16:]).any():
                    raise AssertionError("Masked lanes must contain positive zero")
            print(f"PASS {torch_dtype} masked={masked}")

            if masked:
                buffers[1].copy_(torch.full_like(slopes, float("inf")))
                artifact(*inputs, block_num=1)
                torch.npu.synchronize()
                for name, out in zip(("PReLU", "Leaky ReLU"), buffers[2:]):
                    positive_zero, negative_zero = out.cpu()[:2]
                    if positive_zero != 0.0 or torch.signbit(positive_zero):
                        raise AssertionError(f"{name} must preserve +0.0")
                    if not torch.isnan(negative_zero):
                        raise AssertionError(
                            f"{name} must return NaN for -0.0 with +inf slope"
                        )
                print(f"PASS {torch_dtype} signed zero with +inf slope")


if __name__ == "__main__":
    main()
