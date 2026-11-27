# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check SIMD arithmetic and logical dtype coverage on Ascend 950."""

import argparse
import operator

import numpy as np

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


def make_kernel(op, dtype, arity, mask_kind):
    lanes = 2048 // dtype.width if dtype is not tla.Bool else 256

    @tla.kernel
    def compute(src: tla.Tensor, rhs: tla.Tensor, dst: tla.Tensor) -> None:
        shape = tla.make_shape(lanes)
        layout = tla.make_layout(shape, tla.make_stride(1))
        ub_lhs = tla.make_tensor(
            tla.allocate(lanes, dtype, tla.AddressSpace.ub, 256), layout
        )
        ub_rhs = tla.make_tensor(
            tla.allocate(lanes, dtype, tla.AddressSpace.ub, 256), layout
        )
        ub_out = tla.make_tensor(
            tla.allocate(lanes, dtype, tla.AddressSpace.ub, 256), layout
        )
        loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
        computed = tla.flag("computed", tla.pipes.VECTOR, tla.pipes.MTE3)
        with tla.vector():
            tla.copy(ub_lhs, src)
            if tla.const_expr(arity == "vector"):
                tla.copy(ub_rhs, rhs)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                lhs = ub_lhs.load()
                mask = None
                if tla.const_expr(mask_kind == "negative"):
                    mask = tla.cmp(lhs, 0, mode="lt")
                elif tla.const_expr(mask_kind == "prefix16"):
                    mask = tla.create_mask(pattern=tla.mask.VL16, dtype=dtype)
                if tla.const_expr(arity == "unary"):
                    result = op(lhs, mask=mask)
                elif tla.const_expr(arity == "scalar"):
                    result = op(lhs, rhs[0], mask=mask)
                else:
                    result = op(lhs, ub_rhs.load(), mask=mask)
                ub_out.store(result)
            tla.set_flag(computed)
            tla.wait_flag(computed)
            tla.copy(dst, ub_out)
            tla.pipe_barrier(tla.pipes.ALL)

    return compute


def trunc_div(x, y):
    if (x < 0) != (y < 0):
        return -(abs(x) // abs(y))
    return abs(x) // abs(y)


def _cases():
    for dtype in (tla.Int64, tla.UInt64):
        for op, ref in (
            (tla.add, operator.add),
            (tla.sub, operator.sub),
            (tla.mul, operator.mul),
            (tla.div, trunc_div),
            (tla.max, max),
            (tla.min, min),
        ):
            for arity in ("vector", "scalar"):
                yield dtype, arity, op, ref
    for dtype, op, ref in (
        (tla.UInt8, tla.add, operator.add),
        (tla.UInt16, tla.div, trunc_div),
        (tla.UInt32, tla.max, max),
        (tla.BFloat16, tla.add, np.add),
        (tla.BFloat16, tla.mul, np.multiply),
        (tla.BFloat16, tla.max, np.maximum),
        (tla.BFloat16, tla.min, np.minimum),
    ):
        yield dtype, "scalar", op, ref
    for dtype in (tla.Int64, tla.UInt64, tla.Bool):
        for op, ref in (
            (tla.bitwise_and, operator.and_),
            (tla.bitwise_or, operator.or_),
            (tla.bitwise_xor, operator.xor),
        ):
            yield dtype, "vector", op, ref
    for dtype in (tla.Int16, tla.UInt8, tla.UInt16, tla.UInt32, tla.Int64, tla.UInt64):
        yield dtype, "unary", tla.bitwise_not, operator.invert
    yield tla.Int64, "unary", tla.abs, abs
    yield tla.Int64, "unary", tla.neg, operator.neg


def run(device):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    passed = 0
    for dtype, arity, op, ref in _cases():
        is_i64_abs = dtype is tla.Int64 and op is tla.abs
        if dtype is tla.Bool:
            lhs = np.arange(256) % 3 == 0
            rhs = np.arange(256) % 5 == 0
            carrier = lhs.dtype
        elif dtype is tla.BFloat16:
            carrier = np.dtype("float32")
            lhs = np.resize(np.array([-5.0, 2.0, 0.0, 0.5], dtype=carrier), 128)
            rhs = np.full_like(lhs, 3)
        else:
            bits = dtype.width
            unsigned = dtype in (tla.UInt8, tla.UInt16, tla.UInt32, tla.UInt64)
            sign_bit = 1 << (bits - 1)
            carrier = np.dtype(f"int{bits}")
            np_dtype = np.dtype(f"uint{bits}" if unsigned else carrier)
            edge = sign_bit if unsigned else -sign_bit
            # Large and small operands exercise high words, carries and the
            # difference between signed and unsigned comparisons.
            lhs = np.resize(
                np.array(
                    [0, 1, 3, edge, edge + 3, sign_bit - 1, 83, 11], dtype=np_dtype
                ),
                2048 // bits,
            )
            if is_i64_abs:
                # Both words must survive in the first and last upper-half lanes.
                lhs[[16, 31]] = [-(1 << 32) - 17, -(2 << 32) - 31]
            first_rhs = edge + 7 if unsigned and op in (tla.max, tla.min) else 3
            rhs = np.resize(
                np.array(
                    [first_rhs, 2, edge + 7, 1, 3, edge + 7, 2, 17], dtype=np_dtype
                ),
                2048 // bits,
            )

        def bind(values):
            buffer = torch.from_numpy(values.view(carrier))
            if dtype is tla.BFloat16:
                buffer = buffer.to(torch.bfloat16)
            return buffer.npu()

        lhs_dev, rhs_dev, dst_dev = bind(lhs), bind(rhs), bind(np.zeros_like(lhs))
        if lhs.dtype.kind == "u":
            torch_dtype = getattr(torch, f"uint{dtype.width}")
            views = (buf.view(torch_dtype) for buf in (lhs_dev, rhs_dev, dst_dev))
        else:
            views = (lhs_dev, rhs_dev, dst_dev)
        bound = tuple(from_dlpack(buf, layout_tag=tla.arch.RowMajor) for buf in views)

        if dtype is tla.BFloat16:
            expected = ref(lhs, rhs[0])
        else:
            x = lhs.tolist()
            if arity == "unary":
                values = [ref(a) for a in x]
            else:
                y = rhs.tolist() if arity == "vector" else [int(rhs[0])] * len(x)
                values = [ref(a, b) for a, b in zip(x, y, strict=True)]
            if lhs.dtype.kind != "b":
                mask_bits = (1 << dtype.width) - 1
                expected = np.array(
                    [int(v) & mask_bits for v in values], dtype=f"uint{dtype.width}"
                ).view(lhs.dtype)
            else:
                expected = np.array(values, dtype=lhs.dtype)

        mask_kinds = (
            (None, "prefix16", "negative") if is_i64_abs else (None, "prefix16")
        )
        for mask_kind in mask_kinds:
            label = f"{dtype.__name__} {op.__name__} {arity} mask={mask_kind}"
            print(label, flush=True)
            kernel = make_kernel(op, dtype, arity, mask_kind)
            artifact = tla.compile(kernel, *bound, options="--npu-arch 3510")
            artifact(*bound, block_num=1)
            torch.npu.synchronize()
            result = (
                dst_dev.float().cpu().numpy()
                if dtype is tla.BFloat16
                else dst_dev.cpu().numpy().view(lhs.dtype)
            )
            want = expected.copy()
            if mask_kind == "negative":
                want[lhs >= 0] = 0
            elif mask_kind == "prefix16":
                if (dtype, op) == (tla.UInt16, tla.div):
                    # Only active lanes are defined.
                    result, want = result[:16], want[:16]
                else:
                    want[16:] = 0
            np.testing.assert_array_equal(result, want, err_msg=label)
            passed += 1
    print(f"PASS: {passed} SIMD arithmetic and logical cases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    run(parser.parse_args().device)


if __name__ == "__main__":
    main()
