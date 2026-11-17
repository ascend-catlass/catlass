# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end ``tla.full`` runtime-scalar broadcast across dtypes.

Every case fills one UB tile from a runtime scalar and copies it out; the
cases differ only in where the scalar comes from and its dtype:

- scalar ABI parameter: ``flag`` matches the vector dtype exactly
  (``i8``/``i16`` launch typed scalars, ``i32``/``f32`` launch raw values);
- tensor input: the scale tensor is the vector dtype itself and the
  ``[]``-read scalar matches ``dtype`` exactly -- no scalar ABI parameter
  (the supported form for f16/bf16, whose kernel ABI scalar parameters
  cannot combine with SIMD vector-helper scalar calls on device).

Battery case: ``full-scalar-broadcast``.
"""

from __future__ import annotations

import argparse

from catlass import tla


def _flat(ptr, n):
    return tla.make_tensor(ptr, tla.make_layout(tla.make_shape(n), tla.make_stride(1)))


def make_fill_kernel(out_dtype, vl):
    """Fill ``vl`` elements from a scalar kernel ABI parameter of ``out_dtype``."""

    @tla.kernel
    def fill_kernel(gm_out: tla.Tensor, flag: out_dtype) -> None:
        p_out = tla.allocate(vl, out_dtype, tla.AddressSpace.ub, 512)
        ub_out = _flat(p_out, vl)
        st = tla.flag("fsb_st", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            with tla.vec.func(mode="simd"):
                out_tile = tla.tile_view(ub_out, tla.make_shape(vl), tla.make_coord(0))
                out_tile.store(tla.full(flag, out_dtype))
            tla.set_flag(st)
            tla.wait_flag(st)
            tla.copy(_flat(gm_out.ptr, vl), ub_out)
            tla.pipe_barrier(tla.pipes.ALL)

    return fill_kernel


def make_fill_from_tensor_kernel(out_dtype, vl):
    """Fill ``vl`` elements from the scalar read off a scale tensor (the
    scale tensor carries ``out_dtype`` itself)."""

    @tla.kernel
    def fill_from_tensor(gm_scale: tla.Tensor, gm_out: tla.Tensor) -> None:
        p_out = tla.allocate(vl, out_dtype, tla.AddressSpace.ub, 512)
        ub_out = _flat(p_out, vl)
        st = tla.flag("fsb_t_st", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            with tla.vec.func(mode="simd"):
                scale = gm_scale[0]
                out_tile = tla.tile_view(ub_out, tla.make_shape(vl), tla.make_coord(0))
                out_tile.store(tla.full(scale, out_dtype))
            tla.set_flag(st)
            tla.wait_flag(st)
            tla.copy(_flat(gm_out.ptr, vl), ub_out)
            tla.pipe_barrier(tla.pipes.ALL)

    return fill_from_tensor


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    torch_dtype = {
        tla.Int8: torch.int8,
        tla.Int16: torch.int16,
        tla.Int32: torch.int32,
        tla.Float16: torch.float16,
        tla.BFloat16: torch.bfloat16,
        tla.Float32: torch.float32,
    }

    torch.npu.set_device(args.device)

    def bind(buf):
        return from_dlpack(buf.contiguous(), layout_tag=tla.arch.RowMajor)

    def check(name, out, expected, atol):
        got = out.cpu().float()
        # torch.allclose rejects NaN/inf by default, so a non-finite output
        # cannot pass the tolerance check.
        if not torch.allclose(got, torch.full_like(got, expected), rtol=0.0, atol=atol):
            raise AssertionError(
                f"{name}: expected all {expected}, got head={got[:8].tolist()}"
            )
        print(f"PASS case={name} value={expected} head={got[:4].tolist()}")

    passed = 0

    # Scalar ABI parameter cases: i8/i16 launch typed scalars (kernel ABI),
    # i32/f32 launch raw values. One register per dtype: 256B.
    scalar_cases = (
        ("fill-i32", tla.Int32, 64, 9, 0.0),
        ("fill-f32", tla.Float32, 64, 2.5, 1e-6),
        ("fill-i8", tla.Int8, 256, 9, 0.0),
        ("fill-i16", tla.Int16, 128, 9, 0.0),
    )
    for name, dtype, vl, value, atol in scalar_cases:
        kernel = make_fill_kernel(dtype, vl)
        launch = value if dtype in (tla.Int32, tla.Float32) else dtype(value)
        out = torch.zeros(vl, dtype=torch_dtype[dtype]).npu()
        to = bind(out)
        torch.npu.synchronize()
        artifact = tla.compile(kernel, to, launch, options="--npu-arch 3510")
        artifact(to, launch, block_num=args.block_num)
        torch.npu.synchronize()
        check(name, out, value, atol)
        passed += 1

    # Tensor-input cases: the scale tensor carries the vector dtype itself, so
    # the ``[]``-read scalar matches ``dtype`` exactly. This is the supported
    # form for f16/bf16 (16-bit float kernel ABI parameters cannot combine
    # with SIMD vector-helper scalar calls on device).
    tensor_cases = (
        ("fill-f16-from-tensor", tla.Float16, 128, 2.5, 1e-6),
        ("fill-bf16-from-tensor", tla.BFloat16, 128, 2.5, 1e-6),
    )
    for name, dtype, vl, value, atol in tensor_cases:
        kernel = make_fill_from_tensor_kernel(dtype, vl)
        out = torch.zeros(vl, dtype=torch_dtype[dtype]).npu()
        scale = torch.full((1,), value, dtype=torch_dtype[dtype]).npu()
        ts, to = bind(scale), bind(out)
        torch.npu.synchronize()
        artifact = tla.compile(kernel, ts, to, options="--npu-arch 3510")
        artifact(ts, to, block_num=args.block_num)
        torch.npu.synchronize()
        check(name, out, value, atol)
        passed += 1

    print(f"tla.full runtime scalar: {passed} passed, 0 failed, 0 skipped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
