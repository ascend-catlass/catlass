# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PostMode UPDATE e2e: aligned ``tile.load`` / ``tile.store``.

* ``add`` — arithmetic dtypes with GM↔UB (b8/b16/b32 + f16/bf16/f32).
* ``copy`` — byte encodings that UPDATE can move but not ``add``
  (f8e4m3fn/f8e5m2/f8e8m0/fp4x2); host buffers are uint8 + ``element_type``.

b64 (i64/u64) and ``hif8`` stay frontend UT: no GM↔UB route for those types.
Default ``post_update_stride=0`` advances by the tile extent (= VL);
tail uses ``update_mask``. Also ``--dynamic-extent`` / ``--stride-geoms`` (f32).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import catlass.tla as tla
from catlass.params import NormalLoadParams, NormalStoreParams, PostMode

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    make_tla_tensor,
    require_torch_npu,
    runtime_kwargs,
    vector_kernel_config,
)

VECTOR_ELE = 400
VL_ELE = 64
STRIDE_ELE = 64
GEOM_LOOPS = 3
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
# Arithmetic UPDATE e2e (GM↔UB + add).
ADD_DTYPES = (
    "i8",
    "u8",
    "i16",
    "u16",
    "f16",
    "bf16",
    "i32",
    "u32",
    "f32",
)
# Move-only UPDATE e2e (GM↔UB as bytes + load/store UPDATE, no add).
# hif8 omitted: dma.cpp has no GM↔UB registration for it.
COPY_DTYPES = (
    "f8e4m3fn",
    "f8e5m2",
    "f8e8m0",
    "f4e2m1x2",
    "f4e1m2x2",
)
ALL_DTYPES = ADD_DTYPES + COPY_DTYPES
FLOAT_DTYPES = frozenset({"f16", "bf16", "f32"})
SENTINEL = -7.0
_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)

_LOAD_U = NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
_STORE_U = NormalStoreParams(post_mode=PostMode.POST_MODE_UPDATE)


@tla.kernel
def load_store_update_op(
    mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            full_loops = VECTOR_ELE // VL_ELE
            tail_elems = VECTOR_ELE % VL_ELE
            x_tile = tla.tile_view(x_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            y_tile = tla.tile_view(y_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            for _ in tla.range(full_loops):
                x_reg, x_tile = x_tile.load(_LOAD_U)
                y_reg, y_tile = y_tile.load(_LOAD_U)
                z_tile = z_tile.store(tla.add(x_reg, y_reg), _STORE_U)
            if tail_elems != 0:
                x_tail, x_tile = x_tile.load(_LOAD_U)
                y_tail, y_tile = y_tile.load(_LOAD_U)
                tail, _ = tla.update_mask(tail_elems, dtype=_KERNEL_DTYPE)
                z_tile = z_tile.store(tla.add(x_tail, y_tail), _STORE_U, mask=tail)
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def load_store_update_dynamic_extent_op(
    mem_x: tla.Tensor,
    mem_y: tla.Tensor,
    mem_z: tla.Tensor,
    vl: "index",  # noqa: F821
) -> None:
    """UPDATE scan with dynamic ``make_shape(vl)``."""
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(x_ub, tla.make_shape(vl), tla.make_coord(0))
            y_tile = tla.tile_view(y_ub, tla.make_shape(vl), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(vl), tla.make_coord(0))
            for _ in tla.range(LOOPS):
                x_reg, x_tile = x_tile.load(_LOAD_U)
                y_reg, y_tile = y_tile.load(_LOAD_U)
                z_tile = z_tile.store(tla.add(x_reg, y_reg), _STORE_U)
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def load_store_update_stride_geom_op(
    mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor
) -> None:
    """Align UPDATE with explicit ``post_update_stride != VL`` (overlap or hole)."""
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256),
        x_gm,
        tla.arch.RowMajor,
    )
    y_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256),
        y_gm,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256),
        z_gm,
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.copy(z_ub, z_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(x_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            y_tile = tla.tile_view(y_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            for _ in tla.range(GEOM_LOOPS):
                x_reg, x_tile = x_tile.load(_LOAD_U)
                y_reg, y_tile = y_tile.load(_LOAD_U)
                z_tile = z_tile.store(tla.add(x_reg, y_reg), _STORE_U)
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def load_store_update_copy_op(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """UPDATE load/store memcpy for move-only encodings (no arithmetic)."""
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256),
        x_gm,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256),
        z_gm,
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            full_loops = VECTOR_ELE // VL_ELE
            tail_elems = VECTOR_ELE % VL_ELE
            x_tile = tla.tile_view(x_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            for _ in tla.range(full_loops):
                x_reg, x_tile = x_tile.load(_LOAD_U)
                z_tile = z_tile.store(x_reg, _STORE_U)
            if tail_elems != 0:
                x_tail, x_tile = x_tile.load(_LOAD_U)
                tail, _ = tla.update_mask(tail_elems, dtype=_KERNEL_DTYPE)
                z_tile = z_tile.store(x_tail, _STORE_U, mask=tail)
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "add": {"op": "add", "dtypes": ADD_DTYPES, "default_atol": 1e-4},
        "copy": {"op": "copy", "dtypes": COPY_DTYPES, "default_atol": 0.0},
    }


def _set_kernel_config(
    op_name: str,
    dtype_name: str,
    shape: tuple[int, ...] | None = None,
    *,
    stride: int | None = None,
    geom_loops: int | None = None,
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, STRIDE_ELE, GEOM_LOOPS, LOOPS, VECTOR_ELE
    global _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES, _KERNEL_SHAPE, _LOAD_U, _STORE_U
    del op_name
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    STRIDE_ELE = VL_ELE if stride is None else stride
    if geom_loops is not None:
        GEOM_LOOPS = geom_loops
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    if STRIDE_ELE == VL_ELE:
        _LOAD_U = NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
        _STORE_U = NormalStoreParams(post_mode=PostMode.POST_MODE_UPDATE)
    else:
        _LOAD_U = NormalLoadParams(
            post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=STRIDE_ELE
        )
        _STORE_U = NormalStoreParams(
            post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=STRIDE_ELE
        )
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _expected_stride_geom(x: Any, y: Any, z_sentinel: Any) -> Any:
    """VL-fill + stride-advance (Align decoupling; LWW on overlap)."""
    z = z_sentinel.clone()
    for i in range(GEOM_LOOPS):
        start = i * STRIDE_ELE
        z[start : start + VL_ELE] = (
            x[start : start + VL_ELE] + y[start : start + VL_ELE]
        )
    return z


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    op = getattr(args, "op", "add")
    _, dtype, _ = _set_kernel_config(op, dtype_name, args.shape)
    if dtype is None:
        raise RuntimeError(
            f"torch dtype unavailable for op={op!r} dtype={dtype_name!r}"
        )
    if op == "copy" or dtype_name in COPY_DTYPES:
        # CPU uint8 then .to(npu): NPU arange/zeros reject uint8.
        x = ((torch.arange(VECTOR_ELE) % 17) + 1).to(torch.uint8).to("npu")
        return (x,)
    # Keep integer add inside a small range so b8/u8 cannot wrap.
    # Arithmetic before .to(dtype): torch_npu lacks aclnnAdds for uint32
    # (DT_UINT32 / 161002) — casting first then ``+ 1`` never reaches the kernel.
    x = ((torch.arange(VECTOR_ELE, device="npu") % 17) + 1).to(dtype)
    if dtype_name in FLOAT_DTYPES:
        y = (torch.arange(VECTOR_ELE, device="npu").to(torch.float32) * 0.5 + 2).to(
            dtype
        )
    else:
        y = ((torch.arange(VECTOR_ELE, device="npu") % 13) + 2).to(dtype)
    return x, y


def _expected(_op_name: str, inputs: tuple[Any, ...]) -> Any:
    if _op_name == "copy":
        return inputs[0].clone()
    x, y = inputs
    # torch / torch_npu lack unsigned elementwise add for several dtypes
    # (e.g. uint16 on CPU, uint32 on NPU). Promote via int64; example values
    # stay small enough that wraparound is not in play.
    import torch

    if x.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        return (x.to(torch.int64) + y.to(torch.int64)).to(dtype=x.dtype)
    return x + y


def _unsupported_case(op_name: str, dtype_name: str) -> bool:
    if op_name == "add":
        return dtype_name in COPY_DTYPES
    if op_name == "copy":
        return dtype_name not in COPY_DTYPES
    return True


class _UpdateHarness(DirectVectorOpHarness):
    """Swap kernel / input_count for ``add`` vs move-only ``copy``."""

    def _run_single_case(self, args: Any, dtype_name: str, torch: Any) -> int:
        saved = self.config
        if args.op == "copy":
            self.config = replace(
                saved, kernel=load_store_update_copy_op, input_count=1
            )
        else:
            self.config = replace(saved, kernel=load_store_update_op, input_count=2)
        try:
            return super()._run_single_case(args, dtype_name, torch)
        finally:
            self.config = saved


HARNESS = _UpdateHarness(
    DirectVectorOpConfig(
        description=(
            "PostMode UPDATE e2e: add (arith dtypes) / copy (f8/e8m0/fp4x2); "
            "b64/hif8 via frontend UT"
        ),
        kernel=load_store_update_op,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_unsupported_case,
        print_skip=lambda o, d, s: print(f"skip op={o} dtype={d} shape={s}"),
        script_path=Path(__file__).resolve(),
        float_dtypes=FLOAT_DTYPES,
        input_count=2,
    )
)


_STRIDE_GEOM_CASES = (
    ("overlap", 64, 32, 3),  # stride < VL: later trip LWW-overwrites overlap
    ("hole", 64, 96, 3),  # stride > VL: gap elements stay at sentinel
)


def _run_stride_geoms(args: argparse.Namespace) -> int:
    """Align UPDATE overlap (stride<VL) and hole (stride>VL) correctness."""
    torch = require_torch_npu(args.device, Path(__file__).stem)
    torch.npu.set_device(args.device)
    want = args.geom
    cases = tuple(c for c in _STRIDE_GEOM_CASES if want in ("all", c[0]))
    if not cases:
        raise SystemExit(f"unknown --geom {want!r}; choose overlap|hole|all")
    rc = 0
    for name, vl, stride, loops in cases:
        n = (loops - 1) * stride + vl + 64
        tla_dtype, torch_dt, _ = _set_kernel_config(
            "add", "f32", (n,), stride=stride, geom_loops=loops
        )
        if not (VL_ELE == vl and STRIDE_ELE == stride and STRIDE_ELE != VL_ELE):
            raise SystemExit(
                f"stride-geom config mismatch: want vl={vl} stride={stride}, "
                f"got VL_ELE={VL_ELE} STRIDE_ELE={STRIDE_ELE}"
            )
        x = torch.arange(VECTOR_ELE, dtype=torch_dt, device="npu")
        y = torch.ones(VECTOR_ELE, dtype=torch_dt, device="npu")
        z = torch.full((VECTOR_ELE,), SENTINEL, dtype=torch_dt, device="npu")
        expected = _expected_stride_geom(x, y, z)
        tla_x = make_tla_tensor(x, tla_dtype, _KERNEL_SHAPE)
        tla_y = make_tla_tensor(y, tla_dtype, _KERNEL_SHAPE)
        tla_z = make_tla_tensor(z, tla_dtype, _KERNEL_SHAPE)
        artifact = tla.compile(
            load_store_update_stride_geom_op,
            tla_x,
            tla_y,
            tla_z,
            **runtime_kwargs(args),
        )
        artifact(tla_x, tla_y, tla_z, block_num=args.block_num)
        torch.npu.synchronize()
        ok = bool(torch.equal(z, expected))
        written = int((z.cpu() != SENTINEL).sum())
        print(
            f"--- stride-geom={name} vl={VL_ELE} stride={STRIDE_ELE} "
            f"loops={GEOM_LOOPS} n={VECTOR_ELE} n_written={written} ok={ok} ---"
        )
        if not ok:
            bad = (z.cpu() != expected.cpu()).nonzero(as_tuple=False)
            i = int(bad[0].item())
            print(
                f"first mismatch index={i} act={float(z[i])} exp={float(expected[i])}"
            )
        rc |= 0 if ok else 1
    return rc


def _run_dynamic_extent(args: argparse.Namespace) -> int:
    torch = require_torch_npu(args.device, Path(__file__).stem)
    torch.npu.set_device(args.device)
    shape = args.shape if args.shape is not None else (256,)
    tla_dtype, torch_dt, sentinel = _set_kernel_config("add", "f32", shape)
    if VECTOR_ELE % VL_ELE != 0:
        raise SystemExit(
            f"--dynamic-extent needs vector length multiple of VL={VL_ELE}, "
            f"got {VECTOR_ELE}"
        )
    x = torch.arange(VECTOR_ELE, dtype=torch_dt, device="npu") + 1
    y = (torch.arange(VECTOR_ELE, dtype=torch_dt, device="npu") * 0.5) + 2
    z = torch.full((VECTOR_ELE,), sentinel, dtype=torch_dt, device="npu")
    expected = x + y
    tla_x = make_tla_tensor(x, tla_dtype, _KERNEL_SHAPE)
    tla_y = make_tla_tensor(y, tla_dtype, _KERNEL_SHAPE)
    tla_z = make_tla_tensor(z, tla_dtype, _KERNEL_SHAPE)
    vl = tla.Int32(VL_ELE)
    artifact = tla.compile(
        load_store_update_dynamic_extent_op,
        tla_x,
        tla_y,
        tla_z,
        vl,
        **runtime_kwargs(args),
    )
    artifact(tla_x, tla_y, tla_z, vl, block_num=args.block_num)
    torch.npu.synchronize()
    ok = bool(torch.isclose(z, expected, rtol=0.0, atol=args.atol).all())
    mismatch = (~torch.isclose(z, expected, rtol=0.0, atol=args.atol)).nonzero(
        as_tuple=False
    )
    first = None
    if mismatch.numel():
        i = int(mismatch[0].item())
        first = {"index": i, "actual": z[i].item(), "expected": expected[i].item()}
    print(
        f"compile_ok=True host=torch_npu op=add dtype=f32 shape={shape} "
        f"dynamic_extent=True vl={VL_ELE}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"outputs equal expected add? {ok}")
    print(f"first mismatch={first}")
    return 0 if ok else 1


def main() -> int:
    if "--dynamic-extent" in sys.argv or "--stride-geoms" in sys.argv:
        parser = argparse.ArgumentParser(
            description="UPDATE e2e: dynamic extent or Align stride geoms"
        )
        parser.add_argument("op", nargs="?", default="add")
        parser.add_argument("--device", type=int, default=2)
        parser.add_argument("--block-num", type=int, default=1)
        parser.add_argument("--shape", type=int, nargs="+", default=None)
        parser.add_argument("--atol", type=float, default=1e-4)
        parser.add_argument("--dynamic-extent", action="store_true")
        parser.add_argument(
            "--stride-geoms",
            action="store_true",
            help="Align UPDATE overlap (stride<VL) and hole (stride>VL)",
        )
        parser.add_argument(
            "--geom",
            choices=("overlap", "hole", "all"),
            default="all",
            help="Which Align stride geom(s) to run with --stride-geoms",
        )
        parser.add_argument("--sentinel", type=float, default=None)
        args = parser.parse_args()
        if args.shape is not None:
            args.shape = tuple(args.shape)
        if args.stride_geoms:
            return _run_stride_geoms(args)
        return _run_dynamic_extent(args)
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
