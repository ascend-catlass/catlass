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
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import catlass.tla as tla

DEFAULT_SHAPES = (31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 500, 1024, 2000)


@dataclass(frozen=True)
class DTypeSpec:
    name: str
    tla_dtype: Any
    torch_dtype_name: str
    default_sentinel: float | int
    element_bytes: int
    register_elements: int | None = None


REG_BYTES = 256

DTYPES: dict[str, DTypeSpec] = {
    "f32": DTypeSpec("f32", tla.Float32, "float32", -7.0, 4),
    "f16": DTypeSpec("f16", tla.Float16, "float16", -7.0, 2),
    "bf16": DTypeSpec("bf16", tla.BFloat16, "bfloat16", -7.0, 2),
    "i8": DTypeSpec("i8", tla.Int8, "int8", -101, 1),
    "i16": DTypeSpec("i16", tla.Int16, "int16", -7, 2),
    "i32": DTypeSpec("i32", tla.Int32, "int32", -7, 4),
    "i64": DTypeSpec("i64", tla.Int64, "int64", -7, 8),
    "u8": DTypeSpec("u8", tla.UInt8, "uint8", 0, 1),
    "u16": DTypeSpec("u16", tla.UInt16, "uint16", 0, 2),
    "u32": DTypeSpec("u32", tla.UInt32, "uint32", 0, 4),
    "u64": DTypeSpec("u64", tla.UInt64, "uint64", 0, 8),
    # Host storage is uint8: torch cannot DLPack-export float8; from_dlpack
    # takes element_type= to retype the byte buffer (see make_tla_tensor).
    "f8e4m3fn": DTypeSpec("f8e4m3fn", tla.Float8E4M3FN, "uint8", 0, 1),
    "f8e5m2": DTypeSpec("f8e5m2", tla.Float8E5M2, "uint8", 0, 1),
    "f8e8m0": DTypeSpec("f8e8m0", tla.Float8E8M0, "uint8", 0, 1),
    "f4e2m1x2": DTypeSpec("f4e2m1x2", tla.Float4E2M1x2, "uint8", 0, 1),
    "f4e1m2x2": DTypeSpec("f4e1m2x2", tla.Float4E1M2x2, "uint8", 0, 1),
    "hif8": DTypeSpec("hif8", tla.HiFloat8, "uint8", 0, 1),
    "bool": DTypeSpec("bool", tla.Bool, "bool", 0, 1),
}

# Element types that must be bound via from_dlpack(..., element_type=...).
_DLPACK_ELEMENT_TYPE_OVERRIDE: frozenset[Any] = frozenset(
    {
        tla.Float8E4M3FN,
        tla.Float8E5M2,
        tla.Float8E8M0,
        tla.Float4E2M1x2,
        tla.Float4E1M2x2,
        tla.HiFloat8,
    }
)


@dataclass(frozen=True)
class VectorKernelConfig:
    vector_elements: int
    lanes: int
    loops: int
    tla_dtype: Any
    torch_dtype: Any
    default_sentinel: float | int
    element_bytes: int


@dataclass(frozen=True)
class DirectVectorOpConfig:
    description: str
    kernel: Any
    all_dtypes: tuple[str, ...]
    operator_specs: Callable[[], dict[str, dict[str, Any]]]
    set_kernel_config: Callable[
        [str, str, tuple[int, ...] | None], tuple[type[Any], Any, float | int]
    ]
    get_vector_elements: Callable[[], int]
    get_kernel_shape: Callable[[], tuple[int, ...]]
    make_inputs: Callable[[argparse.Namespace, str, Any], tuple[Any, ...]]
    expected: Callable[[str, tuple[Any, ...]], Any]
    unsupported_case: Callable[[str, str], bool]
    print_skip: Callable[[str, str, tuple[int, ...]], None]
    script_path: Path
    float_dtypes: frozenset[str]
    input_count: int = 0
    output_count: int = 1
    launch_blocks: int = 1
    batch_kernel: Any | None = None
    configure_batch: (
        Callable[
            [tuple[str, ...], str, tuple[int, ...]],
            tuple[type[Any], Any, float | int],
        ]
        | None
    ) = None


def shape_num_elements(shape: tuple[int, ...]) -> int:
    elements = 1
    for dim in shape:
        elements *= dim
    return elements


def shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def dtype_config(dtype_name: str, all_dtypes: tuple[str, ...]) -> DTypeSpec:
    if dtype_name not in all_dtypes or dtype_name not in DTYPES:
        raise SystemExit(
            f"unsupported dtype={dtype_name!r}; expected one of: {', '.join(all_dtypes)}"
        )
    return DTYPES[dtype_name]


def torch_dtype(spec: DTypeSpec, torch: Any | None = None) -> Any:
    if torch is None:
        try:
            import torch as torch_mod
        except ImportError:
            return None
        torch = torch_mod
    return getattr(torch, spec.torch_dtype_name)


def vector_kernel_config(
    dtype_name: str, shape: tuple[int, ...] | None, all_dtypes: tuple[str, ...]
) -> VectorKernelConfig:
    spec = dtype_config(dtype_name, all_dtypes)
    vector_elements = shape_num_elements(shape) if shape is not None else 400
    lanes = spec.register_elements or (REG_BYTES // spec.element_bytes)
    return VectorKernelConfig(
        vector_elements=vector_elements,
        lanes=lanes,
        loops=(vector_elements + lanes - 1) // lanes,
        tla_dtype=spec.tla_dtype,
        torch_dtype=torch_dtype(spec),
        default_sentinel=spec.default_sentinel,
        element_bytes=spec.element_bytes,
    )


def parse_shape(value: str, *, script_name: str) -> tuple[int, ...]:
    text = value.strip().lower().replace(",", "x")
    if not text:
        raise argparse.ArgumentTypeError("shape must not be empty")
    try:
        dims = tuple(int(part) for part in text.split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; expected N or AxB"
        ) from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(
            f"shape dimensions must be positive: {value!r}"
        )
    if len(dims) != 1:
        raise argparse.ArgumentTypeError(
            f"{script_name} currently supports flat 1D vector shapes only"
        )
    return dims


def _parse_batch_size(value: str) -> int:
    try:
        size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "batch size must be an integer from 1 to 4"
        ) from exc
    if not 1 <= size <= 4:
        raise argparse.ArgumentTypeError("batch size must be an integer from 1 to 4")
    return size


def operation_batches(
    operations: tuple[str, ...], batch_size: int
) -> tuple[tuple[str, ...], ...]:
    if not 1 <= batch_size <= 4:
        raise ValueError("batch_size must be between 1 and 4")
    return tuple(
        operations[start : start + batch_size]
        for start in range(0, len(operations), batch_size)
    )


def runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {"options": "--npu-arch 3510"}


def make_tla_tensor(
    dev_buf: Any, tla_dtype: type[Any], kernel_shape: tuple[int, ...]
) -> Any:
    from catlass.tla.runtime import from_dlpack

    kwargs: dict[str, Any] = {
        "layout_tag": tla.arch.RowMajor,
        "origin_shape": kernel_shape,
    }
    if tla_dtype in _DLPACK_ELEMENT_TYPE_OVERRIDE:
        kwargs["element_type"] = tla_dtype
    return from_dlpack(dev_buf.contiguous(), **kwargs)


def require_torch_npu(device_id: int, script_name: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"{script_name} requires PyTorch.") from exc
    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit(f"{script_name} requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


class DirectVectorOpHarness:
    def __init__(self, config: DirectVectorOpConfig) -> None:
        self.config = config
        self.script_name = config.script_path.stem

    def _runtime_kwargs(self, args: argparse.Namespace) -> dict[str, Any]:
        return runtime_kwargs(args)

    def _case_args(
        self, args: argparse.Namespace, dtype_name: str, shape: tuple[int, ...]
    ) -> argparse.Namespace:
        case_args = argparse.Namespace(**vars(args))
        case_args.dtype = dtype_name
        case_args.shape = shape
        return case_args

    def _run_single_case(
        self, args: argparse.Namespace, dtype_name: str, torch: Any
    ) -> int:
        if self.config.unsupported_case(args.op, dtype_name):
            self.config.print_skip(args.op, dtype_name, args.shape)
            return 0
        tla_dtype, torch_dt, default_sentinel = self.config.set_kernel_config(
            args.op, dtype_name, args.shape
        )
        inputs = self.config.make_inputs(args, dtype_name, torch)
        device = "npu"
        sentinel = args.sentinel if args.sentinel is not None else default_sentinel
        # Build on CPU then .to(npu): NPU fill rejects uint fill_value promotion.
        outputs = tuple(
            torch.full(
                (self.config.get_vector_elements(),),
                sentinel,
                dtype=torch_dt,
            ).to(device)
            for _ in range(self.config.output_count)
        )
        expected = self.config.expected(args.op, inputs)
        expected_outputs = expected if isinstance(expected, tuple) else (expected,)
        if len(expected_outputs) != self.config.output_count:
            raise SystemExit(
                f"expected() returned {len(expected_outputs)} outputs, "
                f"but output_count={self.config.output_count}"
            )

        # Bind each input with its true buffer extent: kernels whose source is
        # larger than the kernel shape (e.g. stride gather) rely on the origin
        # shape to size the GM->UB copy; tile_view crops views to this extent.
        tla_inputs = tuple(
            make_tla_tensor(input_tensor, tla_dtype, tuple(input_tensor.shape))
            for input_tensor in inputs
        )
        tla_outputs = tuple(
            make_tla_tensor(output, tla_dtype, self.config.get_kernel_shape())
            for output in outputs
        )

        artifact = tla.compile(
            self.config.kernel,
            *tla_inputs,
            *tla_outputs,
            **self._runtime_kwargs(args),
        )
        artifact(*tla_inputs, *tla_outputs, block_num=args.block_num)

        torch.npu.synchronize()
        first_mismatch: dict[str, Any] | None = None
        output_matches = []
        for output_index, (actual, expected_output) in enumerate(
            zip(outputs, expected_outputs, strict=True)
        ):
            # Compare on CPU: NPU eq lacks u16/u32 (and some other integer dtypes).
            act_c, exp_c = actual.cpu(), expected_output.cpu()
            if dtype_name in self.config.float_dtypes:
                expected_match = torch.isclose(act_c, exp_c, rtol=0.0, atol=args.atol)
            else:
                expected_match = act_c.eq(exp_c)
            output_matches.append(bool(expected_match.all()))
            mismatch = expected_match.logical_not().nonzero(as_tuple=False)
            bad = [int(v) for v in mismatch.flatten().tolist()]
            if bad:
                print(
                    f"[ALL-MISMATCH] output={output_index} count={len(bad)} indices={bad}"
                )
            if first_mismatch is None and mismatch.numel():
                index = int(mismatch[0].item())
                first_mismatch = {
                    "output": output_index,
                    "index": index,
                    "actual": act_c[index].item(),
                    "expected": exp_c[index].item(),
                }

        print(
            f"compile_ok=True host=torch_npu op={args.op} dtype={dtype_name} "
            f"shape={shape_label(args.shape)} layout=row"
        )
        print(f"kernel.o path={artifact.kernel_binary_path}")
        print("launch_ok=True")
        print(f"outputs equal expected {args.op}? {all(output_matches)}")
        print(f"first mismatch={first_mismatch}")
        return 0 if first_mismatch is None else 1

    def _verify_batch_case(
        self,
        op_name: str,
        dtype_name: str,
        actual_outputs: tuple[Any, ...],
        expected_outputs: tuple[Any, ...],
    ) -> bool:
        import torch

        atol = self.config.operator_specs()[op_name]["default_atol"]
        first_mismatch: dict[str, Any] | None = None
        output_matches = []
        for output_index, (actual, expected) in enumerate(
            zip(actual_outputs, expected_outputs, strict=True)
        ):
            # Compare on CPU: NPU eq lacks u16/u32 (and some other integer dtypes).
            act_c, exp_c = actual.cpu(), expected.cpu()
            if dtype_name in self.config.float_dtypes:
                matches = torch.isclose(act_c, exp_c, rtol=0.0, atol=atol)
            else:
                matches = act_c.eq(exp_c)
            output_matches.append(bool(matches.all()))
            mismatch = matches.logical_not().nonzero(as_tuple=False)
            if first_mismatch is None and mismatch.numel():
                index = int(mismatch[0].item())
                first_mismatch = {
                    "output": output_index,
                    "index": index,
                    "actual": act_c[index].item(),
                    "expected": exp_c[index].item(),
                }
        print(
            f"batch_case op={op_name} dtype={dtype_name} "
            f"passed={all(output_matches)} first_mismatch={first_mismatch}"
        )
        return first_mismatch is None

    def _run_batch(
        self,
        args: argparse.Namespace,
        ops: tuple[str, ...],
        dtype_name: str,
        torch: Any,
    ) -> int:
        batch_kernel = self.config.batch_kernel
        configure_batch = self.config.configure_batch
        if batch_kernel is None or configure_batch is None:
            raise SystemExit(f"{self.script_name} does not support operation batching")
        shape = args.shape
        case_inputs: list[tuple[Any, ...]] = []
        case_expected: list[tuple[Any, ...]] = []
        sentinels: list[float | int] = []

        for op_name in ops:
            case_args = argparse.Namespace(**vars(args))
            case_args.op = op_name
            _, _, sentinel = self.config.set_kernel_config(op_name, dtype_name, shape)
            inputs = self.config.make_inputs(case_args, dtype_name, torch)
            expected = self.config.expected(op_name, inputs)
            expected_outputs = expected if isinstance(expected, tuple) else (expected,)
            if len(inputs) != self.config.input_count:
                raise SystemExit(
                    f"{self.script_name}: op={op_name} produced {len(inputs)} inputs; "
                    f"batch ABI requires {self.config.input_count}"
                )
            if len(expected_outputs) != self.config.output_count:
                raise SystemExit(
                    f"{self.script_name}: op={op_name} produced "
                    f"{len(expected_outputs)} outputs; "
                    f"batch ABI requires {self.config.output_count}"
                )
            case_inputs.append(inputs)
            case_expected.append(expected_outputs)
            sentinels.append(sentinel)

        tla_dtype, batch_torch_dtype, _ = configure_batch(ops, dtype_name, shape)
        packed_shape = (len(ops) * shape_num_elements(shape),)
        packed_inputs = tuple(
            torch.cat([inputs[index] for inputs in case_inputs])
            for index in range(self.config.input_count)
        )
        # Build on CPU then .to(npu): NPU fill rejects uint fill_value promotion.
        packed_outputs = tuple(
            torch.cat(
                [
                    torch.full(
                        (shape_num_elements(shape),),
                        sentinels[case_index],
                        dtype=batch_torch_dtype,
                    )
                    for case_index in range(len(ops))
                ]
            ).to("npu")
            for _ in range(self.config.output_count)
        )
        tla_inputs = tuple(
            make_tla_tensor(value, tla_dtype, packed_shape) for value in packed_inputs
        )
        tla_outputs = tuple(
            make_tla_tensor(value, tla_dtype, packed_shape) for value in packed_outputs
        )
        artifact = tla.compile(
            batch_kernel,
            *tla_inputs,
            *tla_outputs,
            **self._runtime_kwargs(args),
        )
        artifact(*tla_inputs, *tla_outputs, block_num=len(ops))
        torch.npu.synchronize()

        extent = shape_num_elements(shape)
        failed = 0
        for case_index, (op_name, expected_outputs) in enumerate(
            zip(ops, case_expected, strict=True)
        ):
            start = case_index * extent
            end = start + extent
            actual_outputs = tuple(output[start:end] for output in packed_outputs)
            if not self._verify_batch_case(
                op_name, dtype_name, actual_outputs, expected_outputs
            ):
                failed += 1
        print(
            f"batch_kernel_ok={failed == 0} dtype={dtype_name} "
            f"ops={','.join(ops)} blocks={len(ops)}"
        )
        print(f"kernel.o path={artifact.kernel_binary_path}")
        return failed

    def batch_run(self, args: argparse.Namespace) -> int:
        if self.config.batch_kernel is None or self.config.configure_batch is None:
            raise SystemExit(f"{self.script_name} does not support operation batching")
        requested_ops = tuple(args.batch_run)
        dtypes = tuple(args.dtypes)
        skipped = 0
        batches: list[tuple[str, tuple[str, ...]]] = []
        for dtype_name in dtypes:
            supported = []
            for op_name in requested_ops:
                if self.config.unsupported_case(op_name, dtype_name):
                    skipped += 1
                    self.config.print_skip(op_name, dtype_name, args.shape)
                else:
                    supported.append(op_name)
            batches.extend(
                (dtype_name, ops)
                for ops in operation_batches(tuple(supported), args.batch_size)
            )
        if not batches:
            print(f"batch summary: passed=0 failed=0 skipped={skipped} batches=0")
            return 0
        launch_args = argparse.Namespace(**vars(args))
        torch = require_torch_npu(args.device, self.script_name)
        torch.npu.set_device(args.device)
        failed = 0
        passed = 0
        batch_count = 0
        for dtype_name, ops in batches:
            batch_count += 1
            batch_failed = self._run_batch(launch_args, ops, dtype_name, torch)
            failed += batch_failed
            passed += len(ops) - batch_failed
            if batch_failed and args.fail_fast:
                break
        print(
            f"batch summary: passed={passed} failed={failed} skipped={skipped} "
            f"batches={batch_count} total={passed + failed + skipped}"
        )
        return 0 if failed == 0 else 1

    def sweep(self, args: argparse.Namespace) -> int:
        torch = require_torch_npu(args.device, self.script_name)
        torch.npu.set_device(args.device)
        total = 0
        passed = 0
        failed = 0
        skipped = 0
        start = time.perf_counter()
        for dtype_name in args.dtypes:
            for shape in args.shapes:
                total += 1
                case_args = self._case_args(args, dtype_name, shape)
                if self.config.unsupported_case(args.op, dtype_name):
                    skipped += 1
                    print(
                        f"===== SKIP {args.op} dtype={dtype_name} "
                        f"shape={shape_label(shape)} ====="
                    )
                    self.config.print_skip(args.op, dtype_name, shape)
                    continue
                print(
                    f"===== START {args.op} dtype={dtype_name} "
                    f"shape={shape_label(shape)} ====="
                )
                rc = self._run_single_case(case_args, dtype_name, torch)
                if rc == 0:
                    passed += 1
                    print(
                        f"===== PASS {args.op} dtype={dtype_name} "
                        f"shape={shape_label(shape)} ====="
                    )
                else:
                    failed += 1
                    print(
                        f"===== FAIL {args.op} dtype={dtype_name} "
                        f"shape={shape_label(shape)} rc={rc} ====="
                    )
                    if args.fail_fast:
                        break
            if failed and args.fail_fast:
                break
        print(
            f"{args.op} summary: passed={passed} failed={failed} "
            f"skipped={skipped} total={total}"
        )
        print(f"timing.sweep_total_seconds={time.perf_counter() - start:.6f}")
        return 0 if failed == 0 else 1

    def run(self, args: argparse.Namespace) -> int:
        torch = require_torch_npu(args.device, self.script_name)
        torch.npu.set_device(args.device)
        failed = 0
        dtypes = self.config.all_dtypes if args.all_dtypes else (args.dtype,)
        for dtype_name in dtypes:
            print("---", f"op={args.op}", f"dtype={dtype_name}", "---")
            failed += self._run_single_case(args, dtype_name, torch)
        return 0 if failed == 0 else 1

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.config.description)
        parser.add_argument(
            "op",
            choices=tuple(sorted(self.config.operator_specs())),
            nargs="?",
        )
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--sweep", action="store_true")
        mode.add_argument(
            "--batch-run",
            choices=tuple(sorted(self.config.operator_specs())),
            nargs="+",
        )
        parser.add_argument("--device", type=int, default=2)
        parser.add_argument("--block-num", type=int, default=None)
        default_dtype = (
            "f32" if "f32" in self.config.all_dtypes else self.config.all_dtypes[0]
        )
        parser.add_argument(
            "--dtype", choices=self.config.all_dtypes, default=default_dtype
        )
        parser.add_argument(
            "--shape",
            type=lambda value: parse_shape(value, script_name=self.script_name),
            default=(self.config.get_vector_elements(),),
        )
        parser.add_argument(
            "--shapes",
            type=lambda value: parse_shape(value, script_name=self.script_name),
            nargs="+",
            help="Flat vector shapes for --sweep.",
        )
        parser.add_argument(
            "--sizes",
            type=int,
            nargs="+",
            help="Alias for --shapes with one-dimensional vector sizes.",
        )
        parser.add_argument(
            "--dtypes",
            choices=self.config.all_dtypes,
            nargs="+",
            default=self.config.all_dtypes,
            help="Operand dtypes for --sweep.",
        )
        parser.add_argument("--all-dtypes", action="store_true")
        parser.add_argument("--batch-size", type=_parse_batch_size, default=4)
        parser.add_argument("--sentinel", type=float, default=None)
        parser.add_argument("--atol", type=float, default=None)
        parser.add_argument("--fail-fast", action="store_true")
        return parser

    def main(self) -> int:
        args = self._build_parser().parse_args()
        if args.block_num is None:
            args.block_num = self.config.launch_blocks
        if args.batch_run is not None:
            if args.op is not None:
                raise SystemExit("positional op cannot be combined with --batch-run")
            return self.batch_run(args)
        if args.op is None:
            raise SystemExit("a positional op is required unless --batch-run is used")
        if args.atol is None:
            args.atol = self.config.operator_specs()[args.op]["default_atol"]
        if args.sizes is not None:
            args.shapes = tuple((size,) for size in args.sizes)
        elif args.shapes is None:
            args.shapes = tuple((size,) for size in DEFAULT_SHAPES)
        if args.sweep:
            return self.sweep(args)
        return self.run(args)
