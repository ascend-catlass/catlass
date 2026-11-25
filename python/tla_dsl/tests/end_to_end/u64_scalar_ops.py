# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar uint64 bit-query end-to-end sample.

  ``tla.arch.bcnt0``  number of ``0`` bits
  ``tla.arch.bcnt1``  number of ``1`` bits (population count)
  ``tla.arch.clz``    leading zero count; ``64`` when the value is ``0``
  ``tla.arch.sff0``   position of the first ``0`` bit from the LSB; ``-1`` if none
  ``tla.arch.sff1``   position of the first ``1`` bit from the LSB; ``-1`` if none

Each sample is queried on every input element plus one compile-time literal;
both kernels also run all five queries on a runtime ``UInt64`` kernel scalar
argument and append those five results to the output.

The single-op mode selects one query with ``--op`` and runs it in both a
``tla.vector()`` and a ``tla.cube()`` region; ``--batch-run`` does the same for
all queries at once -- one launch per scope, one block per query
(``bcnt0``/``bcnt1``/``clz``/``sff0``/``sff1``). Queries are scalar ops, so
every kernel places them directly in the core region and outside
``tla.vec.func``.

Run on device:

    python tests/end_to_end/u64_scalar_ops.py --device 1 --op bcnt1
    python tests/end_to_end/u64_scalar_ops.py --device 1 --batch-run
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

_MASK64 = (1 << 64) - 1

# Bit patterns covering every edge case: 0 (clz == 64, sff1 == -1), all ones
# (bcnt0 == 0, sff0 == -1), both single-bit ends, the high bit, and mixed
# alternating patterns.
_VALUES = (
    0,
    1,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0xFFFFFFFFFFFFFFFF,
    0x1000000000000100,
    0x00FF00FF00FF00FF
)

_COUNT = len(_VALUES)

# Compile-time literal with the high bit set (>= 2^63): the host value must
# wrap into a negative signless i64 constant before reaching the query.
_SCALE = 0x8000000000000001
_ROWS = _COUNT + 1

# Block index -> query for --batch-run, in the order the reference expects.
_OP_NAMES = ("bcnt0", "bcnt1", "clz", "sff0", "sff1")
_RUNTIME_ROWS = len(_OP_NAMES)

# Core regions a scalar op may be placed in (see the API's placement rules).
_SCOPES = ("vector", "cube")


def _emit_rows(op, src: tla.Tensor, out: tla.Tensor, base: int = 0) -> None:
    """Emit one query over every loaded sample plus the compile-time literal."""
    for i in range(_COUNT):
        out[base + i] = op(src[i])
    out[base + _COUNT] = op(tla.as_numeric(_SCALE).to(tla.UInt64))


def _emit_queries(value, out: tla.Tensor, base: int) -> None:
    """Emit all five queries on one value into five output slots."""
    out[base + 0] = tla.arch.bcnt0(value)
    out[base + 1] = tla.arch.bcnt1(value)
    out[base + 2] = tla.arch.clz(value)
    out[base + 3] = tla.arch.sff0(value)
    out[base + 4] = tla.arch.sff1(value)


def make_kernel(op, scope: str):
    """Build a kernel for one uint64 query in a vector or cube region.

    The query runs on the tensor samples and the compile-time literal, then all
    five queries run on the runtime ``UInt64`` scalar argument, appended to the
    output.
    """

    if scope == "cube":

        @tla.kernel
        def u64_scalar_ops(src: tla.Tensor, out: tla.Tensor, value: tla.UInt64) -> None:
            with tla.cube():
                _emit_rows(op, src, out)
                _emit_queries(value, out, _ROWS)
                tla.pipe_barrier(tla.pipes.ALL)

    elif scope == "vector":

        @tla.kernel
        def u64_scalar_ops(src: tla.Tensor, out: tla.Tensor, value: tla.UInt64) -> None:
            with tla.vector():
                _emit_rows(op, src, out)
                _emit_queries(value, out, _ROWS)
                tla.pipe_barrier(tla.pipes.ALL)

    return u64_scalar_ops


def make_batch_kernel(scope: str):
    """Build the all-query batch kernel for one core region.

    Block ``n`` runs query ``n`` and writes into the ``n``-th output row. Each
    scope is its own launch so the cube and vector paths are validated
    separately; putting both regions in one kernel would let one path's correct
    output mask a broken path and would turn the launch into a MIX kernel.
    """

    if scope == "cube":

        @tla.kernel
        def u64_scalar_ops_batch(
            src: tla.Tensor, out: tla.Tensor, value: tla.UInt64
        ) -> None:
            block = tla.arch.block_idx()
            with tla.cube():
                out_row = tla.tile_view(
                    out, tla.make_shape(_ROWS), tla.make_coord(block)
                )
                if block == 0:
                    _emit_rows(tla.arch.bcnt0, src, out_row)
                elif block == 1:
                    _emit_rows(tla.arch.bcnt1, src, out_row)
                elif block == 2:
                    _emit_rows(tla.arch.clz, src, out_row)
                elif block == 3:
                    _emit_rows(tla.arch.sff0, src, out_row)
                else:
                    _emit_rows(tla.arch.sff1, src, out_row)
                if block == 0:
                    _emit_queries(value, out, len(_OP_NAMES) * _ROWS)
                tla.pipe_barrier(tla.pipes.ALL)

    else:

        @tla.kernel
        def u64_scalar_ops_batch(
            src: tla.Tensor, out: tla.Tensor, value: tla.UInt64
        ) -> None:
            block = tla.arch.block_idx()
            with tla.vector():
                out_row = tla.tile_view(
                    out, tla.make_shape(_ROWS), tla.make_coord(block)
                )
                if block == 0:
                    _emit_rows(tla.arch.bcnt0, src, out_row)
                elif block == 1:
                    _emit_rows(tla.arch.bcnt1, src, out_row)
                elif block == 2:
                    _emit_rows(tla.arch.clz, src, out_row)
                elif block == 3:
                    _emit_rows(tla.arch.sff0, src, out_row)
                else:
                    _emit_rows(tla.arch.sff1, src, out_row)
                if block == 0:
                    _emit_queries(value, out, len(_OP_NAMES) * _ROWS)
                tla.pipe_barrier(tla.pipes.ALL)

    return u64_scalar_ops_batch


def _signed64(value: int) -> int:
    """Two's-complement view of an unsigned 64-bit value, for torch.int64."""
    value &= _MASK64
    return value if value < (1 << 63) else value - (1 << 64)


def _reference(value: int) -> tuple[int, int, int, int, int]:
    value &= _MASK64
    bcnt1 = bin(value).count("1")
    bcnt0 = 64 - bcnt1
    clz = 64 if value == 0 else 64 - value.bit_length()
    if value == _MASK64:
        sff0 = -1
    else:
        sff0 = next(bit for bit in range(64) if not (value >> bit) & 1)
    sff1 = -1 if value == 0 else (value & -value).bit_length() - 1
    return bcnt0, bcnt1, clz, sff0, sff1


def _reference_rows(op_name: str) -> list[int]:
    index = _OP_NAMES.index(op_name)
    return [_reference(value)[index] for value in (*_VALUES, _SCALE)]


def _op_functions() -> dict[str, object]:
    return {
        "bcnt0": tla.arch.bcnt0,
        "bcnt1": tla.arch.bcnt1,
        "clz": tla.arch.clz,
        "sff0": tla.arch.sff0,
        "sff1": tla.arch.sff1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--op",
        choices=_OP_NAMES,
        default="bcnt0",
        help="query for single-op mode (ignored with --batch-run)",
    )
    parser.add_argument(
        "--batch-run",
        action="store_true",
        help="run all queries in one launch, one block per op-type",
    )
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    bits = torch.tensor(
        [_signed64(value) for value in _VALUES], dtype=torch.int64, device="npu"
    )
    src = from_dlpack(bits.view(torch.uint64), layout_tag=tla.arch.RowMajor)

    value = tla.UInt64(_SCALE)
    runtime_expected = torch.tensor(_reference(_SCALE), dtype=torch.int64)

    if args.batch_run:
        blocks = len(_OP_NAMES)
        failed = 0
        for scope in _SCOPES:
            out = torch.full(
                (blocks * _ROWS + _RUNTIME_ROWS,),
                -1,
                dtype=torch.int64,
                device="npu",
            )
            out_tensor = from_dlpack(out, layout_tag=tla.arch.RowMajor)
            artifact = tla.compile(
                make_batch_kernel(scope),
                src,
                out_tensor,
                value,
                options="--npu-arch 3510",
            )
            artifact(src, out_tensor, value, block_num=blocks)
            torch.npu.synchronize()

            flat = out.cpu()
            actual = flat[: blocks * _ROWS].reshape(blocks, _ROWS)
            for index, op_name in enumerate(_OP_NAMES):
                expected = torch.tensor(_reference_rows(op_name), dtype=torch.int64)
                ok = bool(torch.equal(actual[index], expected))
                failed += not ok
                print(f"batch_case scope={scope} op={op_name} passed={ok}")
            runtime_ok = bool(torch.equal(flat[blocks * _ROWS :], runtime_expected))
            failed += not runtime_ok
            print(f"batch_case scope={scope} runtime passed={runtime_ok}")
            print(f"batch scope={scope} kernel.o path={artifact.kernel_binary_path}")
        print(
            f"batch_kernel_ok={failed == 0} ops={','.join(_OP_NAMES)} "
            f"scopes={','.join(_SCOPES)}"
        )
        return 0 if failed == 0 else 1

    op_func = _op_functions()[args.op]
    expected = torch.tensor(
        _reference_rows(args.op) + list(_reference(_SCALE)), dtype=torch.int64
    )
    failed = 0
    for scope in _SCOPES:
        out = torch.full(
            (_ROWS + _RUNTIME_ROWS,), -1, dtype=torch.int64, device="npu"
        )
        out_tensor = from_dlpack(out, layout_tag=tla.arch.RowMajor)
        kernel = make_kernel(op_func, scope)
        artifact = tla.compile(
            kernel, src, out_tensor, value, options="--npu-arch 3510"
        )
        artifact(src, out_tensor, value, block_num=1)
        torch.npu.synchronize()

        ok = bool(torch.equal(out.cpu(), expected))
        failed += not ok
        print(f"op={args.op} scope={scope} passed={ok}")
        print(f"kernel.o path={artifact.kernel_binary_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
