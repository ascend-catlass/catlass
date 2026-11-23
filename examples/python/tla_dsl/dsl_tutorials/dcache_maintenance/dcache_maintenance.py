# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify scalar DCCI on the issuing core.

Each kernel writes one GM word, then cleans and invalidates its cache line
or the selected cache domain. The host reads the written value after the
kernel returns.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import catlass.tla as tla
from catlass.params import DcciTarget

if TYPE_CHECKING:
    import torch

INITIAL = 11
DIRTY = 101

# One word per buffer, each in its own allocation, so two cases never share a
# cache line.
LENGTH = 1


# --- kernels ---------------------------------------------------------------


@tla.kernel
def addressed_dcci_vector(probe: tla.Tensor) -> None:
    """One line, published from the AIV scalar unit."""
    with tla.vector():
        probe[0] = tla.Int32(DIRTY)
        tla.arch.dcci_cacheline(probe.ptr)


@tla.kernel
def addressed_dcci_cube(probe: tla.Tensor) -> None:
    """The same operation on the AIC scalar unit: the helper must link there too."""
    with tla.cube():
        probe[0] = tla.Int32(DIRTY)
        tla.arch.dcci_cacheline(probe.ptr)


@tla.kernel
def addressed_dcci_offset_pointer(probe: tla.Tensor) -> None:
    """The address of the element written, not the tensor's base.

    ``probe.ptr`` is the allocation base; the line that holds element 2 is
    selected by the pointer that addresses element 2. Both are the same line for
    a buffer this small, which is the point of the case: the offset pointer is
    accepted and addresses the data, not a different object.
    """
    with tla.vector():
        probe[2] = tla.Int32(DIRTY)
        tla.arch.dcci_cacheline(probe.ptr + 2)


def _addressed_at(index: int):
    """Publish the word at `index` by the address of that word.

    The spread of indices these are built with is the point. A cache line is
    some number of bytes wide and this example does not encode which, so the
    cases walk element offsets whose byte addresses straddle every plausible
    line width: 4, 60, 64, 124 and 128 bytes into the buffer. Whatever the line
    size, some of those sit at a line's first word, some at its last, and some
    in the middle -- and the requirement is the same for all of them, which is
    why none of them has to know which it is.
    """

    @tla.kernel
    def kernel(probe: tla.Tensor) -> None:
        with tla.vector():
            probe[index] = tla.Int32(DIRTY)
            tla.arch.dcci_cacheline(probe.ptr + index)

    return kernel


@tla.kernel
def addressed_dcci_same_line_neighbour(probe: tla.Tensor) -> None:
    """Two adjacent words, one address: the line carries both.

    Elements 0 and 1 are four bytes apart, so they are in the same line on any
    line size the hardware could have. Only element 0's address is maintained,
    and the case reads element 1 back -- a word this kernel never named in a
    dcci. It must be published anyway, because publication happens per line and
    not per address, which is the one thing about line granularity this example
    can establish rather than assume.
    """
    with tla.vector():
        probe[0] = tla.Int32(DIRTY)
        probe[1] = tla.Int32(DIRTY)
        tla.arch.dcci_cacheline(probe.ptr)


def _whole_domain_kernel(target: DcciTarget):
    @tla.kernel
    def kernel(probe: tla.Tensor) -> None:
        with tla.vector():
            probe[0] = tla.Int32(DIRTY)
            tla.arch.dcci_cachelines(target)

    return kernel


@tla.kernel
def whole_domain_default(probe: tla.Tensor) -> None:
    """No argument is ``DcciTarget.ALL``: the issuing core's whole dcache."""
    with tla.vector():
        probe[0] = tla.Int32(DIRTY)
        tla.arch.dcci_cachelines()


# --- harness ---------------------------------------------------------------

# (name, kernel, index written)
CASES = (
    ("addressed_dcci_vector", addressed_dcci_vector, 0),
    ("addressed_dcci_cube", addressed_dcci_cube, 0),
    ("addressed_dcci_offset_pointer", addressed_dcci_offset_pointer, 2),
    ("whole_domain_all", _whole_domain_kernel(DcciTarget.ALL), 0),
    ("whole_domain_default", whole_domain_default, 0),
    ("whole_domain_gm_l2", _whole_domain_kernel(DcciTarget.GM_L2), 0),
    # ST_ATOMIC owns the lines used by scalar atomic stores, which this plain
    # store is not one of. The case checks it executes and leaves the store's
    # own publication path alone; it is not evidence about the atomic domain.
    (
        "whole_domain_st_atomic",
        _whole_domain_kernel(DcciTarget.ST_ATOMIC),
        0,
    ),
    # Aligned and unaligned byte offsets spanning any plausible line width.
    ("addressed_dcci_at_byte_4", _addressed_at(1), 1),
    ("addressed_dcci_at_byte_60", _addressed_at(15), 15),
    ("addressed_dcci_at_byte_64", _addressed_at(16), 16),
    ("addressed_dcci_at_byte_124", _addressed_at(31), 31),
    ("addressed_dcci_at_byte_128", _addressed_at(32), 32),
    # The neighbour is read, not the maintained address.
    (
        "addressed_dcci_same_line_neighbour",
        addressed_dcci_same_line_neighbour,
        1,
    ),
)


def _require_torch_npu(device: int):
    import torch

    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("torch_npu is required for this example") from exc
    torch_npu.npu.set_device(device)
    return torch


def _gm(tensor_1d) -> tla.Tensor:
    return tla.from_dlpack(tensor_1d.contiguous(), layout_tag=tla.arch.RowMajor)


def _run_case(
    torch,
    device: str,
    name: str,
    kernel,
    index: int,
) -> int:
    width = max(LENGTH, index + 1)
    probe = torch.full((width,), INITIAL, dtype=torch.int32, device=device)
    probe_t = _gm(probe)

    tla.compile(kernel, probe_t)(probe_t, block_num=1)
    torch.npu.synchronize()

    actual = int(probe[index].item())
    if actual != DIRTY:
        print(
            f"{name}_failed expected={DIRTY} actual={actual} "
            f"index={index} initial={INITIAL} dirty={DIRTY}"
        )
        return 1
    print(f"{name}_ok=True published value={actual}")
    return 0


def run(args: argparse.Namespace) -> int:
    torch = _require_torch_npu(args.device)
    device = f"npu:{args.device}"

    failures = 0
    for name, kernel, index in CASES:
        failures += _run_case(torch, device, name, kernel, index)

    if failures:
        print(f"verification_failed cases={failures}")
        return 1
    print("verification_ok=True")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scalar dcache maintenance: dcci_cacheline / dcci_cachelines."
    )
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
