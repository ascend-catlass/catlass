# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Element types the vector staging examples run over, and what each implies.

The staging examples (UB->L1 in its three source layouts) differ only in the
element type, but three things must follow it rather than be hard-coded:

* ``vf_len`` -- a vector register is 256 bytes, and ``vsstb`` always moves all
  8 of its DataBlocks, so a block-store chunk is ``256 / sizeof(T)`` elements.
  Getting this wrong writes past the intended blocks and corrupts data rather
  than failing, which is why it is derived here once.
* ``acc`` -- the MMAD accumulator: f32 for the float operands, i32 for i8.
* how the result is checked -- the integer route is exact and is compared bit
  for bit; a tolerance there would hide exactly the placement errors these
  examples exist to catch.

``needs_element_type`` marks the types torch cannot export over DLPack, which
reach the device as a same-width integer view plus an explicit element type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ElemSpec:
    """One element type and everything the staging examples derive from it."""

    name: str
    size_bytes: int
    is_int: bool
    needs_element_type: bool

    @property
    def vf_len(self) -> int:
        """Block-store chunk, in elements: one full 256-byte vector register."""
        return 256 // self.size_bytes

    def tla(self) -> Any:
        import catlass.tla as tla

        return {
            "f32": tla.Float32,
            "f16": tla.Float16,
            "bf16": tla.BFloat16,
            "i8": tla.Int8,
            "f8e4m3fn": tla.Float8E4M3FN,
            "f8e5m2": tla.Float8E5M2,
        }[self.name]

    def tla_acc(self) -> Any:
        import catlass.tla as tla

        return tla.Int32 if self.is_int else tla.Float32

    def torch(self) -> Any:
        import torch

        return {
            "f32": torch.float32,
            "f16": torch.float16,
            "bf16": torch.bfloat16,
            "i8": torch.int8,
            "f8e4m3fn": torch.float8_e4m3fn,
            "f8e5m2": torch.float8_e5m2,
        }[self.name]

    def torch_acc(self) -> Any:
        import torch

        return torch.int32 if self.is_int else torch.float32

    def operands(self, m: int, n: int, k: int):
        """Host-side A, B and the reference product, in this element type."""
        import torch

        if self.is_int:
            # Small magnitudes keep a K-long i8 dot product inside i32, so the
            # reference stays exact.
            a = torch.randint(-8, 8, (m, k), dtype=torch.int8)
            b = torch.randint(-8, 8, (k, n), dtype=torch.int8)
            return a, b, a.to(torch.int32) @ b.to(torch.int32)
        a = (torch.randn(m, k, dtype=torch.float32) * 2.0).to(self.torch())
        b = (torch.randn(k, n, dtype=torch.float32) * 2.0).to(self.torch())
        return a, b, a.to(torch.float32) @ b.to(torch.float32)

    def sentinel(self, value: float) -> Any:
        """The --sentinel fill, narrowed to what this accumulator can hold.

        The flag is a float so a float route can be prefilled with a value it
        would never compute; an integer accumulator takes the truncation.
        """
        return int(value) if self.is_int else value

    def check(self, got: Any, ref: Any, k: int) -> bool:
        import torch

        from .golden import compare

        return bool(torch.equal(got, ref)) if self.is_int else compare(got, ref, k)


ELEM_SPECS = {
    "f32": ElemSpec("f32", 4, False, False),
    "f16": ElemSpec("f16", 2, False, False),
    "bf16": ElemSpec("bf16", 2, False, False),
    "i8": ElemSpec("i8", 1, True, False),
    "f8e4m3fn": ElemSpec("f8e4m3fn", 1, False, True),
    "f8e5m2": ElemSpec("f8e5m2", 1, False, True),
}

ELEM_CHOICES = tuple(ELEM_SPECS)


def elem_spec(name: str) -> ElemSpec:
    return ELEM_SPECS[name]


def _stage(buf: Any, layout: str) -> Any:
    """Lay a host operand out as the tag says, then move it to the device.

    A column-major operand is materialised transposed rather than merely
    re-tagged: the tag describes the storage the kernel will read, so a
    non-contiguous view would hand it the wrong strides.
    """
    storage = buf.contiguous() if layout == "row" else buf.permute(1, 0).contiguous()
    return storage.npu()


def make_operand_tensors(
    spec: ElemSpec,
    a: Any,
    b: Any,
    out: Any,
    layout_a: str = "row",
    layout_b: str = "row",
):
    """Wrap the host operands as TLA tensors, overriding the element type where
    torch cannot carry it over DLPack.

    ``out`` is already on the device and is always row-major; only the two
    inputs carry a layout tag.
    """
    from .utils import create_tla_tensor

    kw = {"element_type": spec.tla()} if spec.needs_element_type else {}
    return (
        create_tla_tensor(_stage(a, layout_a), layout_a, **kw),
        create_tla_tensor(_stage(b, layout_b), layout_b, **kw),
        create_tla_tensor(out, "row"),
    )
