# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch  # noqa: F401
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack
from catlass.types import dtype_size_bytes

#: Byte alignment every Unified Buffer ``tla.allocate`` in the epilogue
#: examples asks for.
UB_ALLOC_ALIGN_BYTES = 256

# ---------------------------------------------------------------------------
# Helper functions below
# ---------------------------------------------------------------------------


def compute_ub_slot_elems(
    dtype_c: str,
    *,
    nodes: int,
    stages: int,
    budget_bytes: int | None = None,
) -> int:
    """Size one UB slot so ``nodes * stages`` of them fit the buffer.

    ``budget_bytes`` defaults to the whole Unified Buffer; pass a smaller
    figure when the kernel reserves part of it for something else.

    The slot is rounded DOWN to ``UB_ALLOC_ALIGN_BYTES``. Both directions of
    getting this wrong end the same way -- the kernel reserves more UB than the
    buffer holds and is refused at launch:

    * rounding UP pushes the slots past the budget they were just divided out
      of, so the parts come to more than the whole;
    * rounding to anything finer than the allocation's own alignment leaves
      each slot to be padded up to it, and that padding is invisible here.
    """
    elem_bytes = dtype_size_bytes(dtype_c)
    if budget_bytes is None:
        budget_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)
    align_elems = UB_ALLOC_ALIGN_BYTES // elem_bytes
    slot_elems = budget_bytes // nodes // stages // elem_bytes
    return slot_elems // align_elems * align_elems


def get_block_num(block_num: int, device: int = 0, *, kind: str = "vector") -> int:
    """Get launch ``block_num``.

    Non-``-1`` uses the host argument. ``-1`` means full-device launch:
    pure vector → ``vector_core_num`` (AIV); cube/mix → ``cube_core_num`` (AIC).
    """
    if block_num != -1 and block_num <= 0:
        raise ValueError("block_num must be -1 (automatic) or positive")
    block_num = int(block_num)
    if block_num != -1:
        return block_num
    props = torch.npu.get_device_properties(int(device))
    if kind == "vector":
        return max(1, int(props.vector_core_num))
    if kind in {"cube", "mix"}:
        return max(1, int(props.cube_core_num))
    raise ValueError(f"Unsupported kernel kind for block_num default: {kind!r}")


# Signed integer view types by storage width, for handing a buffer over as plain
# storage when DLPack cannot describe its real format.
_STORAGE_VIEW_BY_WIDTH = {
    8: torch.int8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}

# Layout spellings accepted by create_tla_tensor. "col" is not a typo to be
# cleaned up: basic_syrk.py passes it literally and the splitk examples expose
# it as the --layout-a / --layout-b choice, so it has to keep working. VectorLayout is
# the 1-D contiguous tag the flat GM->L1 and L1->biasbuf copy routes are keyed
# on.
_LAYOUT_TAG_BY_NAME = {
    "row": tla.arch.RowMajor,
    "col": tla.arch.ColumnMajor,
    "column": tla.arch.ColumnMajor,
    "vector": tla.arch.VectorLayout,
    "zN": tla.arch.zN,
    "nZ": tla.arch.nZ,
}

#: The tags that describe an already-packed fractal buffer. ``create_tla_tensor``
#: binds them, it never produces them: the packing is the caller's explicit step.
_FRACTAL_LAYOUTS = frozenset({"zN", "nZ"})


def _round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the next multiple of ``multiple``."""
    return (value + multiple - 1) // multiple * multiple


def _fractal_numel(logical, layout: str, element_size: int) -> int:
    """Element count of the padded, fractal-ordered buffer for ``logical``.

    ``zN`` pads M up to 16 rows and N up to a 32-byte C0; ``nZ`` is the same
    packing on the transpose, so it pads M up to C0 and N up to 16.
    """
    rows, cols = logical
    elems_per_c0 = 32 // element_size
    if layout == "zN":
        return _round_up(rows, 16) * _round_up(cols, elems_per_c0)
    return _round_up(rows, elems_per_c0) * _round_up(cols, 16)


def create_tla_tensor(buf, layout: str, element_type=None, origin_shape=None):
    """Wrap a device buffer as a TLA tensor.

    ``element_type`` overrides the element type derived from the DLPack dtype.
    torch cannot export fp8 over DLPack at all, so an fp8 buffer is handed over
    as a same-width integer view with the real type supplied here.

    ``layout`` is ``"row"``, ``"col"`` / ``"column"``, ``"vector"``, ``"zN"``, or
    ``"nZ"``.
    ``"col"`` is the spelling the existing examples and their ``--layout-a`` /
    ``--layout-b`` choices already use, so both spellings are accepted.
    ``"vector"`` is the one-dimensional contiguous tag (``tla.arch.VectorLayout``) the
    flat copy routes are keyed on; it is what a 1-D tile such as a bias row uses.
    ``"zN"`` / ``"nZ"`` do **not** pack anything: they describe a buffer that is
    already in that fractal order, so ``buf`` must be the flat, padded run
    ``host_nd2nz`` produces and ``origin_shape`` must say what logical tile it
    holds. The packing is a prepack step the caller spells out, because it moves
    data -- ``from_dlpack`` only ever derives metadata and would otherwise pair a
    fractal descriptor with a plain ND buffer and compute silent nonsense.
    """
    tag = _LAYOUT_TAG_BY_NAME.get(layout)
    if tag is None:
        raise ValueError(
            "create_tla_tensor layout must be one of "
            + ", ".join(sorted(_LAYOUT_TAG_BY_NAME))
            + f"; got {layout!r}"
        )
    if element_type is not None:
        # Derive the view from the override's own width rather than assuming one
        # byte: from_dlpack accepts any same-width override, and viewing through
        # the wrong width silently reshapes the buffer.
        width = element_type.width
        if width not in _STORAGE_VIEW_BY_WIDTH:
            raise ValueError(
                f"no storage view for a {width}-bit element type {element_type.__name__}"
            )
        buf = buf.view(_STORAGE_VIEW_BY_WIDTH[width])
    if layout in _FRACTAL_LAYOUTS:
        if origin_shape is None:
            raise ValueError(
                f"create_tla_tensor layout={layout!r} describes a prepacked fractal "
                "buffer, whose flat length no longer says what tile it holds; "
                "pass origin_shape=(rows, cols) alongside the packed buffer"
            )
        logical = tuple(int(extent) for extent in origin_shape)
        if len(logical) != 2:
            raise ValueError(
                f"create_tla_tensor origin_shape must be a flat 2-tuple "
                f"(rows, cols) for layout={layout!r}; got {logical!r}"
            )
        if buf.dim() != 1:
            raise ValueError(
                f"create_tla_tensor layout={layout!r} expects the flat prepacked "
                f"buffer host_nd2nz returns; got a {buf.dim()}-D buffer of shape "
                f"{tuple(buf.shape)}"
            )
        expected = _fractal_numel(logical, layout, buf.element_size())
        if int(buf.numel()) != expected:
            raise ValueError(
                f"create_tla_tensor layout={layout!r} for logical {logical} needs "
                f"{expected} packed elements but the buffer holds {int(buf.numel())}; "
                "pack it with host_nd2nz before binding"
            )
        return from_dlpack(
            buf.contiguous(),
            layout_tag=tag,
            origin_shape=logical,
            element_type=element_type,
        ).mark_layout_dynamic()
    return from_dlpack(
        buf.contiguous(), layout_tag=tag, element_type=element_type
    ).mark_layout_dynamic()


def to_hf32(
    x: torch.Tensor,
    hf32_mode: tla.params.HF32Mode,
) -> torch.Tensor:
    """Simulate HF32 rounding mode on f32 values.

    HF32 keeps the FP32 sign and 8-bit exponent, and reduces the mantissa to
    11 significant bits (10 explicit mantissa bits, close to FP16):

    - ``HF32_NEAREST_ZERO`` rounds to nearest, ties toward zero.
    - ``HF32_NEAREST_EVEN`` rounds them to nearest-even.
    """
    if not isinstance(hf32_mode, tla.params.HF32Mode):
        raise TypeError(
            f"hf32_mode must be a tla.params.HF32Mode, got {type(hf32_mode).__name__}"
        )
    x = x.float()
    if hf32_mode == tla.params.HF32Mode.HF32_DISABLE:
        return x

    bits = x.contiguous().view(torch.int32)
    if hf32_mode == tla.params.HF32Mode.HF32_NEAREST_ZERO:
        rounded_bits = (bits + 0x0FFF) & ~0x1FFF
    elif hf32_mode == tla.params.HF32Mode.HF32_NEAREST_EVEN:
        lsb = (bits >> 13) & 1
        rounded_bits = (bits + 0x0FFF + lsb) & ~0x1FFF
    else:
        raise ValueError(f"Unsupported HF32 mode: {hf32_mode!r}")

    return rounded_bits.view(torch.float32)


def _host_ND_2_zN_impl(x: torch.Tensor):
    row, col = x.shape
    C0_NUM_PER_FRACTAL = 16
    ELE_NUM_PER_BLK = 32 // x.element_size()
    row_aligned = _round_up(row, C0_NUM_PER_FRACTAL)
    col_aligned = _round_up(col, ELE_NUM_PER_BLK)
    # pad a 2-D tensor (left, right, top, bottom)
    x_pad = torch.nn.functional.pad(
        x, (0, col_aligned - col, 0, row_aligned - row), mode="constant", value=0
    )

    col_blocks = col_aligned // ELE_NUM_PER_BLK
    out = x_pad.view(row_aligned, col_blocks, ELE_NUM_PER_BLK)
    return out.permute(1, 0, 2).contiguous().reshape(-1)


def host_nd2nz(x: torch.Tensor, tag: str) -> torch.Tensor:
    """convert a host side torch tensor from RowMajor to zN/nZ, output tensor is a 1-D tensor without shape"""
    if not x.is_contiguous():
        raise ValueError("x is not contiguous!")
    if x.ndim != 2:
        raise ValueError("x should be a 2-dim tensor!")
    x = x.detach().cpu()

    if tag == "zN":
        return _host_ND_2_zN_impl(x)
    elif tag == "nZ":
        return _host_ND_2_zN_impl(x.t().contiguous())
    raise ValueError(f"Unsupported tag: {tag}")
