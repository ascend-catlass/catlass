# -----------------------------------------------------------------------------------------------------------
# Host tensor helpers for FFN Phase-0 runners (copied from catlass_master e2e common).
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import torch  # noqa: F401
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack
from catlass.types import dtype_size_bytes

UB_ALLOC_ALIGN_BYTES = 256

_STORAGE_VIEW_BY_WIDTH = {
    8: torch.int8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}


def compute_ub_slot_elems(
    dtype_c: str,
    *,
    nodes: int,
    stages: int,
    budget_bytes: int | None = None,
) -> int:
    """Size one UB slot so ``nodes * stages`` of them fit the buffer."""
    elem_bytes = dtype_size_bytes(dtype_c)
    if budget_bytes is None:
        budget_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)
    align_elems = UB_ALLOC_ALIGN_BYTES // elem_bytes
    slot_elems = budget_bytes // nodes // stages // elem_bytes
    return slot_elems // align_elems * align_elems


def get_block_num(block_num: int, device: int = 0, *, kind: str = "vector") -> int:
    """Get launch ``block_num``."""
    if int(block_num) != -1:
        return max(1, int(block_num))
    props = torch.npu.get_device_properties(int(device))
    if kind == "vector":
        return max(1, int(props.vector_core_num))
    if kind in {"cube", "mix"}:
        return max(1, int(props.cube_core_num))
    raise ValueError(f"Unsupported kernel kind for block_num default: {kind!r}")


def create_tla_tensor(
    buf, layout: str, element_type=None, *, force_contiguous: bool = True
):
    """Wrap a device buffer as a TLA tensor.

    ``layout``: ``"row"``, ``"col"``/``"column"``, or ``"vector"`` (1-D bias).
    ``force_contiguous=False`` keeps a non-contiguous view (e.g. ``w1.t()`` on
    contiguous ``[N,K]`` → shape ``[K,N]`` strides ``(1,K)``) so ColumnMajor can
    match CPP linear weight storage without a host transpose copy.
    """
    layout_l = str(layout).lower()
    if layout_l == "row":
        tag = tla.arch.RowMajor
    elif layout_l in ("col", "column"):
        tag = tla.arch.ColumnMajor
    elif layout_l == "vector":
        tag = tla.arch.VectorLayout
    else:
        raise ValueError(
            f"create_tla_tensor layout must be row/col/column/vector; got {layout!r}"
        )
    if element_type is not None:
        width = element_type.width
        if width not in _STORAGE_VIEW_BY_WIDTH:
            raise ValueError(
                f"no storage view for a {width}-bit element type {element_type.__name__}"
            )
        buf = buf.view(_STORAGE_VIEW_BY_WIDTH[width])
    if force_contiguous:
        buf = buf.contiguous()
    return from_dlpack(
        buf, layout_tag=tag, element_type=element_type
    ).mark_layout_dynamic()


__all__ = [
    "UB_ALLOC_ALIGN_BYTES",
    "compute_ub_slot_elems",
    "create_tla_tensor",
    "get_block_num",
]
