"""Native bridge for Tla TypeDef construction/access from structured data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]


class TlaTypeBridgeUnavailableError(RuntimeError):
    """Raised when native Tla type construction is unavailable."""


_EXTENSION: ModuleType | None | bool = None
_TREE_OPEN = -2
_TREE_CLOSE = -3
_DYNAMIC = -(2**63)


def shape_type_get(context: mlir_ir.Context, tree: Any) -> mlir_ir.Type:
    return _load_bridge_extension().shape_type_get(
        context, list(_encode_index_tree(tree))
    )


def coord_type_get(context: mlir_ir.Context, tree: Any) -> mlir_ir.Type:
    return _load_bridge_extension().coord_type_get(
        context, list(_encode_index_tree(tree))
    )


def stride_type_get(context: mlir_ir.Context, tree: Any) -> mlir_ir.Type:
    return _load_bridge_extension().stride_type_get(
        context, list(_encode_index_tree(tree))
    )


def layout_type_get(
    context: mlir_ir.Context,
    shape: Any,
    stride: Any,
    origin_shape: Any | None = None,
    layout: str = "RowMajor",
) -> mlir_ir.Type:
    origin = None if origin_shape is None else list(_encode_index_tree(origin_shape))
    return _load_bridge_extension().layout_type_get(
        context,
        list(_encode_index_tree(shape)),
        list(_encode_index_tree(stride)),
        origin,
        str(layout),
    )


def tensor_type_get(
    context: mlir_ir.Context,
    *,
    shape: Any,
    stride: Any,
    coord: Any,
    origin_shape: Any,
    element_type: mlir_ir.Type,
    addrspace: str,
    layout: str,
    ptr_alignment: int,
) -> mlir_ir.Type:
    if not isinstance(element_type, mlir_ir.Type):
        raise TypeError(
            "Tla tensor bridge expects element_type as mlir.ir.Type, "
            f"got {type(element_type).__name__}"
        )
    return _load_bridge_extension().tensor_type_get(
        context,
        list(_encode_index_tree(shape)),
        list(_encode_index_tree(stride)),
        list(_encode_index_tree(coord)),
        list(_encode_index_tree(origin_shape)),
        element_type,
        str(addrspace),
        str(layout),
        int(ptr_alignment),
    )


def ptr_type_get(
    context: mlir_ir.Context,
    pointee: mlir_ir.Type,
    addrspace: str,
    alignment: int,
) -> mlir_ir.Type:
    return _load_bridge_extension().ptr_type_get(
        context, pointee, str(addrspace), int(alignment)
    )


def ptr_pointee_type_get(
    context: mlir_ir.Context, ptr_type: mlir_ir.Type
) -> mlir_ir.Type:
    del context
    return _load_bridge_extension().ptr_pointee_type_get(ptr_type)


def ptr_addrspace(ptr_type: mlir_ir.Type) -> str:
    return str(_load_bridge_extension().ptr_addrspace(ptr_type))


def ptr_alignment(ptr_type: mlir_ir.Type) -> int:
    return int(_load_bridge_extension().ptr_alignment(ptr_type))


def tensor_ptr_type_get(tensor_type: mlir_ir.Type) -> mlir_ir.Type:
    """Return the ``!tla.ptr`` type embedded in a ``!tla.tensor<...>`` type."""
    return _load_bridge_extension().tensor_ptr_type_get(tensor_type)


def layout_type_from_components_get(
    context: mlir_ir.Context,
    shape_type: mlir_ir.Type,
    stride_type: mlir_ir.Type,
    origin_shape_type: mlir_ir.Type | None = None,
    layout: str = "RowMajor",
) -> mlir_ir.Type:
    return _load_bridge_extension().layout_type_from_components_get(
        context, shape_type, stride_type, origin_shape_type, str(layout)
    )


def vector_ssa_type_get(
    context: mlir_ir.Context,
    valid_lanes: int | None,
    element_type: mlir_ir.Type,
) -> mlir_ir.Type:
    if valid_lanes is not None and (
        isinstance(valid_lanes, bool) or not isinstance(valid_lanes, int)
    ):
        raise TypeError(
            "Tla vector bridge expects valid_lanes as int or None, "
            f"got {type(valid_lanes).__name__}"
        )
    if not isinstance(element_type, mlir_ir.Type):
        raise TypeError(
            "Tla vector bridge expects element_type as mlir.ir.Type, "
            f"got {type(element_type).__name__}"
        )
    return _load_bridge_extension().vector_ssa_type_get(
        context, valid_lanes, element_type
    )


def vector_ssa_element_type_get(vector_type: mlir_ir.Type) -> mlir_ir.Type:
    return _load_bridge_extension().vector_ssa_element_type_get(vector_type)


def vector_ssa_valid_lanes_get(vector_type: mlir_ir.Type) -> int | None:
    lanes = _load_bridge_extension().vector_ssa_valid_lanes_get(vector_type)
    return None if lanes is None else int(lanes)


def mask_ssa_type_get(context: mlir_ir.Context, physical_lanes: int) -> mlir_ir.Type:
    if isinstance(physical_lanes, bool) or not isinstance(physical_lanes, int):
        raise TypeError(
            f"Tla mask bridge expects physical_lanes as int, got {type(physical_lanes).__name__}"
        )
    return _load_bridge_extension().mask_ssa_type_get(context, physical_lanes)


def mask_ssa_physical_lanes_get(mask_type: mlir_ir.Type) -> int:
    return int(_load_bridge_extension().mask_ssa_physical_lanes_get(mask_type))


def flag_type_get(context: mlir_ir.Context) -> mlir_ir.Type:
    return _load_bridge_extension().flag_type_get(context)


def cross_flag_type_get(context: mlir_ir.Context, mode: int) -> mlir_ir.Type:
    return _load_bridge_extension().cross_flag_type_get(context, mode)


def cross_flag_mode(type_like: mlir_ir.Type) -> int:
    return int(_load_bridge_extension().cross_flag_mode(type_like))


def mutex_type_get(context: mlir_ir.Context) -> mlir_ir.Type:
    return _load_bridge_extension().mutex_type_get(context)


def float4_e2m1_type_get(context: mlir_ir.Context) -> mlir_ir.Type:
    return _load_bridge_extension().float4_e2m1_type_get(context)


def float4_e1m2_type_get(context: mlir_ir.Context) -> mlir_ir.Type:
    return _load_bridge_extension().float4_e1m2_type_get(context)


def float8_e8m0_type_get(context: mlir_ir.Context) -> mlir_ir.Type:
    return _load_bridge_extension().float8_e8m0_type_get(context)


def hifloat8_type_get(context: mlir_ir.Context) -> mlir_ir.Type:
    return _load_bridge_extension().hifloat8_type_get(context)


def copy_l0c2dst_params_type_get(context: mlir_ir.Context) -> mlir_ir.Type:
    return _load_bridge_extension().copy_l0c2dst_params_type_get(context)


def load_tla_dialect(context: mlir_ir.Context) -> None:
    _load_bridge_extension().load_tla_dialect(context)


def type_is_ptr(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_ptr(type_like))


def type_is_tensor(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_tensor(type_like))


def type_is_shape(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_shape(type_like))


def type_is_coord(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_coord(type_like))


def type_is_stride(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_stride(type_like))


def type_is_layout(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_layout(type_like))


def type_is_vector_ssa(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_vector_ssa(type_like))


def type_is_mask_ssa(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_mask_ssa(type_like))


def type_is_flag(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_flag(type_like))


def type_is_cross_flag(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_cross_flag(type_like))


def type_is_mutex(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_mutex(type_like))


def type_is_copy_l0c2dst_params(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_copy_l0c2dst_params(type_like))


def tla_type_category(type_like: mlir_ir.Type) -> str | None:
    category = _load_bridge_extension().tla_type_category(type_like)
    return None if category is None else str(category)


def _encode_index_tree(tree: Any) -> tuple[int, ...]:
    encoded: list[int] = []

    def append(node: Any, *, tuple_depth: int) -> None:
        if isinstance(node, bool):
            raise TypeError("boolean not allowed in Tla index tree")
        if node is None:
            encoded.append(_DYNAMIC)
            return
        if isinstance(node, int):
            encoded.append(int(node))
            return
        if isinstance(node, tuple):
            if len(node) == 0:
                raise ValueError("empty tuple in index tree")
            if tuple_depth >= 2:
                raise ValueError(
                    "Tla index tree supports only top-level leaves or one-level leaf groups"
                )
            encoded.append(_TREE_OPEN)
            for child in node:
                append(child, tuple_depth=tuple_depth + 1)
            encoded.append(_TREE_CLOSE)
            return
        if isinstance(node, list):
            raise ValueError(
                "expected nested tuple tree for index components (use parentheses, not brackets)"
            )
        raise TypeError(
            "Tla index type metadata expects static int leaves or None for dynamic leaves; "
            f"got {type(node).__name__}"
        )

    if isinstance(tree, tuple):
        if len(tree) == 0:
            raise ValueError("empty tuple in index tree")
        for child in tree:
            append(child, tuple_depth=1)
    else:
        append(tree, tuple_depth=0)
    return tuple(encoded)


def _load_bridge_extension() -> ModuleType:
    global _EXTENSION
    if _EXTENSION is False:
        raise TlaTypeBridgeUnavailableError("Tla type bridge extension is unavailable")
    if isinstance(_EXTENSION, ModuleType):
        return _EXTENSION
    path = _resolve_bridge_extension_path()
    if path is None:
        _EXTENSION = False
        raise TlaTypeBridgeUnavailableError(
            "Tla type bridge extension not found. Build the native type bridge "
            "module (e.g. via build.sh)."
        )
    spec = importlib.util.spec_from_file_location(
        "catlass._tla_type_bridge_native", path
    )
    if spec is None or spec.loader is None:
        _EXTENSION = False
        raise TlaTypeBridgeUnavailableError(f"Failed to load Tla type bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EXTENSION = module
    return module


def _resolve_bridge_extension_path() -> Path | None:
    packaged_root = Path(__file__).resolve().parent
    # Nested in Catlass: .../python/tla_dsl/catlass/this.py -> .../python/tla_dsl/csrc/mlir/build/...
    dsl_root = Path(__file__).resolve().parents[1]
    nested = sorted(
        (dsl_root / "csrc" / "mlir" / "build" / "python" / "catlass").glob(
            "_tla_type_bridge_native*.so"
        )
    )
    if nested:
        return nested[0]
    # A source worktree can retain an older editable-install artifact in the
    # package directory. Prefer the nested build above whenever it exists;
    # packaged artifacts remain the fallback for wheels and install-only use.
    packaged = sorted(packaged_root.glob("_tla_type_bridge_native*.so"))
    if packaged:
        return packaged[0]
    # Legacy standalone ascend-catlass-DSL repo: .../python/<pkg>/this.py -> repo/mlir/build/...
    repo_root = Path(__file__).resolve().parents[2]
    candidates = sorted(
        (repo_root / "mlir" / "build" / "python" / "catlass").glob(
            "_tla_type_bridge_native*.so"
        )
    )
    if candidates:
        return candidates[0]
    return None


__all__ = [
    "TlaTypeBridgeUnavailableError",
    "tla_type_category",
    "coord_type_get",
    "cross_flag_type_get",
    "cross_flag_mode",
    "flag_type_get",
    "layout_type_from_components_get",
    "layout_type_get",
    "load_tla_dialect",
    "mask_ssa_physical_lanes_get",
    "mask_ssa_type_get",
    "mutex_type_get",
    "ptr_addrspace",
    "ptr_alignment",
    "ptr_pointee_type_get",
    "ptr_type_get",
    "shape_type_get",
    "stride_type_get",
    "tensor_type_get",
    "type_is_coord",
    "type_is_cross_flag",
    "type_is_flag",
    "type_is_layout",
    "type_is_mask_ssa",
    "type_is_mutex",
    "type_is_ptr",
    "type_is_shape",
    "type_is_stride",
    "type_is_tensor",
    "type_is_vector_ssa",
    "vector_ssa_element_type_get",
    "vector_ssa_type_get",
    "vector_ssa_valid_lanes_get",
]
