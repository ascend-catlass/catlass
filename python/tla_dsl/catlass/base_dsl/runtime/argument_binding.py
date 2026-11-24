"""TLA-specific physical bindings for struct-like runtime arguments.

The generic structure lives in ``base_dsl.utils.tree_utils``.  This module only
describes how a flattened logical leaf maps to the existing TLA launcher ABI.
A Dynamic-GM tensor is one logical leaf whose physical ABI is a descriptor
field group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..utils.tree_utils import TreeDef


@dataclass(frozen=True)
class RuntimeLeafBinding:
    """Describe how a logical leaf is lowered and reconstructed."""

    path: tuple[Any, ...]
    mlir_type: Any
    # Binding strategy, not the final physical ABI type.
    kind: str  # pointer/scalar/dynamic_gm


class _RuntimeArgumentProxy:
    """Frontend proxy for a materialized Dynamic-GM descriptor.

    The lowering layer owns the mapping from this proxy to an MLIR block
    value. This class only defines the tensor-facing operations that user
    kernel code may perform while the frontend is emitting IR.
    """

    __slots__ = ()

    @property
    def ptr(self) -> Any:
        """Resolve ``arg.ptr`` to a ``tla.tensor_ptr`` operation."""
        from ... import runtime
        from ...base_dsl.op import _capture_user_loc
        from ...core_api import _as_value, _emit_tensor_ptr

        loc = (
            _capture_user_loc()
            if runtime._current_frontend_state() is not None
            else None
        )
        return _emit_tensor_ptr(_as_value(self), loc)

    @property
    def ptr_with_coord(self) -> Any:
        """Resolve ``arg.ptr_with_coord`` to a ``tla.tensor_ptr_with_coord`` operation."""
        from ... import runtime
        from ...base_dsl.op import _capture_user_loc
        from ...core_api import _as_value, _emit_tensor_ptr_with_coord

        loc = (
            _capture_user_loc()
            if runtime._current_frontend_state() is not None
            else None
        )
        return _emit_tensor_ptr_with_coord(_as_value(self), loc)

    def _metadata(self, field: str) -> Any:
        from ...core_api import _as_value, _tensor_metadata_field

        return _tensor_metadata_field(_as_value(self), field)

    @property
    def shape(self) -> Any:
        return self._metadata("shape")

    @property
    def stride(self) -> Any:
        return self._metadata("stride")

    @property
    def origin_shape(self) -> Any:
        return self._metadata("origin_shape")

    def __getitem__(self, crd: Any) -> Any:
        """Lower indexed reads through the TLA tensor implementation."""
        from ... import runtime
        from ...base_dsl.op import _capture_user_loc
        from ...tla.tensor import _Tensor

        # ``_Tensor.__getitem__`` resolves the proxy through frontend state.
        # Keep the location capture here so diagnostics point at user code.
        loc = (
            _capture_user_loc()
            if runtime._current_frontend_state() is not None
            else None
        )
        return _Tensor.__getitem__(self, crd, loc=loc)

    def __setitem__(self, crd: Any, data: Any) -> None:
        """Lower indexed stores through the TLA tensor implementation."""
        from ... import runtime
        from ...base_dsl.op import _capture_user_loc
        from ...tla.tensor import _Tensor

        loc = (
            _capture_user_loc()
            if runtime._current_frontend_state() is not None
            else None
        )
        return _Tensor.__setitem__(self, crd, data, loc=loc)


@dataclass(frozen=True)
class RuntimeArgumentTree:
    """Compiled view of one top-level argument tree.

    ``treedef`` is the generic structural tree.  ``bindings`` is the TLA
    physical view used by lowering and launch.  Keeping this small bundle in a
    backend runtime module prevents generic tree utilities from depending on
    the Ascend ABI.
    """

    treedef: TreeDef
    bindings: tuple[RuntimeLeafBinding, ...]

    @property
    def runtime_leaf_count(self) -> int:
        return len(self.bindings)


__all__ = [
    "_RuntimeArgumentProxy",
    "RuntimeArgumentTree",
    "RuntimeLeafBinding",
]
