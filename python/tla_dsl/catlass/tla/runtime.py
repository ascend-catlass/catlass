"""Runtime DLPack tensor binding."""

from __future__ import annotations

import ctypes
import weakref
from typing import Any, Iterable, Iterator

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

from ..address_space import AddressSpace
from ..base_dsl.runtime.dlpack_types import (
    ASCEND_DEVICE_TYPES,
    DLDeviceType,
    DLManagedTensor,
)
from ..types import (
    RuntimeTensorError,
    TlaIndexTreeType,
    TlaLayoutDescriptor,
    TlaTensorTypeDescriptor,
    _coerce_host_tensor_addrspace,
    _coerce_host_tensor_dtype,
    _deduce_compact_stride_order,
    _flat_layout_leaves,
    _flatten_int_leaves_tree,
    _replace_flat_leaves_in_tree,
    _tree_structure_mask,
    _try_remap_stride_coord_trees,
    dtype_size_bytes,
)
from .typing import Tensor as TensorABC


def _as_host_index_tree(value: Any, *, what: str) -> Any:
    """Normalize a Host index tree: bare ``int`` → ``(int,)``; otherwise require ``tuple``."""
    if isinstance(value, int):
        return (value,)
    if isinstance(value, tuple):
        return value
    raise TypeError(
        f"{what} must be an int or nested int tuple, got {type(value).__name__}"
    )


def _is_contiguous_run(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    """True when a view reads consecutive elements, ignoring size-1 extents.

    A size-1 extent addresses nothing extra, so torch is free to leave its stride
    at whatever the parent had; only extents above 1 constrain the layout. This is
    torch's ``is_contiguous()`` test, spelled out so the reason is visible.
    """
    expected = 1
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if extent <= 1:
            continue
        if stride != expected:
            return False
        expected *= extent
    return True


def _require_flat_1d_binding(
    layout_token: str,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    *,
    projects_to_rank_one: bool,
) -> None:
    """Reject a buffer that a flat one-dimensional layout cannot describe.

    ``RowMajor`` and ``VectorLayout`` flatten a rank-1 tile to ``(N):(1)`` -- the
    bc layer builds exactly that, with a literal unit stride -- so the kernel
    reads ``base + k`` for ``k`` in ``[0, N)``. A strided view
    (``arange(64)[::2]``, strides ``(2,)``) would therefore be addressed dense and
    silently read the wrong elements.

    Whether the tile is flattened depends on the *projected* rank, i.e. on
    ``origin_shape``, not on the physical DLPack rank: a ``(1, 32)`` slice of a
    ``(1, 64)`` buffer with a rank-1 ``origin_shape`` lands in the same flat form
    while its real stride is still 2. So this asks whether the buffer is one
    contiguous run, whatever rank it is spelled in.

    ``tla.make_tensor`` checks the tile metadata the kernel builds, but a DLPack
    buffer's real strides never reach it -- this is the binding's own guard.
    """
    from ..core_api import _FLAT_1D_LAYOUT_TOKENS

    if layout_token not in _FLAT_1D_LAYOUT_TOKENS or not projects_to_rank_one:
        return
    if _is_contiguous_run(shape, strides):
        return
    raise RuntimeTensorError(
        f"from_dlpack layout_tag={layout_token!r} flattens its tile to a "
        f"contiguous run, but got shape={shape} strides={strides}: this view "
        f"does not read consecutive elements, so the kernel would address it "
        f"dense and read the wrong data. Take a contiguous copy first (e.g. "
        f"tensor.contiguous())."
    )


class DlpackBridgeError(RuntimeError):
    """Raised when DLPack export or parsing fails."""


def dlpack_device_type(tensor: Any) -> int | None:
    """Read ``__dlpack_device__()``'s device type, or ``None`` if unavailable."""
    probe = getattr(tensor, "__dlpack_device__", None)
    if probe is None:
        return None
    try:
        device = probe()
    except Exception:  # noqa: BLE001 - a broken probe must not mask the export
        return None
    try:
        return int(device[0])
    except (TypeError, ValueError, IndexError):
        return None


def export_dlpack_capsule(tensor: Any, *, stream: int | None = -1) -> Any:
    """Call ``__dlpack__``, preferring ``stream=-1`` when supported."""
    if not hasattr(tensor, "__dlpack__"):
        raise DlpackBridgeError(
            f"object {type(tensor).__name__!r} does not implement __dlpack__()"
        )
    if hasattr(tensor, "__dlpack_device__") and not hasattr(tensor, "__dlpack__"):
        raise DlpackBridgeError(
            "tensor exposes __dlpack_device__ only; full __dlpack__() is required"
        )
    # A stream is a device-queue handle, so it is meaningless for a host buffer
    # and producers say so bluntly: torch asserts `stream should be None on cpu`.
    # Ask for no stream up front rather than relying on the retry below -- an
    # AssertionError is not something a library should be catching by habit.
    if stream is not None and dlpack_device_type(tensor) == DLDeviceType.kDLCPU:
        stream = None
    if stream is not None:
        try:
            return tensor.__dlpack__(stream=stream)  # type: ignore[attr-defined]
        except (TypeError, RuntimeError, AssertionError):
            # NumPy raises RuntimeError ("only supports stream=None"); others may
            # reject unknown stream via TypeError, or assert on it.
            pass
    return tensor.__dlpack__()  # type: ignore[attr-defined]


def _consume_capsule(capsule: Any) -> None:
    """Take ownership of a ``dltensor`` capsule by renaming it to ``used_dltensor``.

    A ``dltensor`` capsule's destructor calls ``DLManagedTensor.deleter`` when the
    capsule dies unconsumed, which would release the buffer as soon as the local
    reference goes out of scope. Renaming is the DLPack handshake that says the
    consumer now owns the ``DLManagedTensor`` and will call the deleter itself
    (see :func:`_release_managed_tensor`); producers key their capsule destructor
    on the old name, so it becomes a no-op.
    """
    set_name = ctypes.pythonapi.PyCapsule_SetName
    set_name.restype = ctypes.c_int
    set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
    # The name is borrowed by the capsule, so it must outlive it: keep the bytes
    # object alive in a module-level constant rather than a temporary.
    if set_name(capsule, _USED_DLTENSOR_NAME) != 0:
        raise DlpackBridgeError("failed to rename DLPack capsule to 'used_dltensor'")


# PyCapsule_SetName borrows the pointer; this must never be garbage-collected.
_USED_DLTENSOR_NAME = b"used_dltensor"


def _release_managed_tensor(managed_ptr: int) -> None:
    """Call ``DLManagedTensor.deleter`` once, releasing the producer's allocation.

    Registered through :func:`weakref.finalize` on the owning tensor, so it must
    not close over that tensor (a strong reference would keep it alive forever).
    """
    managed = ctypes.cast(ctypes.c_void_p(managed_ptr), ctypes.POINTER(DLManagedTensor))
    deleter = managed.contents.deleter
    # A NULL deleter means the producer has nothing to release (DLPack allows it).
    if deleter:
        deleter(managed)


class _Tensor(TensorABC):
    """TLA runtime host tensor with compile metadata and optional DLPack binding.

    Not a Host public constructor. Use :func:`make_fake_tensor` (unbound sample) or
    :func:`from_dlpack` (NPU buffer). ``tla.Tensor`` is the shared ABC / annotation.
    """

    @staticmethod
    def _parse_capsule(capsule: Any) -> dict[str, Any]:
        """Read a ``dltensor`` capsule's fields; does not consume it (see :func:`_consume_capsule`)."""
        is_valid = ctypes.pythonapi.PyCapsule_IsValid
        is_valid.restype = ctypes.c_int
        is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
        # Check the name before reading it: PyCapsule_GetPointer on a renamed
        # capsule returns NULL *and* sets the error indicator, which would
        # otherwise surface later at an unrelated call site.
        if not is_valid(capsule, b"dltensor"):
            raise DlpackBridgeError(
                "not a fresh 'dltensor' capsule; a DLPack capsule can only be "
                "consumed once (an already-consumed capsule is renamed to "
                "'used_dltensor')"
            )
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
            ctypes.py_object,
            ctypes.c_char_p,
        ]
        # Capsule holds DLManagedTensor*; dl_tensor is its first member (dlpack.h).
        managed_ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")
        dl = ctypes.cast(
            managed_ptr, ctypes.POINTER(DLManagedTensor)
        ).contents.dl_tensor
        ndim = int(dl.ndim)
        shape = tuple(int(dl.shape[i]) for i in range(ndim))
        if not dl.strides:
            raise DlpackBridgeError(
                "DLPack tensor has null strides; explicit per-dimension strides are "
                "required (e.g. torch_npu / PyTorch export). NumPy C-contiguous "
                "buffers are not supported."
            )
        strides = tuple(int(dl.strides[i]) for i in range(ndim))
        data_ptr = int(dl.data or 0) + int(dl.byte_offset)
        return {
            "managed_ptr": int(managed_ptr),
            "data_ptr": data_ptr,
            "device_type": int(dl.device.device_type),
            "device_id": int(dl.device.device_id),
            "shape": shape,
            "strides": strides,
            "dtype_code": int(dl.dtype.code),
            "dtype_bits": int(dl.dtype.bits),
            "dtype_lanes": int(dl.dtype.lanes),
        }

    def __init__(
        self,
        shape: Any,
        dtype: Any,
        *,
        addrspace: Any = AddressSpace.gm,
        data_ptr: int | None = 0,
        origin_shape: Iterable[Any] | None = None,
        coord: Iterable[Any] | None = None,
        stride: Any | None = None,
        layout_tag: Any | None = None,
        flat_shape: tuple[int, ...] | None = None,
        flat_stride: tuple[int, ...] | None = None,
    ) -> None:
        self._external_binding = False
        self._assumed_align: int | None = None
        # DLPack ownership, set by from_dlpack: the consumed DLManagedTensor is
        # released by _dlpack_release when this tensor dies, and _dlpack_source
        # keeps the producer object itself alive meanwhile.
        self._dlpack_source: Any = None
        self._dlpack_release: weakref.finalize | None = None
        self._shape: tuple[Any, ...] | None = None
        self._stride: tuple[Any, ...] | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._flat_stride: tuple[int, ...] | None = None
        self._dynamic_shape_tree: Any | None = None
        self._dynamic_origin_shape_tree: Any | None = None
        self._dynamic_stride_tree: Any | None = None

        # None is treated as unbound (compile-time placeholder tensors).
        self.data_ptr = 0 if data_ptr is None else int(data_ptr)
        # Non-zero data_ptr means the host already owns a device buffer (e.g. torch).
        if self.data_ptr != 0:
            self._external_binding = True
        from ..core_api import _resolve_arch_layout_tag

        self._dtype = _coerce_host_tensor_dtype(dtype)
        self.addrspace = _coerce_host_tensor_addrspace(addrspace)
        self.layout_tag = _resolve_arch_layout_tag(layout_tag, for_op="Tensor")

        shape_tree = _as_host_index_tree(shape, what="shape")
        resolved_flat_shape = (
            tuple(int(value) for value in _flatten_int_leaves_tree(shape_tree))
            if flat_shape is None
            else flat_shape
        )
        self._flat_shape = resolved_flat_shape
        self._shape = shape_tree

        if origin_shape is None:
            raise TypeError("Tensor origin_shape is required")
        self.origin_shape = _as_host_index_tree(origin_shape, what="origin_shape")

        rs_stride, rs_coord = _try_remap_stride_coord_trees(
            shape_tree, self.origin_shape, self._dtype, self.layout_tag
        )

        if coord is None:
            if rs_coord is None:
                raise ValueError(
                    "Tensor(coord=None): cannot derive coord from layout remap; "
                    "use a flat logical origin_shape (M,N) without nested parentheses, "
                    "or pass an explicit coord tree"
                )
            self.coord = rs_coord
        else:
            self.coord = _as_host_index_tree(coord, what="coord")

        if stride is None:
            if rs_stride is None:
                raise ValueError(
                    "Tensor(stride=None): cannot derive stride from layout remap "
                    "(remap stride tree must match shape tree); pass an explicit stride tree"
                )
            self._stride = rs_stride
        else:
            stride_tree = _as_host_index_tree(stride, what="stride")
            if _tree_structure_mask(stride_tree) != _tree_structure_mask(shape_tree):
                raise ValueError(
                    "Tensor stride component tree must match shape tree structure"
                )
            self._stride = stride_tree
        resolved_flat_stride = (
            tuple(int(value) for value in _flatten_int_leaves_tree(self._stride))
            if flat_stride is None
            else flat_stride
        )
        self._flat_stride = resolved_flat_stride
        if len(resolved_flat_stride) != len(resolved_flat_shape):
            raise RuntimeTensorError("Tensor shape/stride flat metadata rank mismatch")

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def shape(self) -> Any:
        if self._shape is None:
            raise NotImplementedError(f"{type(self).__name__} does not expose shape")
        return self._shape

    @property
    def stride(self) -> Any:
        return self._stride

    def _require_bound(self) -> None:
        if not getattr(self, "_external_binding", False) or self.data_ptr == 0:
            raise RuntimeError(
                "Tensor buffer is not bound to a device address, so it can be "
                "compiled against but not launched; it came from "
                "make_fake_tensor or from a host (CPU) buffer. Bind the device "
                "buffer with from_dlpack (e.g. move the torch tensor to 'npu' "
                "first)."
            )

    def mark_layout_dynamic(self, leading_dim: int | None = None) -> "_Tensor":
        """Directory: Host Tensor / Dynamic Layout
        Description:
            Mark every shape mode dynamic so one compiled artifact can run at
            different extents. Strides become dynamic except the leading dimension
            (stride stays `1`). Broadcast strides of `0` are kept. Matching
            `origin_shape` leaves become dynamic so the compile type no longer
            depends on concrete DLPack extents.

            Parameters:
            - *`leading_dim`* (`int | None`): Index of the unit-stride (leading)
              dimension. Optional; default `None` (inferred from `layout_tag` or
              compact stride order).

            Constraints:
            - In-place; returns `self` (chainable).
            - All `coord` leaves must be `0`; sliced views are rejected.
            - `leading_dim` must have stride `1`.
            - For NZFamily layouts, each two-leaf physical shape group maps to one
              logical `origin_shape` axis.

            Example:
            ```python
            ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
            ta = ta.mark_layout_dynamic()
            artifact = tla.compile(my_kernel, ta, options="--npu-arch 3510")
            ```

        """
        # Dynamic GM ABI hard-codes root coord/offset 0 (same rule as
        # TlaLowerFuncPass::validateKernelTensorArg).
        coord_leaves = _flat_layout_leaves(self.coord, allow_dynamic=True)
        if any(leaf is None for leaf in coord_leaves) or not all(
            int(leaf) == 0 for leaf in coord_leaves
        ):
            raise RuntimeTensorError(
                "mark_*_dynamic requires a root tensor with zero coordinates; "
                f"got coord={self.coord!r}"
            )
        flat_strides = _flat_layout_leaves(self.stride)
        shape_tuple = self._flat_shape or ()
        from ..core_api import (
            _MX_SCALE_GM_DYNAMIC_STRIDE_LEAF,
            _MX_SCALE_GM_LAYOUT_TOKENS,
        )

        if leading_dim is None and self.layout_tag in _MX_SCALE_GM_LAYOUT_TOKENS:
            # NZFamily GM MxScale trees have no linear leading dimension: the
            # structural constants (0 broadcast, 1, e8m0 scale C0 = 2) stay
            # static and only the pitch leaf goes dynamic. The linear
            # leading-dim inference below would either pick a 0-stride leaf and
            # reject it, or freeze a pitch that must stay runtime-varying.
            pitch_leaf = _MX_SCALE_GM_DYNAMIC_STRIDE_LEAF[self.layout_tag]
            new_stride_leaves = [
                None if index == pitch_leaf else int(leaf)
                for index, leaf in enumerate(flat_strides)
            ]
            self._dynamic_stride_tree = _replace_flat_leaves_in_tree(
                self.stride, new_stride_leaves
            )
            self._mark_shape_modes_dynamic(range(len(shape_tuple)))
            return self
        if leading_dim is None:
            from ..core_api import (
                _COLUMN_MAJOR_LAYOUT_TOKENS,
                _ROW_MAJOR_LAYOUT_TOKENS,
            )

            # Prefer layout-tag semantics when unit strides are ambiguous
            # (e.g. ColumnMajor with shape[0]==1 → strides (1,1)).
            if self.layout_tag in _ROW_MAJOR_LAYOUT_TOKENS:
                leading_dim = len(flat_strides) - 1
            elif self.layout_tag in _COLUMN_MAJOR_LAYOUT_TOKENS:
                leading_dim = 0
            else:
                leading_dim = _deduce_compact_stride_order(
                    shape_tuple, flat_strides, strict_unit_stride=True
                )[-1]
        if leading_dim < 0 or leading_dim >= len(flat_strides):
            raise RuntimeTensorError(
                f"leading_dim={leading_dim} out of range for rank {len(flat_strides)}"
            )
        if int(flat_strides[leading_dim]) != 1:
            raise RuntimeTensorError(
                f"leading_dim={leading_dim} has stride {flat_strides[leading_dim]}, expected 1"
            )
        new_stride_leaves = [
            1
            if index == leading_dim
            else (0 if int(flat_strides[index]) == 0 else None)
            for index in range(len(flat_strides))
        ]
        self._dynamic_stride_tree = _replace_flat_leaves_in_tree(
            self.stride, new_stride_leaves
        )
        self._mark_shape_modes_dynamic(range(len(shape_tuple)))
        return self

    def mark_compact_shape_dynamic(
        self,
        mode: int,
        stride_order: tuple[int, ...] | None = None,
    ) -> "_Tensor":
        """Directory: Host Tensor / Dynamic Layout
        Description:
            Mark one compact shape mode dynamic. Strides of modes major to `mode`
            (whose compact stride product includes that extent) become dynamic as
            well. Matching `origin_shape` leaves are marked so the compile type does
            not depend on the concrete size.

            Parameters:
            - *`mode`* (`int`): Flattened shape-leaf index to mark dynamic
              (0-based). Required.
            - *`stride_order`* (`tuple[int, ...] | None`): Compact stride order
              (outer → inner). Optional; inferred from current strides when omitted.

            Constraints:
            - In-place; returns `self`.
            - All `coord` leaves must be `0`.
            - `stride_order` must be a permutation of `range(rank)`.
            - For NZFamily layouts, physical modes 0/1 map to logical M and modes
              2/3 map to logical N.

            Example:
            ```python
            ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
            ta = ta.mark_compact_shape_dynamic(mode=0)
            ```

        """
        coord_leaves = _flat_layout_leaves(self.coord, allow_dynamic=True)
        if any(leaf is None for leaf in coord_leaves) or not all(
            int(leaf) == 0 for leaf in coord_leaves
        ):
            raise RuntimeTensorError(
                "mark_*_dynamic requires a root tensor with zero coordinates; "
                f"got coord={self.coord!r}"
            )
        flat_shape = list(self._flat_shape or ())
        rank = len(flat_shape)
        if mode < 0 or mode >= rank:
            raise RuntimeTensorError(f"mode={mode} out of range for rank {rank}")
        if stride_order is None:
            stride_order = _deduce_compact_stride_order(
                flat_shape, _flat_layout_leaves(self.stride)
            )
        elif len(stride_order) != rank or sorted(stride_order) != list(range(rank)):
            raise RuntimeTensorError(
                f"stride_order {stride_order!r} is not a permutation of range({rank})"
            )

        self._mark_shape_modes_dynamic((mode,))

        # Modes major to ``mode`` (appear before it outer→inner) include its size
        # in their compact stride product → mark those strides dynamic.
        mode_pos = stride_order.index(mode)
        major_modes = set(stride_order[:mode_pos])
        flat_strides = _flat_layout_leaves(
            self._layout_stride_components(),
            allow_dynamic=True,
            expected_rank=rank,
        )
        for index in major_modes:
            if int(flat_strides[index] or 0) != 0:
                flat_strides[index] = None
        self._dynamic_stride_tree = _replace_flat_leaves_in_tree(
            self.stride, flat_strides
        )
        return self

    def _mark_shape_modes_dynamic(self, modes: Any) -> None:
        shape_components = self._shape
        if shape_components is None:
            raise TypeError(
                "Tensor metadata is incomplete; construct tensors with tla.make_shape, "
                "origin_shape, coord, and stride metadata"
            )
        rank = len(self._flat_shape or ())
        mode_indices = tuple(int(mode) for mode in modes)
        if any(mode < 0 or mode >= rank for mode in mode_indices):
            raise RuntimeTensorError(
                f"dynamic shape mode out of range for rank {rank}: {mode_indices!r}"
            )
        shape_leaves = _flat_layout_leaves(
            self._layout_shape_components(),
            allow_dynamic=True,
            expected_rank=rank,
        )
        origin_leaves = _flat_layout_leaves(
            self._layout_origin_shape_components(),
            allow_dynamic=True,
        )
        if len(origin_leaves) == rank:
            origin_mode_indices = mode_indices
        else:
            # NZFamily shape/stride are physical 2x2 trees with four leaves,
            # while origin_shape remains the flat logical (M, N) pair. Both
            # leaves in the first physical group map to M; both leaves in the
            # second group map to N.
            from ..core_api import _NZ_FAMILY_LAYOUT_TOKENS

            is_nz_family_2x2 = (
                self.layout_tag in _NZ_FAMILY_LAYOUT_TOKENS
                and rank == 4
                and len(origin_leaves) == 2
                and _tree_structure_mask(self._layout_shape_components())
                == ((None, None), (None, None))
                and _tree_structure_mask(self._layout_origin_shape_components())
                == (None, None)
            )
            if not is_nz_family_2x2:
                raise RuntimeTensorError(
                    f"layout tree rank mismatch: expected {rank} leaves, "
                    f"got {len(origin_leaves)}"
                )
            origin_mode_indices = tuple(sorted({mode // 2 for mode in mode_indices}))

        for mode in mode_indices:
            shape_leaves[mode] = None
        for mode in origin_mode_indices:
            origin_leaves[mode] = None
        self._dynamic_shape_tree = _replace_flat_leaves_in_tree(
            shape_components, shape_leaves
        )
        self._dynamic_origin_shape_tree = _replace_flat_leaves_in_tree(
            self.origin_shape, origin_leaves
        )

    def build_memref_launch_fields(self) -> tuple[int, ...]:
        """Build unified schema-v4 GM launch values in canonical ABI order."""
        self._require_bound()
        data_ptr = int(self.data_ptr)

        shape = self._flat_shape or ()
        if not shape:
            raise RuntimeTensorError(
                "build_memref_launch_fields requires a concrete DLPack shape"
            )
        rank = len(shape)
        if rank > 4:
            raise RuntimeTensorError(
                "build_memref_launch_fields supports at most 4 shape/stride "
                f"leaves, got {rank}"
            )
        strides = self._flat_stride
        # Concrete origin from construction (mark_* only shadows the type tree).
        origin = self.origin_shape

        return (
            data_ptr,
            data_ptr,
            0,
            shape[0],
            shape[1] if rank > 1 else 1,
            shape[2] if rank > 2 else 1,
            shape[3] if rank > 3 else 1,
            strides[0],
            strides[1] if rank > 1 else 1,
            strides[2] if rank > 2 else 1,
            strides[3] if rank > 3 else 1,
            origin[0],
            origin[1] if len(origin) > 1 else 1,
        )

    def _layout_shape_components(self) -> tuple[Any, ...]:
        shape_components = self._shape
        if shape_components is None:
            raise TypeError(
                "Tensor metadata is incomplete; construct tensors with tla.make_shape, "
                "origin_shape, coord, and stride metadata"
            )
        return self._dynamic_shape_tree or shape_components

    def _layout_origin_shape_components(self) -> Any:
        return self._dynamic_origin_shape_tree or self.origin_shape

    def _layout_stride_components(self) -> Any:
        return self._dynamic_stride_tree or self.stride

    def _ptr_alignment(self) -> int:
        assumed = getattr(self, "_assumed_align", None)
        if assumed is not None:
            return max(1, int(assumed))
        return max(1, dtype_size_bytes(str(self.dtype)))

    def tla_tensor_type_descriptor(self) -> TlaTensorTypeDescriptor:
        """Structured ``!tla.tensor`` descriptor from host metadata."""
        st = self._flat_shape
        addr_kw = (self.addrspace or "gm").strip()
        if st is None or self.stride is None or self.layout_tag is None:
            raise TypeError(
                "Tensor metadata is incomplete; construct tensors with tla.make_shape, "
                "origin_shape, coord, and stride metadata"
            )
        return TlaTensorTypeDescriptor(
            layout=TlaLayoutDescriptor(
                shape=TlaIndexTreeType("shape", self._layout_shape_components()),
                stride=TlaIndexTreeType("stride", self._layout_stride_components()),
                origin_shape=TlaIndexTreeType(
                    "shape", self._layout_origin_shape_components()
                ),
                layout_tag=str(self.layout_tag),
            ),
            coord=self.coord,
            element_type=str(self.dtype),
            addrspace=addr_kw,
            ptr_alignment=self._ptr_alignment(),
        )

    def __tla_type__(self) -> str:
        return str(self.tla_tensor_type_descriptor().to_mlir_type())

    def __c_pointers__(self) -> list[int]:
        self._require_bound()
        return [int(self.data_ptr)]

    def __get_mlir_types__(
        self, context: mlir_ir.Context | None = None
    ) -> list[mlir_ir.Type]:
        return [self.tla_tensor_type_descriptor().to_mlir_type(context)]

    def __new_from_mlir_values__(self, values: list[Any]) -> "_Tensor":
        del values
        return self

    def __setitem__(self, index: int | slice, value: Any) -> None:
        raise TypeError("runtime Tensor is not indexable")

    def __getitem__(self, index: int | slice) -> Any:
        raise TypeError("runtime Tensor is not indexable")

    def __len__(self) -> int:
        raise TypeError("runtime Tensor is not indexable")

    def __iter__(self) -> Iterator[Any]:
        raise TypeError("runtime Tensor is not indexable")

    @property
    def data(self) -> Any:
        raise TypeError("runtime Tensor is not indexable")

    def __str__(self) -> str:
        return self.__tla_type__()

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape!r}, dtype={self.dtype!r}, "
            f"addrspace={self.addrspace!r}, data_ptr={self.data_ptr!r}, "
            f"origin_shape={self.origin_shape!r}, coord={self.coord!r}, "
            f"stride={self.stride!r}, layout_tag={self.layout_tag!r})"
        )

    @property  # type: ignore[misc]
    def __class__(self) -> type[TensorABC]:
        return TensorABC


def from_dlpack(
    tensor_dlpack: object,
    *,
    layout_tag: Any,
    origin_shape: Any | None = None,
    assumed_align: int | None = None,
    stream: int | None = -1,
    element_type: type | None = None,
) -> _Tensor:
    """Directory: Host Tensor / Binding
    Description:
        Bind a DLPack NPU tensor to a TLA Host tensor (zero-copy). The returned
        object shares the device buffer of `tensor_dlpack`.

        Parameters:
        - *`tensor_dlpack`* (`object`): Object implementing `__dlpack__()`.
          An Ascend/NPU buffer (e.g. `torch_npu`) gives a launchable tensor. A
          host (CPU) buffer, such as a torch tensor on CPU, is accepted too,
          but only its metadata is taken: the result has `data_ptr == 0`, so
          it can be compiled against and not launched. Required.
        - *`layout_tag`* (`tla.arch.*`): Layout tag such as `tla.arch.RowMajor`,
          `tla.arch.ColumnMajor`, `tla.arch.zN`. Required.
        - *`origin_shape`* (`tuple | int | None`): Logical origin as a Python int
          tree. Optional; derived from the DLPack physical shape and `layout_tag`
          when omitted. Not a Kernel `tla.make_shape`.
        - *`assumed_align`* (`int | None`): Reserved; currently unused.
        - *`stream`* (`int | None`): Passed to `__dlpack__(stream=...)`. Default
          `-1` (no stream sync). `None` omits the `stream` argument.
        - *`element_type`* (`type | None`): Optional override for the element
          type inferred from DLPack. Default `None` keeps the DLPack type.
          Use when DLPack cannot express the real type (e.g. fp8): pass
          `tla.Float8E4M3FN` / `Float8E5M2`. Must have the same per-element
          bit width as the exported buffer.

        Constraints:
        - A host (CPU) buffer binds metadata only. Use it to compile a kernel
          without a device present; launching with it raises.
        - Ownership follows the DLPack consumer contract: the capsule is consumed
          and its deleter runs when the returned tensor is destroyed. A reference
          to `tensor_dlpack` is also retained, so a temporary source such as
          `from_dlpack(x.contiguous().to(device), ...)` is safe.
        - A capsule is single-use; passing an already-consumed capsule raises
          `RuntimeTensorError`. Call `from_dlpack` again for another binding.
        - 2-D `RowMajor` requires `tensor.contiguous()`. 2-D `ColumnMajor`
          requires `tensor.permute(1, 0).contiguous()`. A mismatch raises
          `RuntimeTensorError`. Providing `origin_shape` skips that check.
        - A rank-1 `origin_shape` under `RowMajor`/`VectorLayout` is bound as a
          flat, unit-stride run. The buffer must then read consecutive elements
          (`torch.arange(64)[::2]` is refused); this holds whatever rank the
          physical slice is spelled in, and `origin_shape` does not skip it.
        - Default layout is static. Call `mark_layout_dynamic` /
          `mark_compact_shape_dynamic` for dynamic extents.
        - When `element_type` is set, its per-element bit width must match the
          exported DLPack buffer.

        Example:
        ```python
        tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
        ty = from_dlpack(
            y.permute(1, 0).contiguous(),
            layout_tag=tla.arch.ColumnMajor,
        )
        ```
    """
    from ..base_dsl.runtime.dlpack_types import DLDataTypeCode
    from ..base_dsl.typing import (
        BFloat16,
        Bool,
        Float16,
        Float32,
        Int16,
        Int32,
        Int64,
        Int8,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
    )
    from ..core_api import (
        _MX_SCALE_GM_LAYOUT_TOKENS,
        _remap_tensor_like_prefix_fields_for_layout_trees,
        _resolve_arch_layout_tag,
    )

    try:
        dlpack_data = export_dlpack_capsule(tensor_dlpack, stream=stream)
    except DlpackBridgeError as exc:
        raise RuntimeTensorError(str(exc)) from exc

    try:
        parsed = _Tensor._parse_capsule(dlpack_data)
    except DlpackBridgeError as exc:
        raise RuntimeTensorError(str(exc)) from exc

    phys_shape = parsed["shape"]
    phys_strides = parsed["strides"]
    data_ptr = int(parsed["data_ptr"])

    device_type = int(parsed["device_type"])
    device_id = int(parsed["device_id"])
    # A host buffer describes shape, stride, dtype and layout exactly as well as
    # a device one; the only thing it cannot supply is an address the kernel
    # could be launched against. Bind it metadata-only (``data_ptr == 0``) so a
    # compile-only path can reuse the run path's own tensor construction instead
    # of restating the shapes beside it, and leave launching to _require_bound.
    host_resident = device_type == DLDeviceType.kDLCPU
    if not host_resident and device_type not in ASCEND_DEVICE_TYPES:
        raise RuntimeTensorError(
            f"DLPack device_type={device_type} device_id={device_id} is not "
            f"an Ascend/NPU device; from_dlpack requires a device-resident buffer "
            f"(supported types: {sorted(ASCEND_DEVICE_TYPES)}), or a host buffer "
            f"for compile-only use."
        )
    if host_resident:
        data_ptr = 0

    lanes = int(parsed["dtype_lanes"])
    if lanes != 1:
        raise RuntimeTensorError(f"unsupported DLPack dtype lanes={lanes}")
    dtype_code = int(parsed["dtype_code"])
    dtype_bits = int(parsed["dtype_bits"])
    # DLPack bool is byte-sized storage (kDLBool, bits=8), matching torch.bool /
    # NumPy bool_. Map to TLA Bool (MLIR i1); tensor element size is still 1 byte.
    dtype_mapping: dict[tuple[int, int], type] = {
        (DLDataTypeCode.kDLInt, 8): Int8,
        (DLDataTypeCode.kDLInt, 16): Int16,
        (DLDataTypeCode.kDLInt, 32): Int32,
        (DLDataTypeCode.kDLInt, 64): Int64,
        (DLDataTypeCode.kDLUInt, 8): UInt8,
        (DLDataTypeCode.kDLUInt, 16): UInt16,
        (DLDataTypeCode.kDLUInt, 32): UInt32,
        (DLDataTypeCode.kDLUInt, 64): UInt64,
        (DLDataTypeCode.kDLFloat, 16): Float16,
        (DLDataTypeCode.kDLFloat, 32): Float32,
        (DLDataTypeCode.kDLBfloat, 16): BFloat16,
        (DLDataTypeCode.kDLBool, 8): Bool,
    }
    dtype = dtype_mapping.get((dtype_code, dtype_bits))
    if dtype is None:
        raise RuntimeTensorError(
            f"unsupported DLPack dtype code={dtype_code} bits={dtype_bits} lanes={lanes}"
        )

    if element_type is not None:
        # Read the widths directly rather than through a defaulted getattr: a
        # missing `width` would make both sides compare equal and turn the check
        # below into a no-op, which is exactly when it is most needed.
        try:
            exported_width = int(dtype.width)
            override_width = int(element_type.width)
        except (AttributeError, TypeError) as exc:
            raise RuntimeTensorError(
                "from_dlpack element_type override requires both the exported dtype and the "
                f"override to declare a storage width; got {dtype!r} and {element_type!r}"
            ) from exc
        # Narrowing to a sub-byte element that tiles the exported width is the
        # one width change that is well defined: the host has no 4-bit dtype, so
        # a packed fp4 buffer can only be exported as bytes, and the override is
        # what says how many elements each byte holds. origin_shape carries the
        # true element count. Anything else -- f32 read as i32, say -- is a
        # reinterpretation the caller almost never means.
        packs_into_exported = (
            override_width < exported_width and exported_width % override_width == 0
        )
        if override_width != exported_width and not packs_into_exported:
            raise RuntimeTensorError(
                f"from_dlpack element_type override {element_type.__name__} is "
                f"{override_width}-bit but the exported buffer is {exported_width}-bit; "
                "an override may narrow to a sub-byte element that tiles the "
                "exported width, but must not otherwise change it"
            )
        dtype = element_type

    if layout_tag is None:
        raise RuntimeTensorError(
            "from_dlpack requires layout_tag (e.g. tla.arch.RowMajor or "
            "tla.arch.ColumnMajor)"
        )
    resolved_layout = layout_tag
    dtype_token = str(getattr(dtype, "dtype", "")).strip().lower()
    layout_token = _resolve_arch_layout_tag(resolved_layout, for_op="from_dlpack")
    if layout_token in _MX_SCALE_GM_LAYOUT_TOKENS and dtype_token != "f8e8m0":
        # The GM MxScale tags describe an e8m0 shared-exponent block; carrying
        # any other element type under them would be a silent reinterpretation
        # downstream, where the bc layer hands the buffer to e8m0-typed copies.
        raise RuntimeTensorError(
            f"from_dlpack layout_tag={layout_token!r} requires element_type "
            f"tla.Float8E8M0, got {dtype_token!r}"
        )
    # A flat one-dimensional layout is unit-stride by definition, but the DLPack
    # strides are the only place a strided view shows up: ``make_tensor`` never
    # sees them, and the remap below would otherwise replace the real stride
    # with 1. Reject before any metadata is derived, and judge by the rank the
    # tile is projected to -- a (1, N) physical slice with a rank-1 origin is
    # flattened just the same.
    if origin_shape is None:
        # A physically contiguous buffer is passed here (stride = (shape[1], 1)).
        # A size-1 extent is exempt: torch is free to leave its stride at 1,
        # which would otherwise make a transposed one-row buffer look non-compact.

        compact = (
            len(phys_shape) == 2
            and (phys_shape[1] == 1 or phys_strides[1] == 1)
            and (phys_shape[0] == 1 or phys_strides[0] == phys_shape[1])
        )
        if (
            len(phys_shape) == 2
            and layout_token in ("RowMajor", "ColumnMajor")
            and not compact
        ):
            torch_hint = (
                "tensor.contiguous()"
                if layout_token == "RowMajor"
                else "tensor.permute(1, 0).contiguous()"
            )
            raise RuntimeTensorError(
                f"from_dlpack layout_tag={layout_token!r} requires a buffer prepared as "
                f"{torch_hint}; got shape={phys_shape}, strides={phys_strides}"
            )
        logical_origin = (
            (phys_shape[1], phys_shape[0])
            if len(phys_shape) == 2 and layout_token == "ColumnMajor"
            else phys_shape
        )
    else:
        if isinstance(origin_shape, tuple):
            logical_origin = origin_shape
        elif isinstance(origin_shape, int):
            logical_origin = (origin_shape,)
        else:
            raise TypeError(
                "from_dlpack origin_shape=... must be an int tree / tuple "
                "(Kernel tla.make_shape is not a Host API)"
            )
    # Now that the logical origin fixes the projected rank, check the buffer.
    _require_flat_1d_binding(
        layout_token,
        phys_shape,
        phys_strides,
        projects_to_rank_one=(
            isinstance(logical_origin, tuple)
            and len(logical_origin) == 1
            and not isinstance(logical_origin[0], (tuple, list))
        ),
    )
    # A stride counts elements between physically adjacent rows, and rows sit
    # at byte addresses: for a sub-byte element the leading linear stride must
    # land on a whole byte, i.e. round up to whole element slots. A packed fp4
    # row of ceil(cols/2) bytes carries round_up(cols, 2) slots -- without the
    # rounding an odd cols yields a half-byte stride, and every GM tile whose
    # base sits past row 0 is addressed one byte off per row. Byte-width (and
    # wider) elements keep the exact stride: round_up(cols, 1) is cols.
    element_bits = int(getattr(dtype, "width", 8))
    linear_stride_alignment_bytes = 1 if 0 < element_bits < 8 else None
    remapped = _remap_tensor_like_prefix_fields_for_layout_trees(
        logical_origin,
        dtype_token,
        layout_token,
        linear_stride_alignment_bytes=linear_stride_alignment_bytes,
    )
    if remapped is None:
        raise RuntimeTensorError(
            f"from_dlpack cannot derive layout metadata for origin_shape={logical_origin!r} "
            f"layout={resolved_layout!r}"
        )
    tensor = _Tensor(
        remapped.shape,
        dtype,
        origin_shape=remapped.origin_shape,
        coord=remapped.coord,
        stride=remapped.stride,
        layout_tag=resolved_layout,
        data_ptr=data_ptr,
        flat_shape=remapped.flat_shape,
        flat_stride=remapped.flat_stride,
    )
    if assumed_align is not None:
        tensor._assumed_align = int(assumed_align)
    tensor._external_binding = not host_resident

    # Take ownership of the buffer the tensor is about to point at. Everything
    # that can reject the capsule has run by now, so a failure past this point
    # cannot leave a consumed-but-unowned DLManagedTensor behind.
    _consume_capsule(dlpack_data)
    tensor._dlpack_release = weakref.finalize(
        tensor, _release_managed_tensor, int(parsed["managed_ptr"])
    )
    # Also retain the producer object: its deleter may be a no-op while the
    # object itself owns the allocation, and a temporary source (e.g.
    # ``x.contiguous().to(device)``) would otherwise be freed on return and the
    # kernel would silently read reused memory.
    tensor._dlpack_source = tensor_dlpack
    return tensor


def make_fake_tensor(
    dtype: Any,
    shape: Any,
    stride: Any,
    *,
    layout_tag: Any | None = None,
    addrspace: Any = AddressSpace.gm,
    origin_shape: Iterable[Any] | None = None,
    coord: Iterable[Any] | None = None,
    assumed_align: int | None = None,
) -> _Tensor:
    """Directory: Host Tensor / Binding
    Description:
        Build a metadata-only Host tensor with no device buffer (`data_ptr == 0`).
        Use this as a `tla.compile` type sample when no NPU is needed. Bind real
        buffers with `from_dlpack`.

        Parameters:
        - *`dtype`*: Element type such as `tla.Float16` / `tla.Float32`. Required.
        - *`shape`* (`int | tuple`): Logical shape tree (nested tuples for zN
          physical layouts). Required.
        - *`stride`* (`int | tuple`): Stride tree; structure must match `shape`.
          Required.
        - *`layout_tag`*: `tla.arch` tag. Optional; default `tla.arch.RowMajor`.
        - *`addrspace`*: Address space. Optional; default `AddressSpace.gm`.
        - *`origin_shape`* (`int | tuple | None`): Logical origin. Optional;
          defaults to `shape`.
        - *`coord`* (`int | tuple | None`): Coordinate tree. Optional; derived
          from the layout when omitted (typically zeros).
        - *`assumed_align`* (`int | None`): Reserved; currently unused.

        Constraints:
        - `shape` / `stride` / `origin_shape` / `coord` are Python int trees,
          not Kernel `tla.make_shape` / `tla.make_stride` / `tla.make_coord`.
        - Always unbound; cannot be launched until replaced by `from_dlpack`.
        - Explicit `shape` / `stride` are kept as given (no layout remap).

        Example:
        ```python
        fa = make_fake_tensor(tla.Float16, (128, 64), (64, 1))
        fzn = make_fake_tensor(
            tla.Float16,
            ((16, 2), (16, 4)),
            ((16, 256), (1, 512)),
            layout_tag=tla.arch.zN,
            origin_shape=(32, 64),
        )
        ```
    """
    from ..runtime import _eager_capture

    if origin_shape is None:
        origin_shape = shape

    with _eager_capture():
        tensor = _Tensor(
            shape,
            dtype,
            addrspace=addrspace,
            data_ptr=0,
            origin_shape=origin_shape,
            coord=coord,
            stride=stride,
            layout_tag=layout_tag,
        )
    if assumed_align is not None:
        tensor._assumed_align = int(assumed_align)
    tensor.data_ptr = 0
    tensor._external_binding = False
    return tensor


__all__ = [
    "DlpackBridgeError",
    "export_dlpack_capsule",
    "from_dlpack",
    "make_fake_tensor",
]
