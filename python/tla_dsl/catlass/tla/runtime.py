"""Runtime DLPack tensor binding."""

from __future__ import annotations

import ctypes
import weakref
from collections import namedtuple
from math import prod
from typing import Any, Iterable, Iterator

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

from ..address_space import AddressSpace
from ..base_dsl.runtime.dlpack_types import (
    ASCEND_DEVICE_TYPES,
    DLDataTypeCode,
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
    dtype_size_bits,
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


def _zeros_index_tree(tree: Any) -> Any:
    """Coord default: same nesting as ``tree``, every leaf 0."""
    if isinstance(tree, tuple):
        return tuple(_zeros_index_tree(x) for x in tree)
    return 0


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


def _require_packed_nz_buffer(
    layout_token: str,
    phys_shape: tuple[int, ...],
    phys_strides: tuple[int, ...],
    logical_origin: tuple[int | None, ...],
    *,
    expected_storage_elems: int | None,
) -> None:
    """Reject a buffer that an NZFamily Host bind cannot safely describe.

    Mirrors the 2-D ``RowMajor`` / ``ColumnMajor`` compact check: DLPack still
    carries an ND view, so the binding must prove the storage matches the
    fractal claim before remap metadata is trusted.

    Host prepack (``host_nd2nz``) yields a flat, contiguous run whose length
    is the padded fractal numel for ``origin_shape``. A producer may also expose
    the four physical modes directly, which are validated separately.
    """
    if len(phys_shape) != 1:
        raise RuntimeTensorError(
            f"from_dlpack layout_tag={layout_token!r} expects a flat 1-D "
            f"prepacked buffer (e.g. host_nd2nz) or a matching four-dimensional "
            f"DLPack view; got shape={phys_shape} for origin_shape={logical_origin!r}"
        )
    if not _is_contiguous_run(phys_shape, phys_strides):
        raise RuntimeTensorError(
            f"from_dlpack layout_tag={layout_token!r} requires a contiguous "
            f"prepacked buffer; got shape={phys_shape} strides={phys_strides}. "
            f"Take a contiguous copy of the packed run first "
            f"(e.g. tensor.contiguous())."
        )
    if expected_storage_elems is not None and phys_shape[0] < expected_storage_elems:
        raise RuntimeTensorError(
            f"from_dlpack layout_tag={layout_token!r} for "
            f"origin_shape={logical_origin!r} requires a contiguous prepacked "
            f"buffer of at least {expected_storage_elems} elements "
            f"(host_nd2nz / equivalent); got shape={phys_shape} "
            f"({phys_shape[0]} elements)"
        )


def _require_mx_scale_buffer(
    layout_token: str,
    phys_shape: tuple[int, ...],
    phys_strides: tuple[int, ...],
    expected_storage_elems: int | None,
) -> None:
    # Prepared scale buffers can be transposed views of dense storage. Check
    # contiguity in address order because the remap supplies its own strides.
    storage_elems = prod(phys_shape)
    if storage_elems:
        modes = sorted(zip(phys_strides, phys_shape), reverse=True)
        strides = tuple(stride for stride, _ in modes)
        shape = tuple(extent for _, extent in modes)
        if not _is_contiguous_run(shape, strides):
            raise RuntimeTensorError(
                f"from_dlpack layout_tag={layout_token!r} requires dense MxScale "
                f"storage; got shape={phys_shape} strides={phys_strides}"
            )
    if expected_storage_elems is not None and storage_elems < expected_storage_elems:
        raise RuntimeTensorError(
            f"from_dlpack layout_tag={layout_token!r} requires at least "
            f"{expected_storage_elems} storage elements for the MxScale "
            f"descriptor; got {storage_elems}"
        )


def _require_nz_origin(origin: Any, *, for_op: str, allow_dynamic: bool) -> None:
    if not isinstance(origin, tuple) or len(origin) != 2:
        valid = False
    else:
        valid = all(
            (allow_dynamic and extent is None) or (type(extent) is int and extent >= 0)
            for extent in origin
        )
    if not valid:
        expected = (
            "non-negative integers or None" if allow_dynamic else "concrete integers"
        )
        raise RuntimeTensorError(
            f"{for_op} requires a flat NZFamily origin_shape=(M, N) with "
            f"{expected}; got {origin!r}"
        )


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


# ``restype`` / ``argtypes`` describe the C signature, which never changes, so
# they are declared once here rather than re-assigned on every capsule the
# binding path touches: a model binds one capsule per operand per launch, and
# that assignment is pure per-call overhead (see _parse_capsule).
#
# Each entry point is taken by subscript rather than by attribute: attribute
# access on ``ctypes.pythonapi`` caches one function object per name and hands
# the same object to every library in the process, so a signature set on it is
# shared with any other DLPack consumer and theirs with us. A subscript builds a
# private object, which keeps the signatures below ours alone.
_PyCapsule_IsValid = ctypes.pythonapi["PyCapsule_IsValid"]
_PyCapsule_IsValid.restype = ctypes.c_int
_PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

_PyCapsule_GetPointer = ctypes.pythonapi["PyCapsule_GetPointer"]
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

_PyCapsule_SetName = ctypes.pythonapi["PyCapsule_SetName"]
_PyCapsule_SetName.restype = ctypes.c_int
_PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]

_DLTENSOR_NAME = b"dltensor"


def _consume_capsule(capsule: Any) -> None:
    """Take ownership of a ``dltensor`` capsule by renaming it to ``used_dltensor``.

    A ``dltensor`` capsule's destructor calls ``DLManagedTensor.deleter`` when the
    capsule dies unconsumed, which would release the buffer as soon as the local
    reference goes out of scope. Renaming is the DLPack handshake that says the
    consumer now owns the ``DLManagedTensor`` and will call the deleter itself
    (see :func:`_release_managed_tensor`); producers key their capsule destructor
    on the old name, so it becomes a no-op.
    """
    # The name is borrowed by the capsule, so it must outlive it: keep the bytes
    # object alive in a module-level constant rather than a temporary.
    if _PyCapsule_SetName(capsule, _USED_DLTENSOR_NAME) != 0:
        raise DlpackBridgeError("failed to rename DLPack capsule to 'used_dltensor'")


# PyCapsule_SetName borrows the pointer; this must never be garbage-collected.
_USED_DLTENSOR_NAME = b"used_dltensor"


def _release_managed_tensor(managed_ptr: int) -> None:
    """Call ``DLManagedTensor.deleter`` once, releasing the producer's allocation.

    Registered through :func:`weakref.finalize` on the owning tensor, so it must
    not close over that tensor (a strong reference would keep it alive forever).
    """
    managed = DLManagedTensor.from_address(managed_ptr)
    deleter = managed.deleter
    # A NULL deleter means the producer has nothing to release (DLPack allows it).
    if deleter:
        deleter(ctypes.byref(managed))


class _Tensor(TensorABC):
    """TLA runtime host tensor with compile metadata and optional DLPack binding.

    Not a Host public constructor. Use :func:`make_fake_tensor` (unbound sample) or
    :func:`from_dlpack` (NPU buffer). ``tla.Tensor`` is the shared ABC / annotation.
    """

    @staticmethod
    def _parse_capsule(capsule: Any) -> dict[str, Any]:
        """Read a ``dltensor`` capsule's fields; does not consume it (see :func:`_consume_capsule`)."""
        # Check the name before reading it: PyCapsule_GetPointer on a renamed
        # capsule returns NULL *and* sets the error indicator, which would
        # otherwise surface later at an unrelated call site.
        if not _PyCapsule_IsValid(capsule, _DLTENSOR_NAME):
            raise DlpackBridgeError(
                "not a fresh 'dltensor' capsule; a DLPack capsule can only be "
                "consumed once (an already-consumed capsule is renamed to "
                "'used_dltensor')"
            )
        # Capsule holds DLManagedTensor*; dl_tensor is its first member (dlpack.h).
        managed_ptr = _PyCapsule_GetPointer(capsule, _DLTENSOR_NAME)
        # from_address reads the struct in place; casting through a pointer type
        # would allocate two ctypes objects per capsule for the same fields.
        dl = DLManagedTensor.from_address(managed_ptr).dl_tensor
        ndim = int(dl.ndim)
        if ndim < 0 or (ndim and not dl.shape):
            raise DlpackBridgeError("DLPack tensor has invalid rank or missing shape")
        # Slicing a ctypes int64 pointer reads the whole run in one call, where
        # an index-by-index generator pays Python's indexing cost per mode.
        shape = tuple(dl.shape[:ndim])
        if dl.strides:
            strides = tuple(dl.strides[:ndim])
        else:
            # Use C-contiguous strides when the exporter leaves them unset.
            compact = [1] * ndim
            for axis in range(ndim - 2, -1, -1):
                compact[axis] = compact[axis + 1] * max(shape[axis + 1], 1)
            strides = tuple(compact)
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
        self._cache_key_memo: tuple[Any, ...] | None = None
        self._memref_launch_metadata: tuple[Any, Any, tuple[int, ...]] | None = None
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

    @classmethod
    def _from_host_binding(
        cls,
        *,
        shape_tree: Any,
        stride_tree: Any,
        origin_tree: Any,
        coord_tree: Any,
        dtype: Any,
        addrspace: Any,
        data_ptr: int,
        layout_tag: Any,
        assumed_align: int | None = None,
        flat_shape: tuple[int, ...] | None = None,
        flat_stride: tuple[int, ...] | None = None,
    ) -> "_Tensor":
        """Fast Host bind: trust already-normalized make_ptr/make_layout trees.

        Skips ``__init__`` remap / re-validation used by the general constructor.
        ``flat_shape`` / ``flat_stride`` are the flattened int leaves of the two
        trees; pass them when a caller already holds them (the layout remap
        returns them), and they are flattened here when it does not.
        """
        self = object.__new__(cls)
        self._external_binding = data_ptr != 0
        self._assumed_align = None if assumed_align is None else int(assumed_align)
        self._dlpack_source = None
        self._dlpack_release = None
        self._dynamic_shape_tree = None
        self._dynamic_origin_shape_tree = None
        self._dynamic_stride_tree = None
        self.data_ptr = int(data_ptr)
        self._dtype = _coerce_host_tensor_dtype(dtype)
        self.addrspace = _coerce_host_tensor_addrspace(addrspace)
        self.layout_tag = layout_tag
        self._shape = shape_tree
        self._stride = stride_tree
        self.origin_shape = origin_tree
        self.coord = coord_tree
        self._flat_shape = (
            tuple(_flatten_int_leaves_tree(shape_tree))
            if flat_shape is None
            else flat_shape
        )
        self._flat_stride = (
            tuple(_flatten_int_leaves_tree(stride_tree))
            if flat_stride is None
            else flat_stride
        )
        self._cache_key_memo: tuple[Any, ...] | None = None
        self._memref_launch_metadata = None
        return self

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
        """Build unified schema-v4 GM launch values in canonical ABI order.

        Fixed 13 slots: ``allocated/aligned/offset`` + ``size0..3`` +
        ``stride0..3`` + ``originShape0/1``. Linear layouts pad unused size /
        stride leaves with ``1``; NZFamily uses all four physical leaves and a
        two-leaf logical origin ``(M, N)``.
        """
        self._require_bound()
        data_ptr = int(self.data_ptr)
        # Dynamic markers only change the compile-time type. Reuse the
        # concrete fields until the origin or layout is replaced.
        origin = self.origin_shape
        layout_tag = self.layout_tag
        cached = self._memref_launch_metadata
        if cached is not None and cached[0] is origin and cached[1] == layout_tag:
            fields = cached[2]
        else:
            fields = self._build_memref_metadata_fields(origin, layout_tag)
            self._memref_launch_metadata = (origin, layout_tag, fields)
        return (data_ptr, data_ptr, 0, *fields)

    def _build_memref_metadata_fields(
        self, origin: tuple[Any, ...], layout_tag: str
    ) -> tuple[int, ...]:
        shape = self._flat_shape or ()
        if not shape:
            raise RuntimeTensorError(
                "build_memref_launch_fields requires a concrete DLPack shape"
            )
        rank = len(shape)
        if rank > 4:
            raise RuntimeTensorError(
                "build_memref_launch_fields supports at most 4 physical "
                f"shape/stride leaves (schema v4 size0..size3); got {rank}. "
                "NZFamily packs a nested 2x2 into exactly 4 leaves; logical "
                "higher-rank views are not encoded in the 13-slot GM ABI"
            )
        strides = self._flat_stride
        if strides is None or len(strides) != rank:
            raise RuntimeTensorError(
                "build_memref_launch_fields requires concrete strides with "
                f"the same leaf count as shape ({rank}); got "
                f"{0 if strides is None else len(strides)}"
            )
        if any(type(leaf) is not int for leaf in (*shape, *strides)):
            raise RuntimeTensorError(
                "build_memref_launch_fields requires concrete integer shape/stride leaves"
            )
        # Rank-4 alone is not NZFamily — linear views can also pad to 4 leaves.
        from ..core_api import _NZ_FAMILY_LAYOUT_TOKENS

        if layout_tag in _NZ_FAMILY_LAYOUT_TOKENS and rank == 4:
            _require_nz_origin(
                origin, for_op="build_memref_launch_fields", allow_dynamic=False
            )
        elif not origin or any(
            type(leaf) is not int or leaf < 0 for leaf in origin[:2]
        ):
            raise RuntimeTensorError(
                "build_memref_launch_fields requires concrete integer origin_shape "
                f"slots; got {origin!r}"
            )

        return (
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

    @property
    def __cache_key__(self) -> tuple[Any, ...]:
        """Hashable description of what a kernel specializes on.

        Covers the element type, address space, layout token, pointer alignment
        and the shape / stride / origin / coord trees. Dynamic modes contribute
        ``None`` rather than the bound extent, so equal specializations share a
        key. Memoized; the memo is invalidated when the binding is re-marked.
        """
        cached = getattr(self, "_cache_key_memo", None)
        if cached is not None:
            guard = cached[0]
            if (
                guard[0] is self._shape
                and guard[1] is self._stride
                and guard[2] is self.origin_shape
                and guard[3] is self.coord
                and guard[4] is self._dynamic_shape_tree
                and guard[5] is self._dynamic_stride_tree
                and guard[6] is self._dynamic_origin_shape_tree
                and guard[7] is self._dtype
                and guard[8] == self.addrspace
                and guard[9] == self.layout_tag
                and guard[10] == self._assumed_align
            ):
                return cached[1]
        guard = (
            self._shape,
            self._stride,
            self.origin_shape,
            self.coord,
            self._dynamic_shape_tree,
            self._dynamic_stride_tree,
            self._dynamic_origin_shape_tree,
            self._dtype,
            self.addrspace,
            self.layout_tag,
            self._assumed_align,
        )
        key = (
            str(self._dtype),
            str(self.addrspace),
            str(self.layout_tag),
            self._ptr_alignment(),
            self._layout_shape_components(),
            self._layout_stride_components(),
            self._layout_origin_shape_components(),
            self.coord,
        )
        self._cache_key_memo = (guard, key)
        return key

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


_BindMetadata = namedtuple("_BindMetadata", "dtype layout_token remapped")

# Keyed on everything the derivation reads, and on nothing else: the buffer's
# address is deliberately absent, so an entry describes a *shape* rather than a
# buffer. A model therefore fills this with one entry per distinct operand
# geometry -- it cannot grow with the number of buffers bound, it retains no
# buffer and pins nothing in device memory, and an entry cannot go stale because
# the metadata it holds is a pure function of the key.
_BIND_METADATA_MEMO: dict[tuple[Any, ...], _BindMetadata] = {}

# A model's distinct operand geometries number in the dozens; a cap three orders
# of magnitude above that only bounds a caller that binds unique shapes forever,
# and clearing is the honest response to that -- an LRU would quietly keep
# paying the eviction cost to serve a workload the memo cannot help.
_BIND_METADATA_MEMO_LIMIT = 4096

# Built on first use: ``base_dsl.typing`` is not importable at module scope here.
_DLPACK_DTYPE_MAP: dict[tuple[int, int], type] | None = None


def _dlpack_dtype_map() -> dict[tuple[int, int], type]:
    """The DLPack ``(code, bits)`` to TLA element-type table, built once.

    DLPack bool is byte-sized storage (kDLBool, bits=8), matching torch.bool /
    NumPy bool_. It maps to TLA Bool (MLIR i1); tensor element size is still 1
    byte.
    """
    global _DLPACK_DTYPE_MAP
    if _DLPACK_DTYPE_MAP is None:
        from ..base_dsl.typing import (
            BFloat16,
            Bool,
            Float16,
            Float32,
            Int8,
            Int16,
            Int32,
            Int64,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
        )

        _DLPACK_DTYPE_MAP = {
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
    return _DLPACK_DTYPE_MAP


def _canonical_origin_shape(origin_shape: Any) -> Any:
    """Normalize a ``from_dlpack origin_shape=...`` to a canonical index tree.

    The memo is keyed on the origin, and Python compares ``True`` equal to ``1``
    and ``8.0`` equal to ``8``, so a key built from the caller's own object would
    let one spelling of an extent be served the metadata derived for another.
    Settling the leaves here, ahead of the key, makes the value the memo stores
    and the value it is looked up by the same validated tree.

    An omitted origin (``None`` rather than a tree) is the caller asking for one
    to be derived from the buffer, and is passed through as such.
    """
    if origin_shape is None:
        return None
    if isinstance(origin_shape, tuple):
        return tuple(_canonical_origin_shape_leaf(x) for x in origin_shape)
    return (_canonical_origin_shape_leaf(origin_shape),)


def _canonical_origin_shape_leaf(value: Any) -> Any:
    """One ``origin_shape`` leaf: a nested tree, a non-negative extent, or ``None``.

    These are the leaves a Tla index tree is made of, so the rule is
    :func:`catlass.types._normalize_index_tree_leaf`'s: ``None`` stands for an
    extent known only at run time, a static extent is a non-negative ``int``,
    and ``bool`` is not an index at all. Applying it here rather than leaving it
    to the tensor means a leaf the tree would reject is never keyed on, and the
    diagnostic names ``from_dlpack``'s own argument.
    """
    if isinstance(value, tuple):
        return tuple(_canonical_origin_shape_leaf(x) for x in value)
    if value is None:
        return None
    # bool is an int subclass, so it must be excluded before the int test.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "from_dlpack origin_shape=... must be an int tree / tuple "
            "(Kernel tla.make_shape is not a Host API)"
        )
    if value < 0:
        raise ValueError(
            "from_dlpack origin_shape=... leaves must be non-negative integers "
            f"or None for a dynamic extent; got {value}"
        )
    return int(value)


def _packed_nz_view_matches(
    phys_shape: tuple[int, ...],
    phys_strides: tuple[int, ...],
    remapped: Any,
) -> bool:
    """Check the layout exported by DLPack in TLA leaf order."""
    # A singleton stride never contributes to the address. Keep the canonical
    # TLA stride for the ABI, including when a producer reports 1 for that mode.
    return phys_shape == remapped.flat_shape and all(
        extent <= 1 or step == expected_stride
        for extent, step, expected_stride in zip(
            phys_shape, phys_strides, remapped.flat_stride
        )
    )


def _derive_bind_metadata(
    layout_tag: Any,
    phys_shape: tuple[int, ...],
    phys_strides: tuple[int, ...],
    dtype_code: int,
    dtype_bits: int,
    element_type: type | None,
    origin_shape: Any,
    *,
    host_resident: bool,
) -> _BindMetadata:
    """Validate a DLPack buffer's geometry and derive the tile metadata for it.

    Reads only its arguments, so :func:`from_dlpack` can memoise the result: both
    the checks and the derivation are pure, and every rejection below depends on
    the same geometry the memo is keyed on.

    For packed NZ layouts, the exporter can supply a flat prepacked run or a
    four-dimensional view in TLA leaf order.
    """
    from ..core_api import (
        _MX_SCALE_GM_LAYOUT_TOKENS,
        _NZ_FAMILY_LAYOUT_TOKENS,
        _remap_tensor_like_prefix_fields_for_layout_trees,
        _resolve_arch_layout_tag,
    )

    if any(extent < 0 for extent in phys_shape):
        raise RuntimeTensorError(
            f"from_dlpack requires non-negative DLPack extents; got {phys_shape}"
        )
    dtype = _dlpack_dtype_map().get((dtype_code, dtype_bits))
    if dtype is None:
        raise RuntimeTensorError(
            f"unsupported DLPack dtype code={dtype_code} bits={dtype_bits} lanes=1"
        )

    element_bits = dtype_bits
    if element_type is not None:
        try:
            declared_bits = int(element_type.width)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeTensorError(
                "from_dlpack element_type override must declare a width; "
                f"got {element_type!r}"
            ) from exc
        # Bool has an i1 IR type but occupies one byte. Logical FP4 elements
        # are packed two-per-byte in the exported integer buffer.
        override_dtype = str(getattr(element_type, "dtype", "")).strip().lower()
        element_bits = dtype_size_bits(override_dtype) or declared_bits
        packed_sub_byte = (
            dtype_bits == 8
            and override_dtype in {"f4e1m2", "f4e2m1"}
            and dtype_code in (DLDataTypeCode.kDLInt, DLDataTypeCode.kDLUInt)
        )
        if element_bits != dtype_bits and not packed_sub_byte:
            raise RuntimeTensorError(
                f"from_dlpack element_type override {element_type.__name__} is "
                f"{element_bits}-bit but the exported buffer is {dtype_bits}-bit; "
                "use matching storage widths or packed 4-bit types in integer bytes"
            )
        dtype = element_type

    dtype_token = str(getattr(dtype, "dtype", "")).strip().lower()
    padded_origin = None
    if len(phys_shape) == 4:
        m0, m1, n0, n1 = phys_shape
        padded_origin = (m0 * m1, n0 * n1)
    layout_token = _resolve_arch_layout_tag(layout_tag, for_op="from_dlpack")
    is_nz_family = layout_token in _NZ_FAMILY_LAYOUT_TOKENS
    is_mx_scale = layout_token in _MX_SCALE_GM_LAYOUT_TOKENS
    is_packed_nz_view = is_nz_family and not is_mx_scale and padded_origin is not None
    if is_mx_scale and dtype_token != "f8e8m0":
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
    if is_nz_family:
        if origin_shape is not None:
            logical_origin = origin_shape
        elif is_packed_nz_view:
            logical_origin = padded_origin
        elif is_mx_scale and len(phys_shape) == 2:
            logical_origin = phys_shape
        else:
            raise RuntimeTensorError(
                f"from_dlpack layout_tag={layout_token!r} requires "
                "origin_shape=(M, N) when the DLPack shape cannot supply it"
            )
        _require_nz_origin(
            logical_origin, for_op="from_dlpack", allow_dynamic=host_resident
        )
    elif origin_shape is None:
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
        # Already a canonical int tuple tree: _canonical_origin_shape normalizes
        # and validates it before the memo is keyed on it.
        logical_origin = origin_shape
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
    linear_stride_alignment_bytes = 1 if 0 < element_bits < 8 else None
    remapped = _remap_tensor_like_prefix_fields_for_layout_trees(
        logical_origin,
        dtype_token,
        layout_token,
        linear_stride_alignment_bytes=linear_stride_alignment_bytes,
    )
    view_geometry = remapped
    if is_packed_nz_view and None in logical_origin:
        # Validate the export with concrete sizes while retaining the dynamic
        # geometry for the CPU-only compile sample.
        concrete_origin = tuple(
            padded if extent is None else extent
            for extent, padded in zip(logical_origin, padded_origin)
        )
        view_geometry = _remap_tensor_like_prefix_fields_for_layout_trees(
            concrete_origin, dtype_token, layout_token
        )
    if remapped is None or view_geometry is None:
        raise RuntimeTensorError(
            f"from_dlpack cannot derive layout metadata for "
            f"origin_shape={logical_origin!r} layout={layout_tag!r}"
        )
    if is_packed_nz_view:
        if element_bits != dtype_bits:
            raise RuntimeTensorError(
                "from_dlpack four-dimensional packed NZ views require export and "
                "TLA storage widths to match. Use a flat packed buffer for "
                "sub-byte elements exported as bytes."
            )
        if not _packed_nz_view_matches(phys_shape, phys_strides, view_geometry):
            raise RuntimeTensorError(
                "from_dlpack four-dimensional DLPack shape/stride do not match "
                f"layout {layout_token!r} for origin_shape={logical_origin!r}: "
                f"expected shape={view_geometry.flat_shape!r} stride={view_geometry.flat_stride!r}, "
                f"got shape={phys_shape!r} stride={phys_strides!r}"
            )
    elif is_nz_family:
        expected_storage = None
        if None not in logical_origin:
            extents, strides = remapped.flat_shape, remapped.flat_stride
            # The last address follows descriptor strides, including MxScale
            # pair padding. DLPack extents count exported storage elements.
            span = (
                1 + sum((extent - 1) * step for extent, step in zip(extents, strides))
                if all(extents)
                else 0
            )
            expected_storage = (span * element_bits + dtype_bits - 1) // dtype_bits
        if is_mx_scale:
            _require_mx_scale_buffer(
                layout_token, phys_shape, phys_strides, expected_storage
            )
        else:
            _require_packed_nz_buffer(
                layout_token,
                phys_shape,
                phys_strides,
                logical_origin,
                expected_storage_elems=expected_storage,
            )
    return _BindMetadata(dtype, layout_token, remapped)


def _bind_metadata(
    layout_tag: Any,
    phys_shape: tuple[int, ...],
    phys_strides: tuple[int, ...],
    dtype_code: int,
    dtype_bits: int,
    element_type: type | None,
    origin_shape: Any,
    *,
    host_resident: bool,
) -> _BindMetadata:
    """:func:`_derive_bind_metadata` with its result memoised on its arguments.

    A rejected geometry is never stored, so a caller that keeps binding a buffer
    the layout cannot describe keeps being told so; only the derivation is made
    cheap the second time, never a check.
    """
    origin_shape = _canonical_origin_shape(origin_shape)
    key = (
        layout_tag,
        phys_shape,
        phys_strides,
        dtype_code,
        dtype_bits,
        element_type,
        origin_shape,
        host_resident,
    )
    try:
        hit = _BIND_METADATA_MEMO.get(key)
    except TypeError:
        # A layout tag that is not hashable cannot be keyed; derive it every
        # time rather than refuse it here, since the derivation is what decides
        # whether it is legal at all.
        return _derive_bind_metadata(
            layout_tag,
            phys_shape,
            phys_strides,
            dtype_code,
            dtype_bits,
            element_type,
            origin_shape,
            host_resident=host_resident,
        )
    if hit is not None:
        return hit
    derived = _derive_bind_metadata(
        layout_tag,
        phys_shape,
        phys_strides,
        dtype_code,
        dtype_bits,
        element_type,
        origin_shape,
        host_resident=host_resident,
    )
    if len(_BIND_METADATA_MEMO) >= _BIND_METADATA_MEMO_LIMIT:
        _BIND_METADATA_MEMO.clear()
    _BIND_METADATA_MEMO[key] = derived
    return derived


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
        Bind a DLPack buffer to a TLA Host tensor. NPU buffers are shared without
        copying, and CPU buffers supply metadata for compilation.

        Parameters:
        - *`tensor_dlpack`* (`object`): Object implementing `__dlpack__()`.
          An Ascend/NPU buffer (e.g. `torch_npu`) gives a launchable tensor. A
          host (CPU) buffer, such as a torch tensor on CPU, is accepted too,
          but only its metadata is taken: the result has `data_ptr == 0`, so
          it can be compiled against and not launched. Required.
        - *`layout_tag`* (`tla.arch.*`): Required layout tag such as
          `tla.arch.RowMajor`, `tla.arch.ColumnMajor`, `tla.arch.zN`, or
          `tla.arch.nZ`. The four-dimensional packed NZ view's shape and
          strides are checked against the specified layout.
        - *`origin_shape`* (`tuple | int | None`): Logical origin as a Python int
          tree. Optional for linear layouts (derived from the DLPack physical
          shape and `layout_tag` when omitted). Flat packed NZ inputs require
          `(M, N)`. Four-dimensional packed NZ views default to the padded
          dimensions. Pass `(M, N)` to retain the logical size before padding.
          GM MxScale infers this pair from a 2-D input when omitted.
          CPU-only samples can use `None` for a dynamic extent. Four-dimensional
          views still have their actual export geometry checked.
        - *`assumed_align`* (`int | None`): Reserved; currently unused.
        - *`stream`* (`int | None`): Passed to `__dlpack__(stream=...)`. Default
          `-1` (no stream sync). `None` omits the `stream` argument.
        - *`element_type`* (`type | None`): Optional override for the element
          type inferred from DLPack. Default `None` keeps the DLPack type.
          Use when DLPack cannot express the real type (e.g. fp8): pass
          `tla.Float8E4M3FN` / `Float8E5M2`. Storage widths must match, except
          for packed 4-bit elements (`Float4E1M2` / `Float4E2M1`)
          exported as two elements per integer byte. `Bool` occupies one byte.

        Constraints:
        - A host (CPU) buffer binds metadata only. Use it to compile a kernel
          without a device present; launching with it raises.
        - Ownership follows the DLPack consumer contract: the capsule is consumed
          and its deleter runs when the returned tensor is destroyed. A reference
          to `tensor_dlpack` is also retained, so a temporary source such as
          `from_dlpack(x.contiguous().to(device), ...)` is safe.
        - A capsule is single-use; passing an already-consumed capsule raises
          `RuntimeTensorError`. Call `from_dlpack` again for another binding.
        - If the DLPack export omits strides, assume C-contiguous strides.
        - 2-D `RowMajor` requires `tensor.contiguous()`. 2-D `ColumnMajor`
          requires `tensor.permute(1, 0).contiguous()`. A mismatch raises
          `RuntimeTensorError`. Providing `origin_shape` skips that check.
        - Packed NZ layouts accept a flat, contiguous prepacked buffer (e.g.
          `host_nd2nz`) with at least the padded storage length, or a
          four-dimensional DLPack view in physical leaf order
          `(m0, m1, n0, n1)`. The view's shape and strides must match the
          layout, with singleton strides ignored. Export and TLA storage widths
          must match for four-dimensional views. Use the flat form for FP4
          exported as bytes.
        - GM MxScale accepts dense scale storage spanning the remapped
          descriptor, including any pair padding. Supply `origin_shape` if the
          DLPack dimensions do not describe the logical scale matrix.
        - A rank-1 `origin_shape` under `RowMajor`/`VectorLayout` is bound as a
          flat, unit-stride run. The buffer must then read consecutive elements
          (`torch.arange(64)[::2]` is refused); this holds whatever rank the
          physical slice is spelled in, and `origin_shape` does not skip it.
        - The default layout is static. Call `mark_layout_dynamic()` before
          compiling with a packed NZ tensor as a GM kernel argument. For linear
          layouts, `mark_layout_dynamic()` and `mark_compact_shape_dynamic()`
          make extents dynamic when needed.

        Example:
        ```python
        tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
        ty = from_dlpack(
            y.permute(1, 0).contiguous(),
            layout_tag=tla.arch.ColumnMajor,
        )
        t_zn = from_dlpack(
            zn_buf,
            layout_tag=tla.arch.zN,
            origin_shape=(M, N),
        )
        # zn_f32 is a flat prepacked buffer for a 32-by-64 FP32 matrix.
        zn_view = zn_f32.as_strided(
            size=(16, 2, 8, 8),
            stride=(8, 128, 1, 256),
        )
        t_zn_view = from_dlpack(zn_view, layout_tag=tla.arch.zN)
        t_zn_view.mark_layout_dynamic()
        t_nz = from_dlpack(
            nz_buf,
            layout_tag=tla.arch.nZ,
            origin_shape=(M, N),
        )
        ```
    """
    if layout_tag is None:
        raise RuntimeTensorError(
            "from_dlpack requires layout_tag (e.g. tla.arch.RowMajor or tla.arch.zN)"
        )
    try:
        dlpack_data = export_dlpack_capsule(tensor_dlpack, stream=stream)
    except DlpackBridgeError as exc:
        raise RuntimeTensorError(str(exc)) from exc

    try:
        parsed = _Tensor._parse_capsule(dlpack_data)
    except DlpackBridgeError as exc:
        raise RuntimeTensorError(str(exc)) from exc

    data_ptr = int(parsed["data_ptr"])
    device_type = int(parsed["device_type"])
    # A host buffer describes shape, stride, dtype and layout exactly as well as
    # a device one; the only thing it cannot supply is an address the kernel
    # could be launched against. Bind it metadata-only (``data_ptr == 0``) so a
    # compile-only path can reuse the run path's own tensor construction instead
    # of restating the shapes beside it, and leave launching to _require_bound.
    host_resident = device_type == DLDeviceType.kDLCPU
    if not host_resident and device_type not in ASCEND_DEVICE_TYPES:
        device_id = int(parsed["device_id"])
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

    # Everything from here to the tensor's metadata is a function of the
    # buffer's geometry, not of its address, so it is derived once per distinct
    # geometry and reused: binding the same resident weight for the four
    # hundredth time re-validates nothing and re-derives nothing.
    dtype, layout_token, remapped = _bind_metadata(
        layout_tag,
        parsed["shape"],
        parsed["strides"],
        int(parsed["dtype_code"]),
        int(parsed["dtype_bits"]),
        element_type,
        origin_shape,
        host_resident=host_resident,
    )

    # The trees come from the remap, which is where they are normalized, so the
    # general constructor's re-derivation and re-validation of them would only
    # recompute what is already in hand.
    tensor = _Tensor._from_host_binding(
        shape_tree=remapped.shape,
        stride_tree=remapped.stride,
        origin_tree=remapped.origin_shape,
        coord_tree=remapped.coord,
        dtype=dtype,
        addrspace=AddressSpace.gm,
        data_ptr=data_ptr,
        layout_tag=layout_token,
        assumed_align=assumed_align,
        flat_shape=remapped.flat_shape,
        flat_stride=remapped.flat_stride,
    )
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


def _host_make_tensor(ptr: Any, layout: Any, *, coord: Any | None = None) -> _Tensor:
    """Host bind implementation behind dual-mode ``tla.make_tensor`` (not public)."""
    from ..core_api import (
        _Coord,
        _Layout,
        _NZ_FAMILY_LAYOUT_TOKENS,
        _Pointer,
        _Shape,
        _Stride,
        _infer_padded_origin_tree_from_nz_family_shape,
        _is_flat_pair,
        _is_nz_family_2x2_tree,
        _resolve_arch_layout_tag,
        _validate_static_make_tensor_layout,
    )
    from ..execution_lowering import TlaLoweringError

    if not isinstance(ptr, _Pointer) or ptr.value is not None:
        raise TypeError(
            "make_tensor ptr must come from tla.make_ptr outside a kernel; "
            f"got {type(ptr).__name__}"
        )
    if not isinstance(layout, _Layout) or layout._layout_value is not None:
        raise TypeError(
            "make_tensor layout must come from tla.make_layout outside a kernel; "
            f"got {type(layout).__name__}"
        )
    shape_obj = layout.shape
    stride_obj = layout.stride
    if not isinstance(shape_obj, _Shape) or not isinstance(stride_obj, _Stride):
        raise TypeError("make_tensor layout is missing Host shape/stride")
    origin_obj = layout.origin_shape
    if origin_obj is not None and not isinstance(origin_obj, _Shape):
        raise TypeError("make_tensor layout origin_shape must be a Host shape")

    shape_tree = shape_obj._components
    stride_tree = stride_obj._components
    layout_token = _resolve_arch_layout_tag(layout.layout_tag, for_op="make_tensor")
    is_nz_family = layout_token in _NZ_FAMILY_LAYOUT_TOKENS
    # Some compile samples describe an on-chip NZ tensor as a logical matrix.
    # An explicit physical layout must give the nesting for both trees.
    is_physical_nz = is_nz_family and not (
        _is_flat_pair(shape_tree) and _is_flat_pair(stride_tree)
    )
    if is_physical_nz and not (
        _is_nz_family_2x2_tree(shape_tree) and _is_nz_family_2x2_tree(stride_tree)
    ):
        raise RuntimeTensorError(
            f"make_tensor layout {layout_token!r} requires shape and stride as "
            f"nested pairs ((m0, m1), (n0, n1)); got {shape_tree!r} and {stride_tree!r}"
        )
    if origin_obj is None:
        if is_nz_family:
            origin_tree = _infer_padded_origin_tree_from_nz_family_shape(
                shape_tree, for_op="make_tensor"
            )
        else:
            origin_tree = shape_tree
    else:
        origin_tree = origin_obj._components

    dtype_token = str(getattr(ptr.dtype, "dtype", "")).strip().lower()
    # Host hot path still trusts linear / non-nested metadata (compile samples
    # and noncompact views). Nested NZ 2x2 trees are the explicit-4D path:
    # validate layout traits, but do not require equality with a full-tile
    # remap — cropped fractal tiles keep a smaller ceil leaf than origin.
    if is_physical_nz:
        _require_nz_origin(
            origin_tree, for_op="make_tensor", allow_dynamic=ptr.data_ptr == 0
        )
        try:
            _validate_static_make_tensor_layout(
                shape_tree,
                stride_tree,
                dtype=dtype_token,
                layout_tag=layout_token,
                for_op="make_tensor",
            )
        except TlaLoweringError as exc:
            raise RuntimeTensorError(str(exc)) from exc

    if coord is None:
        coord_tree = _zeros_index_tree(origin_tree)
    else:
        if isinstance(coord, _Coord):
            if coord._coord_value is not None:
                raise TypeError(
                    "make_tensor coord must come from tla.make_coord outside a kernel"
                )
            coord = coord._components
        coord_tree = _as_host_index_tree(coord, what="coord")

    return _Tensor._from_host_binding(
        shape_tree=shape_tree,
        stride_tree=stride_tree,
        origin_tree=origin_tree,
        coord_tree=coord_tree,
        dtype=ptr.dtype,
        addrspace=ptr.addrspace,
        data_ptr=ptr.data_ptr,
        layout_tag=layout_token,
        assumed_align=ptr.assumed_align,
    )


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
        buffers with `tla.make_tensor` (direct pointer) or `from_dlpack`.

        Parameters:
        - *`dtype`*: Element type such as `tla.Float16` / `tla.Float32`. Required.
        - *`shape`* (`int | tuple`): Shape tree (nested tuples for zN
          physical layouts). Required.
        - *`stride`* (`int | tuple | None`): Stride tree with the same structure
          as `shape`. Pass `None` to derive it from the layout.
        - *`layout_tag`*: `tla.arch` tag. Optional; default `tla.arch.RowMajor`.
        - *`addrspace`*: Address space. Optional; default `AddressSpace.gm`.
        - *`origin_shape`* (`int | tuple | None`): Logical origin. Optional;
          defaults to `shape` for linear layouts and to the padded `(M, N)`
          for nested NZFamily layouts.
        - *`coord`* (`int | tuple | None`): Coordinate tree. Optional; derived
          from the layout when omitted (typically zeros).
        - *`assumed_align`* (`int | None`): Reserved; currently unused.

        Constraints:
        - `shape` / `stride` / `origin_shape` / `coord` are Python int trees,
          not Kernel `tla.make_shape` / `tla.make_stride` / `tla.make_coord`.
        - Always unbound; cannot be launched until replaced by a bound Host
          tensor (`tla.make_tensor` / `from_dlpack`).
        - Explicit `shape` / `stride` are kept. Like Host `make_tensor`,
          nested NZFamily layouts are checked against layout traits and use a
          flat logical origin. A trait-valid partial view is allowed.

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
    from ..core_api import arch as _arch
    from ..core_api import make_layout as _make_layout
    from ..core_api import make_ptr as _make_ptr
    from ..core_api import make_shape as _make_shape
    from ..core_api import make_stride as _make_stride
    from ..core_api import (
        _NZ_FAMILY_LAYOUT_TOKENS,
        _infer_padded_origin_tree_from_nz_family_shape,
        _resolve_arch_layout_tag,
    )

    if layout_tag is None:
        layout_tag = _arch.RowMajor

    def _as_components(tree: Any) -> tuple[Any, ...]:
        if isinstance(tree, tuple):
            return tree
        return (tree,)

    shape_tree = _as_host_index_tree(shape, what="shape")
    layout_token = _resolve_arch_layout_tag(layout_tag, for_op="Tensor")
    if origin_shape is None and layout_token in _NZ_FAMILY_LAYOUT_TOKENS:
        origin_shape = _infer_padded_origin_tree_from_nz_family_shape(
            shape_tree, for_op="make_fake_tensor"
        )
    origin_tree = _as_host_index_tree(
        shape if origin_shape is None else origin_shape, what="origin_shape"
    )
    resolved_stride = stride
    resolved_coord = coord
    # Match historical ``_Tensor(..., stride=None, coord=None)``: derive from
    # the same layout remap helper, then feed concrete trees into Host make_*.
    if resolved_stride is None or resolved_coord is None:
        dtype_token = _coerce_host_tensor_dtype(dtype)
        rs_stride, rs_coord = _try_remap_stride_coord_trees(
            shape_tree, origin_tree, dtype_token, layout_token
        )
        if resolved_stride is None:
            if rs_stride is None:
                raise ValueError(
                    "make_fake_tensor(stride=None): cannot derive stride from layout "
                    "remap (remap stride tree must match shape tree); pass an explicit "
                    "stride tree"
                )
            resolved_stride = rs_stride
        if resolved_coord is None:
            if rs_coord is None:
                raise ValueError(
                    "make_fake_tensor(coord=None): cannot derive coord from layout "
                    "remap; use a flat logical origin_shape (M,N) without nested "
                    "parentheses, or pass an explicit coord tree"
                )
            resolved_coord = rs_coord

    host_shape = _make_shape(*_as_components(shape_tree))
    host_stride = _make_stride(*_as_components(resolved_stride))
    host_origin = (
        None if origin_shape is None else _make_shape(*_as_components(origin_tree))
    )
    return _host_make_tensor(
        _make_ptr(dtype, 0, addrspace, assumed_align=assumed_align),
        _make_layout(
            host_shape,
            host_stride,
            origin_shape=host_origin,
            layoutTag=layout_tag,
        ),
        coord=resolved_coord,
    )


__all__ = [
    "DlpackBridgeError",
    "export_dlpack_capsule",
    "from_dlpack",
    "make_fake_tensor",
]
