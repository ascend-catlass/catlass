"""Unit tests for DLPack helpers in :mod:`catlass.tla.runtime`."""

from __future__ import annotations

import ctypes
import gc
import weakref
from typing import Any

import pytest

import catlass.tla as tla
from catlass.base_dsl.runtime.dlpack_types import (
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLManagedTensor,
    DLManagedTensorDeleter,
)
from catlass.tla.runtime import from_dlpack
from catlass.types import RuntimeTensorError

np = pytest.importorskip("numpy")

# The capsule is passed as a raw PyObject* rather than ctypes.py_object: a
# destructor runs when the refcount has already reached zero, and py_object
# conversion would incref/decref it and free it a second time.
_CAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.c_void_p]

# ctypes trampolines and the structs the capsules point at must outlive the
# producer object: the deleter runs when the *consumer* drops the tensor, which
# in some tests is after the producer is gone. A real exporter has this for free
# (its deleter lives in the extension module), so pin them for the module's life.
_PINS: list[Any] = []
_DELETER_CALLS: dict[int, int] = {}


class _ManagedExport:
    """A minimal DLPack producer that behaves like a conforming C exporter.

    The capsule owns a real ``DLManagedTensor`` and carries a destructor that
    calls its deleter while the capsule is still named ``dltensor`` — exactly
    what a producer whose allocation is owned through ``manager_ctx`` does. The
    deleter records its calls in :attr:`deleter_calls`, so a test can tell
    "released" from "still valid" instead of assuming it.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, int] = (2, 3),
        strides: tuple[int, int] | None = (3, 1),
        device_type: int = int(DLDeviceType.kDLNpuCandidate1),
    ) -> None:
        self._shape = shape
        self._strides = strides
        self._device_type = int(device_type)
        #: Every ``stream=`` value ``from_dlpack`` asked this producer for.
        self.requested_streams: list[int | None] = []
        # Identify this producer by value, so the callbacks below do not close
        # over `self` and keep it alive past the point a test drops it.
        self._id = len(_DELETER_CALLS)
        _DELETER_CALLS[self._id] = 0

    @property
    def deleter_calls(self) -> int:
        return _DELETER_CALLS[self._id]

    def _build_managed(self) -> Any:
        elems = np.ascontiguousarray(
            np.arange(int(np.prod(self._shape)), dtype=np.float32).reshape(self._shape)
        )
        extents = (ctypes.c_int64 * len(self._shape))(*self._shape)
        managed = DLManagedTensor()
        managed.dl_tensor.data = ctypes.cast(elems.ctypes.data, ctypes.c_void_p)
        managed.dl_tensor.device = DLDevice(self._device_type, 0)
        managed.dl_tensor.ndim = len(self._shape)
        managed.dl_tensor.dtype = DLDataType(int(DLDataTypeCode.kDLFloat), 32, 1)
        managed.dl_tensor.shape = ctypes.cast(extents, ctypes.POINTER(ctypes.c_int64))
        if self._strides is None:
            managed.dl_tensor.strides = None
        else:
            stride_buf = (ctypes.c_int64 * len(self._strides))(*self._strides)
            managed.dl_tensor.strides = ctypes.cast(
                stride_buf, ctypes.POINTER(ctypes.c_int64)
            )
            _PINS.append(stride_buf)
        managed.dl_tensor.byte_offset = 0

        producer_id = self._id

        def _deleter(_managed_ptr: Any) -> None:
            _DELETER_CALLS[producer_id] += 1

        deleter = DLManagedTensorDeleter(_deleter)
        managed.deleter = deleter
        managed.manager_ctx = None
        _PINS.extend([elems, extents, managed, deleter])
        return managed

    def __dlpack_device__(self) -> tuple[int, int]:
        return (self._device_type, 0)

    def __dlpack__(self, stream: int | None = None) -> Any:
        self.requested_streams.append(stream)
        managed = self._build_managed()

        producer_id = self._id

        def _destroy(capsule_ptr: int) -> None:
            # A conforming destructor releases only an unconsumed capsule; a
            # consumer that took ownership renames it to 'used_dltensor'.
            if ctypes.pythonapi.PyCapsule_GetName(capsule_ptr) == b"dltensor":
                _DELETER_CALLS[producer_id] += 1

        destructor = _CAPSULE_DESTRUCTOR(_destroy)
        create = ctypes.pythonapi.PyCapsule_New
        create.restype = ctypes.py_object
        # Intentionally leave argtypes unset so ctypes converts from ``c_void_p``.
        payload = ctypes.cast(ctypes.pointer(managed), ctypes.c_void_p)
        capsule = create(payload, b"dltensor", destructor)
        if capsule is None:
            raise MemoryError("failed to allocate dltensor capsule")
        _PINS.append(destructor)
        return capsule


def test_from_dlpack_preserves_column_major_logical_metadata() -> None:
    tensor = from_dlpack(
        _ManagedExport(shape=(3, 2), strides=(2, 1)),
        layout_tag=tla.arch.ColumnMajor,
    )

    assert (tensor.shape, tensor.stride, tensor.origin_shape) == (
        (2, 3),
        (1, 2),
        (2, 3),
    )


def test_from_dlpack_zn_uses_layout_leaves_for_launch() -> None:
    tensor = from_dlpack(
        _ManagedExport(),
        layout_tag=tla.arch.zN,
        origin_shape=(32, 64),
    )

    assert tensor.shape == ((16, 2), (8, 8))
    assert tensor.stride == ((8, 128), (1, 256))
    assert tensor.build_memref_launch_fields() == (
        tensor.data_ptr,
        tensor.data_ptr,
        0,
        16,
        2,
        8,
        8,
        8,
        128,
        1,
        256,
        32,
        64,
    )


def test_parse_rejects_null_dlpack_strides() -> None:
    exporter = _ManagedExport(strides=None)

    with pytest.raises(RuntimeTensorError, match="null strides"):
        from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    # A rejected capsule was never consumed, so the producer still owns it and
    # its destructor frees it — from_dlpack must not leak on the error path.
    gc.collect()
    assert exporter.deleter_calls == 1


def test_from_dlpack_defers_the_deleter_until_the_tensor_dies() -> None:
    """The tensor owns the DLManagedTensor: released on its death, not before."""
    exporter = _ManagedExport()
    tensor = from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    # from_dlpack's own capsule reference is gone by now. Had the capsule not
    # been consumed, its destructor would already have freed the buffer that
    # tensor.data_ptr points at.
    gc.collect()
    assert exporter.deleter_calls == 0

    del tensor
    gc.collect()
    assert exporter.deleter_calls == 1, "the tensor did not release its DLManagedTensor"


def test_from_dlpack_keeps_producer_alive() -> None:
    """Producers whose deleter is a no-op still own the buffer via the object."""
    exporter = _ManagedExport()
    alive = weakref.ref(exporter)
    tensor = from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    # Drop the caller's only reference, as `from_dlpack(x.contiguous().to(dev))` does.
    del exporter
    gc.collect()

    assert alive() is not None, "from_dlpack dropped the DLPack producer"
    assert tensor._dlpack_source is alive()

    del tensor
    gc.collect()
    assert alive() is None, "producer outlived the tensor that borrowed it"


def test_from_dlpack_rejects_an_already_consumed_capsule() -> None:
    """A DLPack capsule is single-use; the second consumer must be told, not fed a stale pointer."""
    exporter = _ManagedExport()
    capsule = exporter.__dlpack__()

    class _Replay:
        def __dlpack__(self, stream: int | None = None) -> Any:
            del stream
            return capsule

    replay = _Replay()
    tensor = from_dlpack(replay, layout_tag=tla.arch.RowMajor)
    with pytest.raises(RuntimeTensorError, match="consumed once"):
        from_dlpack(replay, layout_tag=tla.arch.RowMajor)

    del tensor
    gc.collect()
    assert exporter.deleter_calls == 1


def test_from_dlpack_binds_a_host_buffer_for_compile_only() -> None:
    """A host buffer carries full metadata but no launchable address."""
    tensor = from_dlpack(
        _ManagedExport(shape=(3, 2), strides=(2, 1), device_type=DLDeviceType.kDLCPU),
        layout_tag=tla.arch.RowMajor,
    )

    # Identical metadata to the device binding -- that is the whole point: a
    # compile-only path can reuse the run path's tensor construction verbatim.
    assert (tensor.shape, tensor.stride, tensor.origin_shape) == (
        (3, 2),
        (2, 1),
        (3, 2),
    )
    assert tensor.data_ptr == 0
    with pytest.raises(RuntimeError, match="not bound to a device address"):
        tensor.__c_pointers__()


def test_from_dlpack_asks_a_host_producer_for_no_stream() -> None:
    """A stream is a device-queue handle; torch asserts when given one on CPU."""
    exporter = _ManagedExport(device_type=DLDeviceType.kDLCPU)

    from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    assert exporter.requested_streams == [None]


def test_from_dlpack_keeps_asking_a_device_producer_for_no_sync() -> None:
    exporter = _ManagedExport()

    from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    assert exporter.requested_streams == [-1]


def test_from_dlpack_still_rejects_a_foreign_device() -> None:
    # 2 is dlpack.h's kDLCUDA: neither Ascend nor host, so still a hard error.
    exporter = _ManagedExport(device_type=2)

    with pytest.raises(RuntimeTensorError, match="not .*an Ascend/NPU device"):
        from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    gc.collect()
    assert exporter.deleter_calls == 1


def test_from_dlpack_rejects_a_strided_vector_layout_buffer() -> None:
    """A flat 1-D layout is unit-stride; a strided view must not be bound.

    ``torch.arange(64)[::2]`` exports shape=(32,), strides=(2,). VectorLayout
    (like RowMajor) is built as ``(N):(1)`` downstream, so accepting this buffer
    would make the kernel read every other element while claiming a dense tile --
    silently the wrong data.
    """
    exporter = _ManagedExport(shape=(32,), strides=(2,))

    with pytest.raises(RuntimeTensorError, match="contiguous run"):
        from_dlpack(exporter, layout_tag=tla.arch.VectorLayout)

    # The rejected capsule was never consumed, so the producer still owns it and
    # its destructor releases the buffer exactly once.
    gc.collect()
    assert exporter.deleter_calls == 1


def test_from_dlpack_rejects_a_strided_row_major_buffer() -> None:
    exporter = _ManagedExport(shape=(32,), strides=(2,))

    with pytest.raises(RuntimeTensorError, match="contiguous run"):
        from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    gc.collect()
    assert exporter.deleter_calls == 1


def test_from_dlpack_rejects_a_strided_slice_behind_a_rank_one_origin() -> None:
    """The guard follows the projected rank, not the physical DLPack rank.

    ``arange(64).reshape(1, 64)[:, ::2]`` exports shape=(1, 32), strides=(64, 2)
    -- a rank-2 buffer, but the rank-1 ``origin_shape`` flattens it to
    ``(32):(1)`` exactly like the 1-D case, so judging by the physical rank would
    let the same silent stride loss through.
    """
    exporter = _ManagedExport(shape=(1, 32), strides=(64, 2))

    with pytest.raises(RuntimeTensorError, match="contiguous run"):
        from_dlpack(exporter, layout_tag=tla.arch.RowMajor, origin_shape=(32,))

    gc.collect()
    assert exporter.deleter_calls == 1


def test_from_dlpack_accepts_a_dense_single_row_behind_a_rank_one_origin() -> None:
    """A ``(1, N)`` buffer that is one contiguous row is a legitimate flat tile."""
    tensor = from_dlpack(
        _ManagedExport(shape=(1, 32), strides=(32, 1)),
        layout_tag=tla.arch.RowMajor,
        origin_shape=(32,),
    )

    assert (tensor.shape, tensor.stride, tensor.origin_shape) == (
        (32,),
        (1,),
        (32,),
    )


def test_from_dlpack_accepts_a_contiguous_one_dimensional_buffer() -> None:
    tensor = from_dlpack(
        _ManagedExport(shape=(32,), strides=(1,)),
        layout_tag=tla.arch.VectorLayout,
    )

    assert (tensor.shape, tensor.stride, tensor.origin_shape) == (
        (32,),
        (1,),
        (32,),
    )


def test_from_dlpack_accepts_a_size_one_view_with_any_stride() -> None:
    """A size-1 extent's stride is unobservable, so it must not be rejected.

    torch keeps the parent's stride on a one-element slice; the extent addresses
    nothing extra, so the flat projection is still exact.
    """
    tensor = from_dlpack(
        _ManagedExport(shape=(1,), strides=(7,)),
        layout_tag=tla.arch.VectorLayout,
    )

    assert (tensor.shape, tensor.stride) == ((1,), (1,))


def test_from_dlpack_accepts_an_empty_view_with_any_stride() -> None:
    """A zero-element view addresses nothing, so its stride cannot mislead."""
    tensor = from_dlpack(
        _ManagedExport(shape=(0,), strides=(2,)),
        layout_tag=tla.arch.VectorLayout,
    )

    assert (tensor.shape, tensor.stride) == ((0,), (1,))
