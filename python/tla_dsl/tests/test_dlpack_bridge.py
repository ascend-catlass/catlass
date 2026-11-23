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
from catlass.tla.runtime import _BIND_METADATA_MEMO, from_dlpack
from catlass.types import RuntimeTensorError

np = pytest.importorskip("numpy")

# The capsule is passed as a raw PyObject* rather than ctypes.py_object: a
# destructor runs when the refcount has already reached zero, and py_object
# conversion would incref/decref it and free it a second time.
_CAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

# Taken by subscript, not by attribute: an attribute lookup on
# ``ctypes.pythonapi`` hands out one shared function object per name, so a
# signature set here would also be seen by the binding under test.
_PyCapsule_GetName = ctypes.pythonapi["PyCapsule_GetName"]
_PyCapsule_GetName.restype = ctypes.c_char_p
_PyCapsule_GetName.argtypes = [ctypes.c_void_p]

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
        shape: tuple[int, ...] = (2, 3),
        strides: tuple[int, ...] | None = (3, 1),
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
            if _PyCapsule_GetName(capsule_ptr) == b"dltensor":
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


def _f32_packed_numel_zn(rows: int, cols: int) -> int:
    """Padded f32 zN element count: round_up(M,16) * round_up(N,8)."""
    return ((rows + 15) // 16 * 16) * ((cols + 7) // 8 * 8)


def _f32_packed_numel_nz(rows: int, cols: int) -> int:
    """Padded f32 nZ element count: round_up(M,8) * round_up(N,16)."""
    return ((rows + 7) // 8 * 8) * ((cols + 15) // 16 * 16)


@pytest.mark.parametrize("extra_storage", [0, 1])
def test_from_dlpack_zn_uses_layout_leaves_for_launch(extra_storage) -> None:
    rows, cols = 32, 64
    packed = _f32_packed_numel_zn(rows, cols)
    tensor = from_dlpack(
        _ManagedExport(shape=(packed + extra_storage,), strides=(1,)),
        layout_tag=tla.arch.zN,
        origin_shape=(rows, cols),
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


@pytest.mark.parametrize("extra_storage", [0, 1])
def test_from_dlpack_nz_uses_layout_leaves_for_launch(extra_storage) -> None:
    """nZ is the column-fractal dual of zN; still packs into the 13-slot ABI."""
    rows, cols = 32, 64
    packed = _f32_packed_numel_nz(rows, cols)
    tensor = from_dlpack(
        _ManagedExport(shape=(packed + extra_storage,), strides=(1,)),
        layout_tag=tla.arch.nZ,
        origin_shape=(rows, cols),
    )

    assert tensor.shape == ((8, 4), (16, 4))
    assert tensor.stride == ((1, 512), (8, 128))
    fields = tensor.build_memref_launch_fields()
    assert len(fields) == 13
    assert fields == (
        tensor.data_ptr,
        tensor.data_ptr,
        0,
        8,
        4,
        16,
        4,
        1,
        512,
        8,
        128,
        32,
        64,
    )


def test_from_dlpack_zn_reads_four_dimensional_export() -> None:
    tensor = from_dlpack(
        _ManagedExport(shape=(16, 2, 8, 8), strides=(8, 128, 1, 256)),
        layout_tag=tla.arch.zN,
    )

    assert tensor.layout_tag == "zN"
    assert tensor.shape == ((16, 2), (8, 8))
    assert tensor.stride == ((8, 128), (1, 256))
    assert tensor.build_memref_launch_fields()[-2:] == (32, 64)
    tensor.origin_shape = (28, 62)
    assert tensor.build_memref_launch_fields()[-2:] == (28, 62)


def test_from_dlpack_zn_canonicalizes_singleton_export_stride() -> None:
    tensor = from_dlpack(
        _ManagedExport(shape=(16, 1, 8, 2), strides=(8, 1, 1, 128)),
        layout_tag=tla.arch.zN,
    )
    assert tensor.layout_tag == "zN"
    assert tensor.stride == ((8, 128), (1, 128))


@pytest.mark.parametrize(
    ("strides", "layout_tag"),
    [((128, 64, 8, 1), tla.arch.zN), ((8, 128, 1, 256), tla.arch.nZ)],
)
def test_from_dlpack_rejects_four_dimensional_layout_mismatch(
    strides, layout_tag
) -> None:
    exporter = _ManagedExport(shape=(16, 2, 8, 8), strides=strides)
    with pytest.raises(RuntimeTensorError, match="do not match layout"):
        from_dlpack(exporter, layout_tag=layout_tag)


def test_from_dlpack_validates_four_dimensional_dynamic_cpu_sample() -> None:
    cpu = int(DLDeviceType.kDLCPU)
    kwargs = dict(layout_tag=tla.arch.zN, origin_shape=(None, None))
    shape = (16, 1, 8, 2)
    sample = from_dlpack(
        _ManagedExport(shape=shape, strides=(8, 128, 1, 128), device_type=cpu), **kwargs
    )
    assert sample.data_ptr == 0
    assert sample.origin_shape == (None, None)
    assert sample.shape == ((16, None), (8, None))
    with pytest.raises(RuntimeTensorError, match="do not match layout"):
        from_dlpack(
            _ManagedExport(shape=shape, strides=(8, 128, 1, 0), device_type=cpu),
            **kwargs,
        )


def test_from_dlpack_zn_rejects_four_dimensional_origin_inconsistency() -> None:
    """Trait-legal leaves that disagree with origin remap are still refused."""
    with pytest.raises(RuntimeTensorError, match="do not match layout"):
        from_dlpack(
            _ManagedExport(shape=(16, 4, 8, 8), strides=(8, 128, 1, 512)),
            layout_tag=tla.arch.zN,
            origin_shape=(32, 64),
        )


@pytest.mark.parametrize(
    ("shape", "strides"), [((2048,), (1,)), ((16, 2, 8, 8), (8, 128, 1, 256))]
)
def test_from_dlpack_requires_layout_tag(shape, strides) -> None:
    exporter = _ManagedExport(shape=shape, strides=strides)
    with pytest.raises(TypeError, match="layout_tag"):
        from_dlpack(exporter)
    with pytest.raises(RuntimeTensorError, match="requires layout_tag"):
        from_dlpack(exporter, layout_tag=None)


def test_host_make_tensor_zn_keeps_explicit_trees() -> None:
    host = tla.make_tensor(
        tla.make_ptr(tla.Float32, 0, tla.AddressSpace.gm),
        tla.make_layout(
            tla.make_shape((16, 2), (8, 8)),
            tla.make_stride((8, 128), (1, 256)),
            origin_shape=tla.make_shape(32, 64),
            layoutTag=tla.arch.zN,
        ),
    )
    assert host.shape == ((16, 2), (8, 8))
    assert host.stride == ((8, 128), (1, 256))
    assert host.origin_shape == (32, 64)


@pytest.mark.parametrize(
    ("shape", "stride", "message"),
    [
        (((8, 4), (8, 8)), ((8, 128), (1, 256)), "do not match layout"),
        (((16, 2), (8, 8)), ((8, 7), (1, 256)), "do not match layout"),
        (((16, 2), (8, 8)), (8, 7, 1, 256), "requires shape and stride"),
        ((16, 2, 8, 8), ((8, 128), (1, 256)), "requires shape and stride"),
    ],
)
def test_host_make_tensor_zn_rejects_bad_explicit_trees(shape, stride, message) -> None:
    # Host make_tensor trait-checks nested NZ trees; remap equality is a
    # from_dlpack gate (tiles may use a smaller ceil leaf than origin).
    with pytest.raises(RuntimeTensorError, match=message):
        tla.make_tensor(
            tla.make_ptr(tla.Float32, 0, tla.AddressSpace.gm),
            tla.make_layout(
                tla.make_shape(*shape),
                tla.make_stride(*stride),
                origin_shape=tla.make_shape(32, 64),
                layoutTag=tla.arch.zN,
            ),
        )


@pytest.mark.parametrize("derive_stride", [False, True])
def test_make_fake_tensor_infers_nz_origin_and_coord(derive_stride) -> None:
    shape = ((16, 2), (8, 8))
    stride = ((8, 128), (1, 256))
    fake = tla.make_fake_tensor(
        tla.Float32, shape, None if derive_stride else stride, layout_tag=tla.arch.zN
    )
    assert (fake.shape, fake.stride, fake.origin_shape, fake.coord) == (
        shape,
        stride,
        (32, 64),
        (0, 0),
    )


def test_from_dlpack_zn_requires_origin_shape() -> None:
    with pytest.raises(RuntimeTensorError, match="requires origin_shape"):
        from_dlpack(
            _ManagedExport(shape=(2048,), strides=(1,)),
            layout_tag=tla.arch.zN,
        )


def test_from_dlpack_zn_rejects_nd_logical_view() -> None:
    """Aligned ND (M,N) must not pass as zN even when lengths coincide."""
    with pytest.raises(RuntimeTensorError, match="flat 1-D prepacked"):
        from_dlpack(
            _ManagedExport(shape=(32, 64), strides=(64, 1)),
            layout_tag=tla.arch.zN,
            origin_shape=(32, 64),
        )


def test_from_dlpack_zn_rejects_wrong_packed_numel() -> None:
    with pytest.raises(RuntimeTensorError, match="requires a contiguous prepacked"):
        from_dlpack(
            _ManagedExport(shape=(16,), strides=(1,)),
            layout_tag=tla.arch.zN,
            origin_shape=(32, 64),
        )


def test_from_dlpack_zn_rejects_strided_packed_run() -> None:
    with pytest.raises(RuntimeTensorError, match="contiguous prepacked"):
        from_dlpack(
            _ManagedExport(shape=(2048,), strides=(2,)),
            layout_tag=tla.arch.zN,
            origin_shape=(32, 64),
        )


@pytest.mark.parametrize(
    ("element_type", "origin", "storage_elems"),
    [(tla.Bool, (16, 32), 512), (tla.Float4E2M1, (16, 64), 512)],
)
def test_from_dlpack_zn_checks_storage_width(
    element_type, origin, storage_elems
) -> None:
    kwargs = dict(
        layout_tag=tla.arch.zN, origin_shape=origin, element_type=element_type
    )
    packed = from_dlpack(np.zeros(storage_elems, dtype=np.uint8), **kwargs)
    assert packed.origin_shape == origin
    oversized = from_dlpack(np.zeros(storage_elems + 1, dtype=np.uint8), **kwargs)
    assert (oversized.shape, oversized.stride) == (packed.shape, packed.stride)
    for bad_size in (64, storage_elems - 1):
        with pytest.raises(
            RuntimeTensorError, match=f"buffer of at least {storage_elems} elements"
        ):
            from_dlpack(np.zeros(bad_size, dtype=np.uint8), **kwargs)


def test_from_dlpack_nz_preserves_logical_origin_of_padded_view() -> None:
    tensor = from_dlpack(
        _ManagedExport(shape=(8, 3, 16, 3), strides=(1, 384, 8, 128)),
        layout_tag=tla.arch.nZ,
        origin_shape=(19, 33),
    )
    assert tensor.layout_tag == "nZ"
    assert tensor.build_memref_launch_fields()[3:] == (
        8,
        3,
        16,
        3,
        1,
        384,
        8,
        128,
        19,
        33,
    )


def test_from_dlpack_rejects_float32_storage_as_float16() -> None:
    with pytest.raises(
        RuntimeTensorError, match="16-bit but the exported buffer is 32-bit"
    ):
        from_dlpack(
            np.zeros(512, dtype=np.float32),
            layout_tag=tla.arch.zN,
            origin_shape=(32, 32),
            element_type=tla.Float16,
        )


@pytest.mark.parametrize(
    ("layout", "origin", "stride"),
    [
        (tla.arch.RowMajorMxScaleA, (32, 3), ((0, 4), (1, 2))),
        (tla.arch.ColMajorMxScaleA, (32, 3), ((0, 2), (1, 64))),
        (tla.arch.RowMajorMxScaleB, (3, 32), ((1, 64), (0, 2))),
        (tla.arch.ColMajorMxScaleB, (3, 32), ((1, 2), (0, 4))),
    ],
)
def test_from_dlpack_mx_scale_checks_padded_storage(layout, origin, stride) -> None:
    kwargs = dict(layout_tag=layout, element_type=tla.Float8E8M0, origin_shape=origin)
    # An address-contiguous transpose is a valid prepared scale buffer.
    transposed = np.zeros((4, 32), dtype=np.uint8).T
    assert from_dlpack(transposed, **kwargs).stride == stride
    for bad_size in (1, 127):
        with pytest.raises(RuntimeTensorError, match="at least 128 storage elements"):
            from_dlpack(np.zeros(bad_size, dtype=np.uint8), **kwargs)
    with pytest.raises(RuntimeTensorError, match="requires dense MxScale storage"):
        from_dlpack(np.zeros(256, dtype=np.uint8)[::2], **kwargs)


def test_from_dlpack_mx_scale_infers_origin_from_2d_input() -> None:
    kwargs = dict(layout_tag=tla.arch.RowMajorMxScaleA, element_type=tla.Float8E8M0)
    tensor = from_dlpack(np.zeros((32, 4), dtype=np.uint8), **kwargs)
    assert (tensor.origin_shape, tensor.shape, tensor.stride) == (
        (32, 4),
        ((1, 32), (2, 2)),
        ((0, 4), (1, 2)),
    )
    with pytest.raises(RuntimeTensorError, match="requires origin_shape"):
        from_dlpack(np.zeros(128, dtype=np.uint8), **kwargs)


def test_from_dlpack_mx_scale_rejects_negative_export_extents() -> None:
    class InvalidExport(_ManagedExport):
        def _build_managed(self):
            managed = super()._build_managed()
            managed.dl_tensor.dtype = DLDataType(int(DLDataTypeCode.kDLUInt), 8, 1)
            managed.dl_tensor.shape[0] = -128
            managed.dl_tensor.shape[1] = -128
            return managed

    with pytest.raises(RuntimeTensorError, match="non-negative DLPack extents"):
        from_dlpack(
            InvalidExport(),
            layout_tag=tla.arch.RowMajorMxScaleA,
            element_type=tla.Float8E8M0,
            origin_shape=(32, 4),
        )


def test_from_dlpack_zn_dynamic_origin_is_compile_only() -> None:
    kwargs = dict(layout_tag=tla.arch.zN, origin_shape=(None, 16))
    sample = from_dlpack(np.zeros(256, dtype=np.float32), **kwargs)
    assert sample.data_ptr == 0
    assert sample.origin_shape == (None, 16)
    assert sample.shape == ((16, None), (8, 2))
    # Same geometry as the CPU sample must not get past the device check via the memo.
    with pytest.raises(RuntimeTensorError, match="with concrete integers"):
        from_dlpack(_ManagedExport(shape=(256,), strides=(1,)), **kwargs)


def test_host_make_tensor_zn_rejects_nested_origin() -> None:
    with pytest.raises(RuntimeTensorError, match="flat NZFamily origin_shape"):
        tla.make_tensor(
            tla.make_ptr(tla.Float32, 1, tla.AddressSpace.gm),
            tla.make_layout(
                tla.make_shape((16, 2), (8, 8)),
                tla.make_stride((8, 128), (1, 256)),
                origin_shape=tla.make_shape((16, 2), (8, 8)),
                layoutTag=tla.arch.zN,
            ),
        )


def test_linear_launch_fields_reject_nested_origin() -> None:
    bound = tla.make_tensor(
        tla.make_ptr(tla.Float32, 64, tla.AddressSpace.gm),
        tla.make_layout(
            tla.make_shape(4, 8),
            tla.make_stride(8, 1),
            origin_shape=tla.make_shape((2, 2), (2, 4)),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    with pytest.raises(RuntimeTensorError, match="concrete integer origin_shape"):
        bound.build_memref_launch_fields()


def test_host_make_tensor_zn_keeps_trait_valid_partial_view() -> None:
    shape, stride, origin = ((16, 1), (8, 8)), ((8, 128), (1, 256)), (32, 64)
    layout = tla.make_layout(
        tla.make_shape(*shape),
        tla.make_stride(*stride),
        origin_shape=tla.make_shape(*origin),
        layoutTag=tla.arch.zN,
    )
    host = tla.make_tensor(tla.make_ptr(tla.Float32, 1, tla.AddressSpace.gm), layout)
    assert host.build_memref_launch_fields()[3:] == (
        16,
        1,
        8,
        8,
        8,
        128,
        1,
        256,
        32,
        64,
    )
    fake = tla.make_fake_tensor(
        tla.Float32, shape, stride, layout_tag=tla.arch.zN, origin_shape=origin
    )
    assert (fake.shape, fake.stride, fake.origin_shape) == (shape, stride, origin)
    with pytest.raises(RuntimeTensorError, match="do not match layout"):
        from_dlpack(
            _ManagedExport(shape=(16, 1, 8, 8), strides=(8, 128, 1, 256)),
            layout_tag=tla.arch.zN,
            origin_shape=origin,
        )


def test_from_dlpack_derives_legacy_compact_strides() -> None:
    matrix = from_dlpack(_ManagedExport(strides=None), layout_tag=tla.arch.RowMajor)
    assert (matrix.shape, matrix.stride) == ((2, 3), (3, 1))
    tensor = from_dlpack(
        _ManagedExport(shape=(16, 1, 8, 1), strides=None), layout_tag=tla.arch.zN
    )
    assert tensor.layout_tag == "zN"
    assert tensor.build_memref_launch_fields()[3:] == (
        16,
        1,
        8,
        1,
        8,
        128,
        1,
        128,
        16,
        8,
    )


def test_from_dlpack_releases_a_buffer_with_rejected_geometry() -> None:
    exporter = _ManagedExport(strides=(1, 2))

    with pytest.raises(RuntimeTensorError, match="requires a buffer prepared"):
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


def test_from_dlpack_gives_two_buffers_of_one_geometry_independent_tensors() -> None:
    """Derived metadata is shared between binds; the tensors must not be.

    ``from_dlpack`` derives a buffer's tile metadata once per distinct geometry
    and reuses it, so two buffers of the same shape are described by the same
    trees. Each bind still owns its own tensor, address and coordinate -- sharing
    the tensor instead would make a second operand silently address the first.
    """
    first = _ManagedExport(shape=(4, 8), strides=(8, 1))
    second = _ManagedExport(shape=(4, 8), strides=(8, 1))

    one = from_dlpack(first, layout_tag=tla.arch.RowMajor)
    two = from_dlpack(second, layout_tag=tla.arch.RowMajor)

    assert one is not two
    assert one.data_ptr != two.data_ptr
    assert one.shape == two.shape
    assert one.origin_shape == two.origin_shape

    one.coord = (1, 1)
    assert two.coord == (0, 0)


def test_from_dlpack_keeps_rejecting_a_strided_buffer_after_a_dense_one() -> None:
    """A reused derivation must not become a reused verdict.

    The dense buffer's metadata is derived and kept; the strided buffer below has
    the same logical shape, so a derivation keyed on shape alone would hand it
    the dense tile's unit stride and bind it. The strides are part of what the
    buffer is judged on, so it is still refused.
    """
    dense = _ManagedExport(shape=(32,), strides=(1,))
    assert from_dlpack(dense, layout_tag=tla.arch.RowMajor).stride == (1,)

    strided = _ManagedExport(shape=(32,), strides=(2,))
    with pytest.raises(RuntimeTensorError, match="contiguous run"):
        from_dlpack(strided, layout_tag=tla.arch.RowMajor)

    # Still refused on a repeat: a rejection is never cached as a verdict.
    again = _ManagedExport(shape=(32,), strides=(2,))
    with pytest.raises(RuntimeTensorError, match="contiguous run"):
        from_dlpack(again, layout_tag=tla.arch.RowMajor)


def test_from_dlpack_derives_each_geometry_from_its_own_buffer() -> None:
    """A second shape, layout or origin is not served the first one's metadata."""
    wide = from_dlpack(
        _ManagedExport(shape=(4, 8), strides=(8, 1)), layout_tag=tla.arch.RowMajor
    )
    tall = from_dlpack(
        _ManagedExport(shape=(8, 4), strides=(4, 1)), layout_tag=tla.arch.RowMajor
    )
    assert (wide.origin_shape, tall.origin_shape) == ((4, 8), (8, 4))

    # The same physical buffer under ColumnMajor is a different logical tile.
    column = from_dlpack(
        _ManagedExport(shape=(4, 8), strides=(8, 1)), layout_tag=tla.arch.ColumnMajor
    )
    assert column.origin_shape == (8, 4)

    # And the same buffer flattened by a rank-1 origin is a different tile again.
    flat = from_dlpack(
        _ManagedExport(shape=(1, 8), strides=(8, 1)),
        layout_tag=tla.arch.RowMajor,
        origin_shape=8,
    )
    assert (flat.origin_shape, flat.stride) == ((8,), (1,))

    # Re-binding the first geometry still describes the first geometry.
    assert from_dlpack(
        _ManagedExport(shape=(4, 8), strides=(8, 1)), layout_tag=tla.arch.RowMajor
    ).origin_shape == (4, 8)


def test_from_dlpack_separates_two_geometries_that_differ_only_in_shape() -> None:
    """The physical shape alone decides a tile, with the strides held equal.

    Both buffers below are compact with strides ``(8, 1)``, so the shape is the
    only field between them: a derivation keyed without it would describe the
    second buffer with the first one's row count.
    """
    four_rows = from_dlpack(
        _ManagedExport(shape=(4, 8), strides=(8, 1)), layout_tag=tla.arch.RowMajor
    )
    three_rows = from_dlpack(
        _ManagedExport(shape=(3, 8), strides=(8, 1)), layout_tag=tla.arch.RowMajor
    )

    assert (four_rows.origin_shape, four_rows.shape) == ((4, 8), (4, 8))
    assert (three_rows.origin_shape, three_rows.shape) == ((3, 8), (3, 8))


def test_from_dlpack_separates_two_origins_over_one_physical_geometry() -> None:
    """The caller's origin alone decides a tile, with the buffer held equal.

    One physical ``(1, 8)`` buffer is bound twice, as itself and as a flat
    rank-1 tile: the origin is the only field between them, so a derivation
    keyed without it would hand the second bind the rank-2 tile's metadata.
    """
    kept = from_dlpack(
        _ManagedExport(shape=(1, 8), strides=(8, 1)), layout_tag=tla.arch.RowMajor
    )
    flattened = from_dlpack(
        _ManagedExport(shape=(1, 8), strides=(8, 1)),
        layout_tag=tla.arch.RowMajor,
        origin_shape=8,
    )

    assert (kept.origin_shape, kept.shape, kept.stride) == ((1, 8), (1, 8), (8, 1))
    assert (flattened.origin_shape, flattened.shape, flattened.stride) == (
        (8,),
        (8,),
        (1,),
    )


def test_from_dlpack_rejects_a_boolean_origin_extent_either_way_round() -> None:
    """``True`` is not an extent, and it cannot be mistaken for the extent 1.

    Python compares ``True`` equal to ``1`` and hashes them alike, so an origin
    kept in the caller's own spelling would let a rejected ``True`` bind and a
    valid ``1`` bind collide on one memo entry — in whichever order they arrive.
    """
    for origin in (True, (True, 8)):
        with pytest.raises(TypeError, match="int tree / tuple"):
            from_dlpack(
                _ManagedExport(shape=(1, 8), strides=(8, 1)),
                layout_tag=tla.arch.RowMajor,
                origin_shape=origin,
            )

    # The integer spelling of the same geometry still binds, after the rejection.
    bound = from_dlpack(
        _ManagedExport(shape=(1, 8), strides=(8, 1)),
        layout_tag=tla.arch.RowMajor,
        origin_shape=1,
    )
    assert bound.origin_shape == (1,)

    # And is not itself what makes a later boolean acceptable.
    with pytest.raises(TypeError, match="int tree / tuple"):
        from_dlpack(
            _ManagedExport(shape=(1, 8), strides=(8, 1)),
            layout_tag=tla.arch.RowMajor,
            origin_shape=True,
        )


def test_from_dlpack_rejects_a_float_origin_extent_after_the_integer_one() -> None:
    """``8.0`` is no index, and a bound ``8`` of the same geometry does not make it one."""
    bound = from_dlpack(
        _ManagedExport(shape=(1, 8), strides=(8, 1)),
        layout_tag=tla.arch.RowMajor,
        origin_shape=8,
    )
    assert bound.origin_shape == (8,)

    with pytest.raises(TypeError, match="int tree / tuple"):
        from_dlpack(
            _ManagedExport(shape=(1, 8), strides=(8, 1)),
            layout_tag=tla.arch.RowMajor,
            origin_shape=8.0,
        )


def test_capsule_entry_points_are_private_to_the_binding() -> None:
    """A signature another DLPack reader sets must not reach the binding.

    ``ctypes.pythonapi.PyCapsule_GetPointer`` is one shared object per process,
    so a consumer that declares its own return type on it would otherwise hand
    ``_parse_capsule`` a ctypes pointer where it reads an integer address.
    """
    from catlass.tla import runtime as tla_runtime

    shared = ctypes.pythonapi.PyCapsule_GetPointer
    assert tla_runtime._PyCapsule_GetPointer is not shared
    assert tla_runtime._PyCapsule_IsValid is not ctypes.pythonapi.PyCapsule_IsValid
    assert tla_runtime._PyCapsule_SetName is not ctypes.pythonapi.PyCapsule_SetName

    restore = shared.restype
    shared.restype = ctypes.POINTER(DLManagedTensor)
    try:
        bound = from_dlpack(
            _ManagedExport(shape=(4, 8), strides=(8, 1)), layout_tag=tla.arch.RowMajor
        )
    finally:
        shared.restype = restore
    assert bound.origin_shape == (4, 8)


def test_from_dlpack_binds_a_dynamic_origin_extent() -> None:
    """``None`` is an extent known only at run time, and binds as one.

    It is the leaf a Tla index tree carries for a dynamic extent, so a
    compile-only caller can spell the dynamic tile directly in the origin and
    get the descriptor ``mark_compact_shape_dynamic`` produces.
    """
    dynamic = from_dlpack(
        _ManagedExport(shape=(1, 64), strides=(64, 1)),
        layout_tag=tla.arch.RowMajor,
        origin_shape=(None,),
    )
    assert (dynamic.shape, dynamic.stride, dynamic.origin_shape) == (
        (None,),
        (1,),
        (None,),
    )
    assert dynamic._layout_shape_components() == (None,)

    marked = from_dlpack(
        _ManagedExport(shape=(1, 64), strides=(64, 1)),
        layout_tag=tla.arch.RowMajor,
        origin_shape=64,
    ).mark_compact_shape_dynamic(0)
    assert marked._layout_shape_components() == dynamic._layout_shape_components()


def test_from_dlpack_refuses_a_negative_origin_extent_without_keying_it() -> None:
    """A negative extent is no extent, and a refusal is never kept.

    Every leaf is judged before the derivation is keyed on it, so a tree the
    tensor would reject leaves nothing behind for a later bind to be served.
    """
    before = len(_BIND_METADATA_MEMO)

    for origin in (-1, (4, -8)):
        with pytest.raises(ValueError, match="non-negative"):
            from_dlpack(
                _ManagedExport(shape=(1, 8), strides=(8, 1)),
                layout_tag=tla.arch.RowMajor,
                origin_shape=origin,
            )

    assert len(_BIND_METADATA_MEMO) == before
