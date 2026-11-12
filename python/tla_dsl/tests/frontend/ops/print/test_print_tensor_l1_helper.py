import ctypes
import struct
from types import SimpleNamespace

import pytest

from catlass.base_dsl.runtime import ascend_debug_fifo as fifo_mod
from catlass.base_dsl.runtime.print_config import _PrintLaunchConfig


class _FakeAcl:
    def __init__(
        self,
        monkeypatch,
        allocations: tuple[int, ...] = (0x1000,),
        *,
        fail_copy: int | None = None,
        null_host: bool = False,
        fail_host_free: bool = False,
    ):
        self._allocations = iter(allocations)
        self.fail_copy = fail_copy
        self.null_host = null_host
        self.fail_host_free = fail_host_free
        self.host_buffers: list[ctypes.Array] = []
        self.copies: list[tuple[int, int, bytes]] = []
        self.freed_device: list[int] = []
        self.freed_host: list[int] = []
        monkeypatch.setattr(fifo_mod, "_acl_malloc", self.malloc)
        monkeypatch.setattr(fifo_mod, "_acl_malloc_host", self.malloc_host)
        monkeypatch.setattr(fifo_mod, "_acl_free", self.free)
        monkeypatch.setattr(fifo_mod, "_acl_free_host", self.free_host)
        monkeypatch.setattr(fifo_mod, "_acl_memcpy", self.memcpy)

    def malloc(self, _size: int) -> int:
        return next(self._allocations)

    def malloc_host(self, size: int) -> int:
        if self.null_host:
            return 0
        buffer = ctypes.create_string_buffer(size)
        self.host_buffers.append(buffer)
        return ctypes.addressof(buffer)

    def free(self, ptr: int) -> None:
        self.freed_device.append(ptr)

    def free_host(self, ptr: int) -> None:
        self.freed_host.append(ptr)
        if self.fail_host_free:
            raise RuntimeError("host free failed")

    def memcpy(self, dst: int, src: int, size: int, _kind: int) -> None:
        self.copies.append((dst, size, ctypes.string_at(src, size)))
        if len(self.copies) == self.fail_copy:
            raise RuntimeError("copy failed")


class _FakeHal:
    def __init__(
        self,
        monkeypatch,
        *,
        records: int = 1,
        fail_map: int | None = None,
        map_length: int = 4096,
        fail_unmap: int | None = None,
    ):
        self.fail_map = fail_map
        self.map_length = map_length
        self.fail_unmap = fail_unmap
        self.mapped: list[tuple[int, int, int, int, int]] = []
        self.unmapped: list[tuple[int, int, int, int, int]] = []
        monkeypatch.setattr(fifo_mod, "_DEBUG_CORE_RECORDS", records)
        monkeypatch.setattr(fifo_mod, "_RING_BUFFER_BYTES", 64)
        monkeypatch.setattr(fifo_mod, "_driver_device_id", lambda device: device)
        monkeypatch.setattr(fifo_mod, "_driver_hal", lambda: (self.map, self.unmap))

    @staticmethod
    def _info(device: int, info_ptr: object) -> tuple[int, int, int, int, int]:
        info = info_ptr._obj
        return device, info.target_proc_type, info.res_type, info.res_id, info.flag

    def map(self, device, info, address, length) -> int:
        mapped = self._info(device, info)
        self.mapped.append(mapped)
        if mapped[3] == self.fail_map:
            return 7
        address._obj.value = 0x3FFFFFF6000 - mapped[3] * 0x1000
        length._obj.value = self.map_length
        return 0

    def unmap(self, device, info) -> int:
        unmapped = self._info(device, info)
        self.unmapped.append(unmapped)
        return 9 if unmapped[3] == self.fail_unmap else 0


class _FakeDriverFunction:
    def __init__(self, result: int = 0):
        self.result = result

    def __call__(self, *_args) -> int:
        return self.result


def test_debug_bus_support_probe_reports_driver_load_failure(monkeypatch) -> None:
    def fail_load(_name: str) -> None:
        raise OSError("driver unavailable")

    monkeypatch.setattr(fifo_mod.ctypes, "CDLL", fail_load)

    reason = fifo_mod._debug_bus_support_error()

    assert reason is not None
    assert "requires driver HAL support" in reason


def test_debug_bus_support_probe_reports_missing_symbol(monkeypatch) -> None:
    library = SimpleNamespace(halGetMaxResMapType=_FakeDriverFunction())
    monkeypatch.setattr(fifo_mod.ctypes, "CDLL", lambda _name: library)

    reason = fifo_mod._debug_bus_support_error()

    assert reason is not None
    assert "requires driver HAL support" in reason


def test_debug_bus_support_probe_reports_unsupported_resource_type(monkeypatch) -> None:
    library = SimpleNamespace(
        halGetMaxResMapType=_FakeDriverFunction(
            fifo_mod._DRIVER_RESOURCE_DEBUG_ADDRESS - 1
        ),
        halResMap=_FakeDriverFunction(),
        halResUnmap=_FakeDriverFunction(),
    )
    monkeypatch.setattr(fifo_mod.ctypes, "CDLL", lambda _name: library)

    reason = fifo_mod._debug_bus_support_error()

    assert reason is not None
    assert "supports RES_DBG_ADDR" in reason


def test_debug_bus_support_probe_accepts_supported_driver(monkeypatch) -> None:
    get_max = _FakeDriverFunction(fifo_mod._DRIVER_RESOURCE_DEBUG_ADDRESS)
    map_resource = _FakeDriverFunction()
    unmap_resource = _FakeDriverFunction()
    library = SimpleNamespace(
        halGetMaxResMapType=get_max,
        halResMap=map_resource,
        halResUnmap=unmap_resource,
    )
    monkeypatch.setattr(fifo_mod.ctypes, "CDLL", lambda _name: library)

    assert fifo_mod._debug_bus_support_error() is None
    assert get_max.restype is ctypes.c_uint
    assert map_resource.restype is ctypes.c_int
    assert unmap_resource.restype is ctypes.c_int


@pytest.mark.parametrize("error_cls", [RuntimeError, fifo_mod.AscDebugFifoError])
def test_debug_bus_support_probe_does_not_hide_unexpected_failures(
    monkeypatch,
    error_cls,
) -> None:
    def fail_probe() -> None:
        raise error_cls("unexpected driver failure")

    monkeypatch.setattr(fifo_mod, "_driver_hal", fail_probe)

    with pytest.raises(error_cls, match="unexpected driver failure"):
        fifo_mod._debug_bus_support_error()


@pytest.mark.parametrize("user,logic", [(0, 3), (1, 1)])
def test_driver_device_id_uses_cann_mapping(monkeypatch, user, logic):
    def translate(device, output):
        assert device == user
        output._obj.value = logic
        return 0

    monkeypatch.setattr(
        fifo_mod.ctypes,
        "CDLL",
        lambda name: SimpleNamespace(aclrtGetLogicDevIdByUserDevId=translate),
    )
    assert fifo_mod._driver_device_id(user) == logic
    assert translate.argtypes == [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]


def test_driver_mapping_failure_is_not_unsupported_capability(monkeypatch):
    translate = _FakeDriverFunction(107000)
    monkeypatch.setattr(
        fifo_mod.ctypes,
        "CDLL",
        lambda name: SimpleNamespace(aclrtGetLogicDevIdByUserDevId=translate),
    )
    with pytest.raises(fifo_mod.AscDebugFifoError, match="ret=107000") as error:
        fifo_mod._driver_device_id(0)
    assert not isinstance(error.value, fifo_mod.AscDebugUnsupportedError)


def test_missing_cann_mapping_interface_is_explicit(monkeypatch):
    monkeypatch.setattr(fifo_mod.ctypes, "CDLL", lambda name: SimpleNamespace())
    with pytest.raises(fifo_mod.AscDebugUnsupportedError, match="aclrtGetLogic"):
        fifo_mod._driver_device_id(0)


def test_fifo_map_and_unmap_use_the_same_translated_device(monkeypatch):
    _FakeAcl(monkeypatch)
    hal = _FakeHal(monkeypatch, records=2)
    calls = []
    monkeypatch.setattr(
        fifo_mod, "_driver_device_id", lambda user: calls.append(user) or 3
    )
    capture = fifo_mod.open_fifo(1, needs_debug_bus=True, device=0)
    fifo_mod.destroy_fifo(capture)
    assert calls == [0]
    assert [entry[0] for entry in hal.mapped] == [3, 3]
    assert [entry[0] for entry in hal.unmapped] == [3, 3]


def test_debug_fifo_ignores_shape_slots_beyond_declared_rank() -> None:
    total = 72 + 32
    tensor_tlv = bytearray(total)
    struct.pack_into("<I", tensor_tlv, 4, total - 8)
    struct.pack_into("<I", tensor_tlv, 12, 0)  # float32
    struct.pack_into("<I", tensor_tlv, 16, 0x54500000)
    struct.pack_into("<H", tensor_tlv, 24, 5)  # Hardware::L0C
    struct.pack_into("<I", tensor_tlv, 68, 4)
    # CANN 9.2 fills dim/shape into the tensor TLV itself; slots beyond the
    # declared rank are left untouched by the device.
    struct.pack_into("<I", tensor_tlv, 28, 1)
    struct.pack_into("<8I", tensor_tlv, 32, 1, 9, 9, 9, 9, 9, 9, 9)

    shape_tlv = fifo_mod._embedded_shape_tlv(memoryview(tensor_tlv))
    assert shape_tlv is not None
    fifo_mod._validate_tensor_tlv(memoryview(tensor_tlv), total, shape_tlv)


def _fifo_capture(
    payload: bytes,
    *,
    magic: int = fifo_mod._MAGIC,
    write_type: int = fifo_mod._FIFO_BUF_IN,
    write_len: int = 16,
    write_offset: int | None = None,
) -> tuple[memoryview, fifo_mod._FifoData]:
    ring_bytes = 256
    ring_offset = 80
    block_length = ring_offset + ring_bytes + fifo_mod._WRITE_SIZE
    host = bytearray(block_length)
    struct.pack_into("<H", host, 16, magic)
    host[ring_offset : ring_offset + len(payload)] = payload
    struct.pack_into(
        "<IIQ",
        host,
        ring_offset + ring_bytes,
        write_type,
        write_len,
        len(payload) if write_offset is None else write_offset,
    )
    fifo = fifo_mod._FifoData(
        device_ptr=0,
        region_size=block_length,
        record_count=1,
        launch_block_count=1,
        mixed_handoff=False,
        block_length=block_length,
        ring_buffer_offset=ring_offset,
        ring_buffer_bytes=ring_bytes,
    )
    return memoryview(host), fifo


@pytest.mark.parametrize(
    ("payload", "capture_kwargs", "match"),
    (
        (b"", {"magic": 0}, "invalid record magic"),
        (b"", {"write_offset": 257}, "ring write offset"),
        (b"\x00" * 7, {}, "truncated TLV header"),
        (
            struct.pack("<II", fifo_mod._FIFO_TENSOR, 16) + b"\x00" * 16,
            {},
            "tensor record missing shape",
        ),
    ),
)
def test_tensor_decoder_rejects_malformed_data(
    payload: bytes, capture_kwargs: dict[str, int], match: str
) -> None:
    host, fifo = _fifo_capture(payload, **capture_kwargs)

    with pytest.raises(fifo_mod.AscDebugFifoError, match=match):
        fifo_mod._decode_tensor_records(host, fifo)


def test_open_fifo_publishes_per_core_debug_bus_mappings_to_headers(
    monkeypatch,
) -> None:
    acl = _FakeAcl(monkeypatch)
    hal = _FakeHal(monkeypatch, records=3)

    fifo = fifo_mod.open_fifo(3, needs_debug_bus=True, device=5)
    try:
        expected_infos = [
            (
                5,
                fifo_mod._DRIVER_PROCESS_CP1,
                fifo_mod._DRIVER_RESOURCE_DEBUG_ADDRESS,
                resource_id,
                0,
            )
            for resource_id in range(3)
        ]
        assert hal.mapped == expected_infos
        copied = acl.copies[0][2]
        assert [
            struct.unpack_from("<Q", copied, index * fifo.block_length + 32)[0]
            for index in range(3)
        ] == [0x3FFFFFF6000, 0x3FFFFFF5000, 0x3FFFFFF4000]
    finally:
        fifo_mod.destroy_fifo(fifo)

    assert hal.unmapped == list(reversed(expected_infos))


def test_debug_bus_mapping_rejects_zero_length_and_unmaps(monkeypatch) -> None:
    hal = _FakeHal(monkeypatch, map_length=0)

    with pytest.raises(fifo_mod.AscDebugFifoError, match="zero-length mapping"):
        fifo_mod._DebugBusMapping.open(6, 0)

    assert [item[0] for item in hal.unmapped] == [6]


def test_open_fifo_rolls_back_prior_per_core_mapping_on_map_failure(
    monkeypatch,
) -> None:
    acl = _FakeAcl(monkeypatch)
    hal = _FakeHal(monkeypatch, records=3, fail_map=1)
    monkeypatch.setattr(
        fifo_mod,
        "_acl_memcpy",
        lambda *_args: pytest.fail("FIFO copy must not run after mapping failure"),
    )

    with pytest.raises(fifo_mod.AscDebugFifoError, match="res_id=1.*driver error 7"):
        fifo_mod.open_fifo(2, needs_debug_bus=True, device=3)

    assert [item[3] for item in hal.unmapped] == [0]
    assert acl.freed_device == [0x1000]


def test_destroy_fifo_attempts_every_release_once_and_relinquishes_ownership(
    monkeypatch,
) -> None:
    freed_device: list[int] = []
    hal = _FakeHal(monkeypatch, fail_unmap=1)

    def mapping(resource_id: int) -> object:
        info = fifo_mod._DriverResourceMapInfo()
        info.res_id = resource_id
        return fifo_mod._DebugBusMapping(
            device=4,
            info=info,
            address=0xABC000 + resource_id * 0x1000,
            unmap_resource=hal.unmap,
        )

    mappings = (mapping(0), mapping(1))
    fifo = fifo_mod._FifoData(
        device_ptr=0x2000,
        region_size=64,
        record_count=1,
        launch_block_count=1,
        mixed_handoff=False,
        block_length=64,
        ring_buffer_offset=0,
        ring_buffer_bytes=64,
        debug_bus_mappings=mappings,
    )
    monkeypatch.setattr(fifo_mod, "_acl_free", freed_device.append)

    with pytest.raises(fifo_mod.AscDebugFifoError, match="driver error 9"):
        fifo_mod.destroy_fifo(fifo)

    assert freed_device == [0x2000]
    assert [item[3] for item in hal.unmapped] == [1, 0]
    assert fifo.device_ptr == 0
    assert all(item.closed for item in mappings)
    assert fifo.debug_bus_mappings == ()


def test_debug_tunnel_host_state_matches_cann_layout(monkeypatch) -> None:
    acl = _FakeAcl(monkeypatch, (0x1000, 0x2000))

    state = fifo_mod._DebugTunnelHostState.open(2)

    assert ctypes.sizeof(fifo_mod._DebugTunnelPrintPayloadData) == 32
    assert ctypes.sizeof(fifo_mod._DebugTunnelData) == 40
    assert [
        getattr(fifo_mod._DebugTunnelPrintPayloadData, field).offset
        for field in (
            "log_whole_region",
            "block_num",
            "log_buffer_size",
            "kernel_write_type",
        )
    ] == [0, 8, 16, 24]
    assert fifo_mod._DebugTunnelData.ffts_addr.offset == 32
    assert state.state_ptr == 0x2000
    assert state.payload_ptr == 0x1000
    payload_bytes = fifo_mod._debug_tunnel_payload_bytes(2)
    assert acl.copies == [
        (0x1000, payload_bytes, bytes(payload_bytes)),
        (0x2000, 40, struct.pack("<QI4xQI4xQ", 0x1000, 2, 16 * 1024, 0, 0)),
    ]


def test_debug_tunnel_host_state_cleans_up_partial_open(monkeypatch) -> None:
    acl = _FakeAcl(monkeypatch, (0x1000, 0x2000), fail_copy=2)

    with pytest.raises(RuntimeError, match="copy failed"):
        fifo_mod._DebugTunnelHostState.open(1)

    assert acl.freed_device == [0x2000, 0x1000]


def test_debug_tunnel_host_state_rejects_null_host_allocation(monkeypatch) -> None:
    acl = _FakeAcl(monkeypatch, null_host=True)

    with pytest.raises(fifo_mod.AscDebugFifoError, match="host allocation is null"):
        fifo_mod._DebugTunnelHostState.open(1)

    assert acl.freed_device == [0x1000]


def test_debug_tunnel_open_does_not_retry_failed_host_release_during_cleanup(
    monkeypatch,
) -> None:
    acl = _FakeAcl(monkeypatch, fail_host_free=True)

    with pytest.raises(RuntimeError, match="host free failed"):
        fifo_mod._DebugTunnelHostState.open(1)

    assert acl.freed_host == [ctypes.addressof(acl.host_buffers[0])]
    assert acl.freed_device == [0x1000]


def test_debug_tunnel_host_state_attempts_each_release_once(monkeypatch) -> None:
    freed: list[int] = []

    def free(ptr: int) -> None:
        freed.append(ptr)
        if ptr == 0x1000:
            raise RuntimeError("payload free failed")

    monkeypatch.setattr(fifo_mod, "_acl_free", free)
    state = fifo_mod._DebugTunnelHostState(0x2000, 0x1000)

    with pytest.raises(RuntimeError, match="payload free failed"):
        state.close()

    assert freed == [0x1000, 0x2000]
    assert state.closed is True
    assert state.state_ptr == 0
    assert state.payload_ptr == 0
    state.close()
    assert freed == [0x1000, 0x2000]


def _fake_fifo() -> SimpleNamespace:
    return SimpleNamespace(device_ptr=0x1000)


def _install_local_launch(monkeypatch, tunnel: object) -> object:
    fifo = _fake_fifo()
    monkeypatch.setattr(fifo_mod, "open_fifo", lambda *_a, **_kw: fifo)
    monkeypatch.setattr(fifo_mod._DebugTunnelHostState, "open", lambda _blocks: tunnel)
    monkeypatch.setattr(
        fifo_mod, "prepare_launch_args", lambda *_a, **_kw: b"rewritten"
    )
    return fifo


def _launch_local(launch_kernel, *, position: str = "L1") -> None:
    fifo_mod.launch_with_debug_fifo(
        launch_kernel=launch_kernel,
        payload=b"payload",
        block_num=1,
        device=3,
        stream=7,
        print_config=_PrintLaunchConfig(
            mixed_core=False,
            uses_scalar_print=False,
            print_tensor_position=position,
            print_helper_core="aic",
        ),
    )


def _patch(monkeypatch, **attributes: object) -> None:
    for name, value in attributes.items():
        monkeypatch.setattr(fifo_mod, name, value)


def test_local_launch_synchronizes_once_before_releasing_resources(monkeypatch) -> None:
    events: list[str] = []
    tunnel = SimpleNamespace(
        state_ptr=0x2000, close=lambda: events.append("tunnel-close")
    )
    fifo = _install_local_launch(monkeypatch, tunnel)
    _patch(
        monkeypatch,
        open_fifo=lambda *_args, **kwargs: (
            events.append(f"fifo-open-{kwargs['device']}") or fifo
        ),
        _synchronize_launch=lambda _stream: events.append("sync"),
        close_fifo=lambda _fifo, **_kwargs: events.append("fifo-close"),
    )

    _launch_local(lambda payload: events.append(f"launch-{payload.decode()}"))

    assert events == [
        "fifo-open-3",
        "launch-rewritten",
        "sync",
        "tunnel-close",
        "fifo-close",
    ]


def test_local_launch_retains_resources_when_synchronization_fails(monkeypatch) -> None:
    tunnel = SimpleNamespace(
        state_ptr=0x2000, close=lambda: pytest.fail("freed tunnel")
    )
    fifo = _install_local_launch(monkeypatch, tunnel)
    retained: list[tuple[object, object]] = []
    _patch(
        monkeypatch,
        _UNQUIESCED_LAUNCH_RESOURCES=retained,
        _synchronize_launch=lambda _stream: (_ for _ in ()).throw(
            RuntimeError("sync failed")
        ),
        destroy_fifo=lambda *_args, **_kwargs: pytest.fail("freed fifo"),
    )

    with pytest.raises(fifo_mod.AscDebugFifoError, match="resources were retained"):
        _launch_local(lambda _payload: None)

    assert retained == [(fifo, tunnel)]


def test_local_launch_relinquishes_failed_quiesced_debug_tunnel(monkeypatch) -> None:
    tunnel = fifo_mod._DebugTunnelHostState(0, 0x1000)
    fifo = _install_local_launch(monkeypatch, tunnel)
    closed_fifos: list[object] = []
    _patch(
        monkeypatch,
        _synchronize_launch=lambda _stream: None,
        _acl_free=lambda _ptr: (_ for _ in ()).throw(
            RuntimeError("payload free failed")
        ),
        close_fifo=lambda closed, **_kwargs: closed_fifos.append(closed),
    )

    with pytest.raises(RuntimeError, match="payload free failed"):
        _launch_local(lambda _payload: None)

    assert tunnel.closed is True
    assert tunnel.payload_ptr == 0
    assert closed_fifos == [fifo]


def test_local_launch_preserves_launch_error_when_cleanup_also_fails(
    monkeypatch,
) -> None:
    cleanup_error = RuntimeError("tunnel cleanup failed")
    tunnel = SimpleNamespace(
        state_ptr=0x2000,
        close=lambda: (_ for _ in ()).throw(cleanup_error),
    )
    fifo = _install_local_launch(monkeypatch, tunnel)
    destroyed: list[object] = []
    _patch(
        monkeypatch,
        _synchronize_launch=lambda _stream: None,
        destroy_fifo=lambda destroyed_fifo: destroyed.append(destroyed_fifo),
    )

    with pytest.raises(RuntimeError, match="launch failed") as exc_info:
        _launch_local(
            lambda _payload: (_ for _ in ()).throw(RuntimeError("launch failed")),
            position="L0C",
        )

    assert exc_info.value.__cause__ is cleanup_error
    assert destroyed == [fifo]


@pytest.mark.parametrize(
    ("mixed_handoff", "expected"),
    (
        (False, (0x1000, 0xAA, 0xBB, 0x2000)),
        (True, (0xAA, 0xBB, 0x1000, 0x2000)),
    ),
)
def test_prepare_launch_args_places_fifo_by_explicit_handoff_mode(
    mixed_handoff: bool, expected: tuple[int, ...]
) -> None:
    payload = struct.pack(
        "<QQQQ",
        0xAA,
        0xBB,
        fifo_mod.PRINT_TENSOR_WORKSPACE_SENTINEL,
        fifo_mod.DEBUG_TUNNEL_STATE_SENTINEL,
    )

    rewritten = fifo_mod.prepare_launch_args(
        payload,
        print_config=_PrintLaunchConfig(
            print_helper_core="aic",
            mixed_core=mixed_handoff,
            print_tensor_position="L1",
        ),
        fifo_device_ptr=0x1000,
        debug_tunnel_device_ptr=0x2000,
    )

    assert struct.unpack(f"<{len(expected)}Q", rewritten) == expected
