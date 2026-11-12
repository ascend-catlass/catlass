import ctypes
import struct
import sys
from contextlib import nullcontext

import pytest

from catlass.base_dsl.runtime import ascend_debug_fifo as fifo
from catlass.base_dsl.runtime import ascend_print as session
from catlass.base_dsl.runtime import ascend as runtime
from catlass.base_dsl.runtime.print_config import _PrintLaunchConfig
from catlass.compiler_metadata import KernelRuntimeMetadata


@pytest.mark.parametrize(
    "layout,expected",
    [
        (("fifo", "user", "simt"), (101, 7, 202)),
        (("user", "simt", "fifo"), (7, 202, 101)),
        (("user", "fifo", "tunnel"), (7, 101, 303)),
    ],
)
def test_context_binding_preserves_user_bytes_and_compiler_order(layout, expected):
    addresses = {
        key: value
        for key, value in {"fifo": 101, "simt": 202, "tunnel": 303}.items()
        if key in layout
    }
    payload = session.bind_arguments(struct.pack("<Q", 7), layout, addresses)
    assert struct.unpack("<3Q", payload) == expected


def test_context_binding_aligns_partial_user_payload():
    assert session.bind_arguments(
        b"\x01\x02\x03\x04", ("user", "fifo"), {"fifo": 9}
    ) == (b"\x01\x02\x03\x04" + bytes(4) + struct.pack("<Q", 9))


def test_fifo_capacity_requires_aligned_control_records(monkeypatch):
    monkeypatch.setattr(fifo, "_RING_BUFFER_BYTES", 375)
    with pytest.raises(fifo.AscDebugFifoError, match="8-byte"):
        fifo.open_fifo(1)


@pytest.mark.parametrize(
    "layout", [("user", "fifo", "fifo"), ("fifo",), ("user", "missing")]
)
def test_invalid_layout_is_rejected_before_launch(layout):
    with pytest.raises(fifo.AscDebugFifoError, match="layout"):
        session.bind_arguments(b"", layout, {"fifo": 1})
    with pytest.raises(ValueError, match="layout"):
        KernelRuntimeMetadata(
            "kernel", uses_scalar_print=True, print_argument_layout=layout
        ).validate()


def _tensor_record(*, vf=False, subblock=0, position=1):
    data = bytearray(104)
    descriptor = 0x54500000 | (fifo._VF_RECORD_TAG if vf else (subblock + 1) << 16)
    struct.pack_into("<II", data, 0, 2, 96)
    struct.pack_into("<I", data, 12, 0)
    struct.pack_into("<I", data, 16, descriptor)
    struct.pack_into("<HH", data, 24, position, subblock)
    struct.pack_into("<II", data, 28, 1, 2)
    struct.pack_into("<I", data, 68, 8)
    struct.pack_into("<2f", data, 72, 1.5, -2.0)
    return data


def _scalar_record():
    data = bytearray(40)
    struct.pack_into("<IIIIQQ", data, 0, 1, 32, 0, 0, 16, 17)
    data[32:37] = b"x=%d\0"
    return data


def _read_capture(monkeypatch, payload, *, truncated=False):
    ring_offset, capacity = 88, 512
    block_length = ring_offset + capacity + 24
    data = bytearray(block_length + fifo._FIFO_CONTROL_BYTES)
    struct.pack_into("<H", data, 16, fifo._MAGIC)
    data[ring_offset : ring_offset + len(payload)] = payload
    struct.pack_into(
        "<IIQ", data, ring_offset + capacity, fifo._FIFO_BUF_IN, 16, len(payload)
    )
    struct.pack_into("<I", data, block_length, int(truncated))
    device = ctypes.create_string_buffer(bytes(data))
    host = ctypes.create_string_buffer(len(data))
    freed = []
    monkeypatch.setattr(fifo, "_acl_malloc_host", lambda size: ctypes.addressof(host))
    monkeypatch.setattr(fifo, "_acl_free_host", freed.append)
    monkeypatch.setattr(
        fifo, "_acl_memcpy", lambda dst, src, size, kind: ctypes.memmove(dst, src, size)
    )
    capture = fifo._FifoData(
        ctypes.addressof(device),
        len(data),
        1,
        1,
        True,
        block_length,
        ring_offset,
        capacity,
        aiv_subblock_count=2,
    )
    try:
        fifo.read_print_fifo(capture)
    finally:
        assert freed == [ctypes.addressof(host)]


@pytest.mark.parametrize("position,name", [(1, "UB"), (8, "REG")])
def test_shared_decoder_distinguishes_native_and_vf(monkeypatch, capfd, position, name):
    _read_capture(
        monkeypatch,
        _scalar_record()
        + _tensor_record()
        + _tensor_record(vf=True, subblock=1, position=position),
    )
    output = capfd.readouterr().out
    assert output.index("x=17") < output.index("DumpTensor:")
    assert "block=0, subblock=0" in output
    assert "block=0, subblock=1" in output
    assert output.count("[1.5, -2]") == 2
    assert f"position={name}" in output


def test_shared_decoder_reports_truncation(monkeypatch, capfd):
    _read_capture(monkeypatch, _scalar_record(), truncated=True)
    output = capfd.readouterr().out
    assert "x=17" in output and "output truncated" in output


@pytest.mark.parametrize(
    "payload", [b"bad", struct.pack("<II", 2, 4096), struct.pack("<III", 99, 4, 0)]
)
def test_shared_decoder_rejects_corrupt_records_and_releases_host(monkeypatch, payload):
    with pytest.raises(fifo.AscDebugFifoError, match="malformed"):
        _read_capture(monkeypatch, payload)


@pytest.mark.parametrize("channels", [("fifo",), ("simt",), ("fifo", "simt")])
@pytest.mark.parametrize(
    "failure,error_cls",
    [
        (None, RuntimeError),
        ("launch", RuntimeError),
        ("sync", RuntimeError),
        ("decode", RuntimeError),
        ("launch", KeyboardInterrupt),
        ("sync", KeyboardInterrupt),
        ("decode", KeyboardInterrupt),
    ],
)
def test_one_sync_and_cleanup_or_retention(monkeypatch, channels, failure, error_cls):
    events, freed, allocations = [], [], []
    capture = fifo._FifoData(101, 0, 0, 1, False, 0, 0, 0)
    monkeypatch.setattr(fifo, "open_fifo", lambda *args, **kwargs: capture)
    monkeypatch.setattr(
        fifo, "destroy_fifo", lambda state: freed.append(state.device_ptr)
    )
    monkeypatch.setattr(
        fifo, "_acl_malloc", lambda size: allocations.append(size) or 202
    )
    monkeypatch.setattr(fifo, "_acl_free", freed.append)
    monkeypatch.setattr(fifo, "_acl_memcpy", lambda *args: None)

    def step(name):
        events.append(name)
        if failure == name:
            raise error_cls(name)

    monkeypatch.setattr(fifo, "_synchronize_launch", lambda stream: step("sync"))
    monkeypatch.setattr(fifo, "read_print_fifo", lambda state: step("decode"))
    monkeypatch.setattr(session.simt, "decode_log", lambda *args: step("decode") or [])

    addresses = {"fifo": 101, "simt": 202}

    def launch(payload):
        # A user value equal to a context address must remain untouched.
        assert struct.unpack(f"<{1 + len(channels)}Q", payload) == (
            202,
            *(addresses[channel] for channel in channels),
        )
        step("launch")

    retained = len(session._UNQUIESCED_SESSIONS)
    try:
        expectation = (
            pytest.raises(
                KeyboardInterrupt
                if error_cls is KeyboardInterrupt
                else (RuntimeError, fifo.AscDebugFifoError),
                match="completion is unknown"
                if failure == "sync" and error_cls is RuntimeError
                else failure,
            )
            if failure
            else nullcontext()
        )
        with expectation:
            session.launch_with_print(
                launch_kernel=launch,
                payload=struct.pack("<Q", 202),
                print_config=_PrintLaunchConfig(
                    mixed_core=False,
                    print_argument_layout=("user", *channels),
                    simt_print_formats=b"value=%d\0" if "simt" in channels else None,
                    print_tensor_position=None,
                ),
                block_num=1,
                stream=9,
                device=0,
            )
        # Format strings are host metadata; only control and log data need GM.
        expected_allocations = (
            [session.simt.HEADER_BYTES + session.simt.DATA_CAPACITY]
            if "simt" in channels
            else []
        )
        assert allocations == expected_allocations
        assert events.count("sync") == 1
        expected_freed = (
            [] if failure == "sync" else [addresses[channel] for channel in channels]
        )
        assert sorted(freed) == sorted(expected_freed)
        assert len(session._UNQUIESCED_SESSIONS) - retained == int(failure == "sync")
    finally:
        while len(session._UNQUIESCED_SESSIONS) > retained:
            session._UNQUIESCED_SESSIONS.pop().close()


@pytest.mark.parametrize("layout", [(), ("user", "fifo")])
def test_simt_rejects_missing_context_before_loading_runtime(monkeypatch, layout):
    monkeypatch.setitem(sys.modules, "acl", None)
    with pytest.raises(fifo.AscDebugFifoError, match="context layout"):
        runtime.launch_kernel(
            function_handle=0,
            stream=0,
            block_num=1,
            payload=b"",
            print_config=_PrintLaunchConfig(
                print_argument_layout=layout, simt_print_formats=b""
            ),
        )


def test_interruption_before_launch_releases_prepared_buffers(monkeypatch):
    freed = []
    capture = fifo._FifoData(101, 0, 0, 1, False, 0, 0, 0)
    monkeypatch.setattr(fifo, "open_fifo", lambda *args, **kwargs: capture)
    monkeypatch.setattr(fifo, "destroy_fifo", lambda state: freed.append(101))
    monkeypatch.setattr(fifo, "_acl_malloc", lambda size: 202)
    monkeypatch.setattr(fifo, "_acl_free", freed.append)

    def interrupted_copy(*args):
        raise KeyboardInterrupt("preparation")

    monkeypatch.setattr(fifo, "_acl_memcpy", interrupted_copy)
    retained = len(session._UNQUIESCED_SESSIONS)
    with pytest.raises(KeyboardInterrupt, match="preparation"):
        session.launch_with_print(
            launch_kernel=lambda payload: pytest.fail("must not launch"),
            payload=b"",
            print_config=_PrintLaunchConfig(
                mixed_core=False,
                print_argument_layout=("user", "fifo", "simt"),
                simt_print_formats=b"",
                print_tensor_position=None,
            ),
            block_num=1,
            stream=9,
            device=0,
        )
    assert sorted(freed) == [101, 202]
    assert len(session._UNQUIESCED_SESSIONS) == retained
