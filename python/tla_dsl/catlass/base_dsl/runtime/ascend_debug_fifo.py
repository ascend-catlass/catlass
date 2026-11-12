"""Host-side Ascend debug FIFO transport (Python port of RuntimeWrapper AscDebugFifo).

Allocates the CANN print FIFO, swaps workspace sentinels into launch args,
then D2H-decodes scalar / tensor records to stdout with the same text formats
e2e tests capture (``TLA printf:`` / ``DumpTensor:``).
"""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass
from typing import Any, Callable

from ...execution import TlaRuntimeUnavailableError
from .ascend import check_acl_errors
from .print_config import _PrintLaunchConfig

DEBUG_PRINT_WORKSPACE_SENTINEL = int.from_bytes(b"TLA_PRNT", "big")
PRINT_TENSOR_WORKSPACE_SENTINEL = int.from_bytes(b"TLA_TPRN", "big")
DEBUG_TUNNEL_STATE_SENTINEL = int.from_bytes(b"TLA_DTUN", "big")

_DEBUG_CORE_RECORDS = 108
_RING_BUFFER_BYTES = 1 << 20
_FIFO_CONTROL_BYTES = 128
_MAGIC = 0xAE86
_PRINT_TENSOR_DESCRIPTOR_NAMESPACE = 0x54500000
_PRINT_TENSOR_DESCRIPTOR_NAMESPACE_MASK = 0xFFF80000
_VF_RECORD_TAG = 0x00040000
_GLOBAL_MEMORY_POSITION = 0
_UNIFIED_BUFFER_POSITION = 1
_LEVEL1_MEMORY_POSITION = 2
_LEVEL0C_MEMORY_POSITION = 5
_REGISTER_POSITION = 8

_FIFO_SCALAR = 1
_FIFO_TENSOR = 2
_FIFO_SHAPE = 3
_FIFO_BUF_IN = 8
_FIFO_BUF_OUT = 9

_HEAD_SIZE = 56
_READ_SIZE = 24
_WRITE_SIZE = 24
_PRINT_TLV_SIZE = 24
_PRINT_TENSOR_TLV_SIZE = 72
_PRINT_SHAPE_TLV_SIZE = 48
_MAX_SHAPE_DIM = 8
_TENSOR_PAYLOAD_ALIGNMENT = 32
# FIFO payload capacity leaves at most 262112 f32 values (see print_tensor.cpp).
_MAX_TENSOR_ELEMENTS = 262112
_PRINT_SLOT_BYTES = 8
_PRINT_FMT_OFFSET_BASE = 16

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2
_DRIVER_PROCESS_CP1 = 0
_DRIVER_RESOURCE_DEBUG_ADDRESS = 0x10


class _DriverResourceMapInfo(ctypes.Structure):
    _fields_ = [
        ("target_proc_type", ctypes.c_uint32),
        ("res_type", ctypes.c_uint32),
        ("res_id", ctypes.c_uint32),
        ("flag", ctypes.c_uint32),
        ("rsv", ctypes.c_uint32 * 1),
    ]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _ring_buffer_offset() -> int:
    payload_offset = _HEAD_SIZE + _READ_SIZE + _PRINT_TENSOR_TLV_SIZE
    return _align_up(payload_offset, 32) - _PRINT_TENSOR_TLV_SIZE


@dataclass
class _FifoData:
    device_ptr: int
    region_size: int
    record_count: int
    launch_block_count: int
    mixed_handoff: bool
    block_length: int
    ring_buffer_offset: int
    ring_buffer_bytes: int
    debug_bus_mappings: tuple[_DebugBusMapping, ...] = ()
    aiv_subblock_count: int = 1


class AscDebugFifoError(TlaRuntimeUnavailableError):
    """Raised when FIFO open/close/decode fails."""


class AscDebugUnsupportedError(AscDebugFifoError):
    """The installed runtime or driver lacks a required debug interface."""


@dataclass
class _DebugBusMapping:
    device: int
    info: _DriverResourceMapInfo
    address: int
    unmap_resource: Any
    closed: bool = False

    @classmethod
    def open(
        cls,
        device: int,
        resource_id: int,
        *,
        driver_hal: tuple[Any, Any] | None = None,
    ) -> _DebugBusMapping:
        map_resource, unmap_resource = (
            _driver_hal() if driver_hal is None else driver_hal
        )
        info = _DriverResourceMapInfo(
            _DRIVER_PROCESS_CP1,
            _DRIVER_RESOURCE_DEBUG_ADDRESS,
            int(resource_id),
            0,
            (0,),
        )
        address = ctypes.c_ulong(0)
        length = ctypes.c_uint(0)
        result = int(
            map_resource(
                int(device),
                ctypes.byref(info),
                ctypes.byref(address),
                ctypes.byref(length),
            )
        )
        if result or not address.value or not length.value:
            if not result and address.value:
                unmap_resource(int(device), ctypes.byref(info))
            detail = (
                f"driver error {result}"
                if result
                else "a null address"
                if not address.value
                else "a zero-length mapping"
            )
            raise AscDebugFifoError(
                f"halResMap(RES_DBG_ADDR, res_id={resource_id}) returned {detail}"
            )
        return cls(
            device=int(device),
            info=info,
            address=int(address.value),
            unmap_resource=unmap_resource,
        )

    def close(self) -> None:
        if self.closed:
            return
        # HAL has no retry contract, so relinquish ownership before unmapping.
        self.closed = True
        result = int(self.unmap_resource(self.device, ctypes.byref(self.info)))
        if result:
            raise AscDebugFifoError(
                f"halResUnmap(RES_DBG_ADDR) failed with driver error {result}"
            )


_DEBUG_TUNNEL_LOG_BUFFER_BYTES = 16 * 1024
_DEBUG_TUNNEL_LOG_PADDING_BYTES = 64
_DEBUG_TUNNEL_PHYSICAL_BLOCKS_PER_LOGICAL = 3


class _DebugTunnelPrintPayloadData(ctypes.Structure):
    _fields_ = [
        ("log_whole_region", ctypes.c_void_p),
        ("block_num", ctypes.c_uint32),
        ("log_buffer_size", ctypes.c_size_t),
        ("kernel_write_type", ctypes.c_uint32),
    ]


class _DebugTunnelData(ctypes.Structure):
    _fields_ = [
        ("print_data", _DebugTunnelPrintPayloadData),
        ("ffts_addr", ctypes.c_void_p),
    ]


if ctypes.sizeof(_DebugTunnelPrintPayloadData) != 32:
    raise RuntimeError("CANN DebugTunnel PrintPayloadData host ABI must be 32 bytes")
if ctypes.sizeof(_DebugTunnelData) != 40:
    raise RuntimeError("CANN DebugTunnelData host ABI must be 40 bytes")


def _debug_tunnel_payload_bytes(block_num: int) -> int:
    return (
        (_DEBUG_TUNNEL_LOG_BUFFER_BYTES + _DEBUG_TUNNEL_LOG_PADDING_BYTES)
        * int(block_num)
        * _DEBUG_TUNNEL_PHYSICAL_BLOCKS_PER_LOGICAL
    )


@dataclass
class _DebugTunnelHostState:
    state_ptr: int
    payload_ptr: int
    closed: bool = False

    @classmethod
    def open(cls, block_num: int) -> _DebugTunnelHostState:
        if block_num <= 0:
            raise AscDebugFifoError("CANN DebugTunnel requires a positive block count")
        payload_ptr = 0
        host_payload_ptr = 0
        state_ptr = 0
        try:
            payload_bytes = _debug_tunnel_payload_bytes(block_num)
            payload_ptr = _acl_malloc(payload_bytes)
            if not payload_ptr:
                raise AscDebugFifoError("CANN DebugTunnel payload allocation is null")
            host_payload_ptr = _acl_malloc_host(payload_bytes)
            if not host_payload_ptr:
                raise AscDebugFifoError("CANN DebugTunnel host allocation is null")
            ctypes.memset(host_payload_ptr, 0, payload_bytes)
            _acl_memcpy(
                payload_ptr,
                host_payload_ptr,
                payload_bytes,
                _ACL_MEMCPY_HOST_TO_DEVICE,
            )
            host_payload_to_free, host_payload_ptr = host_payload_ptr, 0
            _acl_free_host(host_payload_to_free)

            host_state = _DebugTunnelData(
                _DebugTunnelPrintPayloadData(
                    ctypes.c_void_p(payload_ptr),
                    int(block_num),
                    _DEBUG_TUNNEL_LOG_BUFFER_BYTES,
                    0,
                ),
                ctypes.c_void_p(),
            )
            state_ptr = _acl_malloc(ctypes.sizeof(host_state))
            if not state_ptr:
                raise AscDebugFifoError("CANN DebugTunnel state allocation is null")
            _acl_memcpy(
                state_ptr,
                ctypes.addressof(host_state),
                ctypes.sizeof(host_state),
                _ACL_MEMCPY_HOST_TO_DEVICE,
            )
            return cls(state_ptr=state_ptr, payload_ptr=payload_ptr)
        except Exception as primary_error:
            cleanup_error: Exception | None = None
            for ptr, free in (
                (host_payload_ptr, _acl_free_host),
                (state_ptr, _acl_free),
                (payload_ptr, _acl_free),
            ):
                if ptr:
                    try:
                        free(ptr)
                    except Exception as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
            if cleanup_error is not None:
                raise primary_error from cleanup_error
            raise

    def close(self) -> None:
        if self.closed:
            return
        first_error: Exception | None = None

        def attempt(action: Callable[[], None]) -> None:
            nonlocal first_error
            try:
                action()
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if self.payload_ptr:
            payload_ptr, self.payload_ptr = self.payload_ptr, 0
            attempt(lambda: _acl_free(payload_ptr))
        if self.state_ptr:
            state_ptr, self.state_ptr = self.state_ptr, 0
            attempt(lambda: _acl_free(state_ptr))
        self.closed = True
        if first_error is not None:
            raise first_error


# Retain launch resources when asynchronous completion is unknown.
_UNQUIESCED_LAUNCH_RESOURCES: list[tuple[_FifoData, _DebugTunnelHostState | None]] = []


def _acl_malloc(size: int) -> int:
    import acl

    ptr, ret = acl.rt.malloc(int(size), _ACL_MEM_MALLOC_HUGE_FIRST)
    check_acl_errors(ret, "acl.rt.malloc", error_cls=AscDebugFifoError)
    return int(ptr)


def _acl_malloc_host(size: int) -> int:
    import acl

    ptr, ret = acl.rt.malloc_host(int(size))
    check_acl_errors(ret, "acl.rt.malloc_host", error_cls=AscDebugFifoError)
    return int(ptr)


def _acl_free(ptr: int) -> None:
    import acl

    if ptr:
        check_acl_errors(
            acl.rt.free(int(ptr)), "acl.rt.free", error_cls=AscDebugFifoError
        )


def _acl_free_host(ptr: int) -> None:
    import acl

    if ptr:
        check_acl_errors(
            acl.rt.free_host(int(ptr)), "acl.rt.free_host", error_cls=AscDebugFifoError
        )


def _acl_memcpy(dst: int, src: int, size: int, kind: int) -> None:
    import acl

    check_acl_errors(
        acl.rt.memcpy(int(dst), int(size), int(src), int(size), int(kind)),
        "acl.rt.memcpy",
        error_cls=AscDebugFifoError,
    )


def _driver_device_id(device: int) -> int:
    """Translate an ACL user device ID to the logic ID required by driver HAL."""
    try:
        translate = ctypes.CDLL("libascendcl.so").aclrtGetLogicDevIdByUserDevId
    except (OSError, AttributeError) as exc:
        raise AscDebugUnsupportedError(
            "L1 tensor printing requires aclrtGetLogicDevIdByUserDevId"
        ) from exc
    translate.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    translate.restype = ctypes.c_int
    logic_device = ctypes.c_int32(-1)
    check_acl_errors(
        translate(int(device), ctypes.byref(logic_device)),
        "aclrtGetLogicDevIdByUserDevId",
        error_cls=AscDebugFifoError,
    )
    if logic_device.value < 0:
        raise AscDebugFifoError("CANN returned a negative driver logic device ID")
    return logic_device.value


def _driver_hal() -> tuple[Any, Any]:
    try:
        lib = ctypes.CDLL("libascend_hal.so")
        get_max = lib.halGetMaxResMapType
        map_resource = lib.halResMap
        unmap_resource = lib.halResUnmap
    except (OSError, AttributeError) as exc:
        raise AscDebugUnsupportedError(
            "L1 tensor printing requires driver HAL support for "
            "halGetMaxResMapType, halResMap, and halResUnmap"
        ) from exc
    get_max.restype = ctypes.c_uint
    map_resource.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(_DriverResourceMapInfo),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_uint),
    ]
    map_resource.restype = ctypes.c_int
    unmap_resource.argtypes = [ctypes.c_uint, ctypes.POINTER(_DriverResourceMapInfo)]
    unmap_resource.restype = ctypes.c_int
    if int(get_max()) < _DRIVER_RESOURCE_DEBUG_ADDRESS:
        raise AscDebugUnsupportedError(
            "L1 tensor printing requires a driver that supports RES_DBG_ADDR"
        )
    return map_resource, unmap_resource


def _debug_bus_support_error() -> str | None:
    """Return why the loaded driver lacks DebugBus support, if applicable."""

    try:
        _driver_hal()
    except AscDebugUnsupportedError as exc:
        return str(exc)
    return None


def open_fifo(
    block_num: int,
    *,
    mixed_handoff: bool = False,
    aiv_subblock_count: int = 1,
    needs_debug_bus: bool = False,
    device: int = 0,
) -> _FifoData:
    if (
        type(aiv_subblock_count) is not int
        or aiv_subblock_count not in (1, 2)
        or (not mixed_handoff and aiv_subblock_count != 1)
    ):
        raise AscDebugFifoError("invalid AIV subblock count for print FIFO")
    if _RING_BUFFER_BYTES <= 0 or _RING_BUFFER_BYTES % 8:
        raise AscDebugFifoError(
            "print FIFO capacity must preserve 8-byte write-control alignment"
        )
    driver_device = _driver_device_id(device) if needs_debug_bus else device
    ring_offset = _ring_buffer_offset()
    block_length = _align_up(ring_offset + _RING_BUFFER_BYTES + _WRITE_SIZE, 64)
    region_size = (block_length + _FIFO_CONTROL_BYTES) * _DEBUG_CORE_RECORDS
    device_ptr = _acl_malloc(region_size)
    host_ptr = 0
    debug_bus_mappings: list[_DebugBusMapping] = []
    try:
        host_ptr = _acl_malloc_host(region_size)
        ctypes.memset(host_ptr, 0, region_size)
        for i in range(_DEBUG_CORE_RECORDS):
            record = host_ptr + i * block_length
            buf = (ctypes.c_char * block_length).from_address(record)
            head = struct.pack(
                "<IIIIHHIQQIIII",
                block_length,
                i,
                _DEBUG_CORE_RECORDS,
                _RING_BUFFER_BYTES,
                _MAGIC,
                0,
                0,
                device_ptr + i * block_length + ring_offset,
                0,
                0,
                0,
                0,
                0,
            )
            if len(head) != _HEAD_SIZE:
                raise AscDebugFifoError("internal FIFO head size mismatch")
            buf[0:_HEAD_SIZE] = head
            read = struct.pack("<IIQQ", _FIFO_BUF_OUT, 16, 0, 0)
            if len(read) != _READ_SIZE:
                raise AscDebugFifoError("internal FIFO read-info size mismatch")
            buf[_HEAD_SIZE : _HEAD_SIZE + _READ_SIZE] = read
            write_off = ring_offset + _RING_BUFFER_BYTES
            write = struct.pack("<IIQQ", _FIFO_BUF_IN, 16, 0, 0)
            if len(write) != _WRITE_SIZE:
                raise AscDebugFifoError("internal FIFO write-info size mismatch")
            buf[write_off : write_off + _WRITE_SIZE] = write
        if needs_debug_bus:
            # Publish each core's DebugBus aperture in its matching FIFO header.
            driver_hal = _driver_hal()
            for i in range(_DEBUG_CORE_RECORDS):
                mapping = _DebugBusMapping.open(driver_device, i, driver_hal=driver_hal)
                debug_bus_mappings.append(mapping)
                ctypes.c_uint64.from_address(
                    host_ptr + i * block_length + 32
                ).value = mapping.address
        _acl_memcpy(device_ptr, host_ptr, region_size, _ACL_MEMCPY_HOST_TO_DEVICE)
        host_ptr_to_free, host_ptr = host_ptr, 0
        _acl_free_host(host_ptr_to_free)
    except Exception as primary_error:
        cleanup_error: Exception | None = None
        for mapping in reversed(debug_bus_mappings):
            try:
                mapping.close()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if host_ptr:
            try:
                _acl_free_host(host_ptr)
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        try:
            _acl_free(device_ptr)
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None:
            raise primary_error from cleanup_error
        raise
    return _FifoData(
        device_ptr=device_ptr,
        region_size=region_size,
        record_count=_DEBUG_CORE_RECORDS,
        launch_block_count=int(block_num),
        mixed_handoff=bool(mixed_handoff),
        aiv_subblock_count=aiv_subblock_count,
        block_length=block_length,
        ring_buffer_offset=ring_offset,
        ring_buffer_bytes=_RING_BUFFER_BYTES,
        debug_bus_mappings=tuple(debug_bus_mappings),
    )


def destroy_fifo(fifo: _FifoData | None) -> None:
    if fifo is None:
        return
    first_error: Exception | None = None
    if fifo.device_ptr:
        device_ptr, fifo.device_ptr = fifo.device_ptr, 0
        try:
            _acl_free(device_ptr)
        except Exception as exc:
            first_error = exc
    mappings, fifo.debug_bus_mappings = fifo.debug_bus_mappings, ()
    for mapping in reversed(mappings):
        try:
            mapping.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _args_to_u64_list(args: bytes) -> list[int]:
    if len(args) % 8 != 0:
        raise AscDebugFifoError(
            "debug workspace kernel arguments must be a multiple of 8 bytes"
        )
    return list(struct.unpack("<" + "Q" * (len(args) // 8), args))


def _u64_list_to_args(values: list[int]) -> bytes:
    return struct.pack("<" + "Q" * len(values), *values) if values else b""


def prepare_launch_args(
    args: bytes,
    *,
    print_config: _PrintLaunchConfig,
    fifo_device_ptr: int,
    debug_tunnel_device_ptr: int = 0,
) -> bytes:
    values = _args_to_u64_list(args)
    expects_debug_fifo = print_config.uses_scalar_print
    expects_print_tensor = print_config.uses_tensor_print
    mixed_tensor_handoff = print_config.mixed_core and expects_print_tensor
    expects_debug_tunnel = print_config.print_tensor_position in {"L1", "L0C"}
    debug_tunnel_value: int | None = None
    if expects_debug_tunnel:
        if not values or values[-1] != DEBUG_TUNNEL_STATE_SENTINEL:
            raise AscDebugFifoError(
                "debug tunnel state marker must occupy the final packed kernel argument"
            )
        if not debug_tunnel_device_ptr:
            raise AscDebugFifoError("debug tunnel state allocation is null")
        values.pop()
        debug_tunnel_value = int(debug_tunnel_device_ptr)
    if expects_debug_fifo and expects_print_tensor:
        raise AscDebugFifoError(
            "scalar debug FIFO and native tensor print cannot share a launch"
        )
    if expects_debug_fifo:
        if not values or values[-1] != DEBUG_PRINT_WORKSPACE_SENTINEL:
            raise AscDebugFifoError(
                "debug print FIFO marker must occupy the final packed kernel argument"
            )
        values[-1] = int(fifo_device_ptr)
        if debug_tunnel_value is not None:
            values.append(debug_tunnel_value)
        return _u64_list_to_args(values)
    if expects_print_tensor:
        if not values or values[-1] != PRINT_TENSOR_WORKSPACE_SENTINEL:
            raise AscDebugFifoError(
                "tensor print FIFO marker must occupy the final packed kernel argument"
            )
        workspace = int(fifo_device_ptr)
        if mixed_tensor_handoff:
            values[-1] = workspace
        else:
            values.pop()
            values.insert(0, workspace)
        if debug_tunnel_value is not None:
            values.append(debug_tunnel_value)
        return _u64_list_to_args(values)
    if debug_tunnel_value is not None:
        values.append(debug_tunnel_value)
        return _u64_list_to_args(values)
    return args


def _write_stdout(text: str) -> None:
    os.write(1, text.encode("utf-8", errors="replace"))


def _decode_float16(bits: int) -> float:
    sign = (bits & 0x8000) << 16
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x03FF
    if exponent == 0:
        if mantissa == 0:
            result = sign
        else:
            exponent = 113
            while (mantissa & 0x0400) == 0:
                mantissa <<= 1
                exponent -= 1
            mantissa &= 0x03FF
            result = sign | (exponent << 23) | (mantissa << 13)
    elif exponent == 0x1F:
        result = sign | 0x7F800000 | (mantissa << 13)
    else:
        result = sign | ((exponent + 112) << 23) | (mantissa << 13)
    return struct.unpack("<f", struct.pack("<I", result & 0xFFFFFFFF))[0]


def _print_tensor_dtype(code: int) -> tuple[str, int] | None:
    return {
        0: ("float32", 4),
        1: ("float16", 2),
        2: ("int8", 1),
        3: ("int32", 4),
        4: ("uint8", 1),
        6: ("int16", 2),
        7: ("uint16", 2),
        8: ("uint32", 4),
    }.get(code)


def _decode_subblock(descriptor: int) -> int:
    if (descriptor & _PRINT_TENSOR_DESCRIPTOR_NAMESPACE_MASK) != (
        _PRINT_TENSOR_DESCRIPTOR_NAMESPACE
    ):
        return -2
    tag = (descriptor >> 16) & 0x3
    if tag == 0:
        return -1
    if tag in (1, 2):
        return int(tag - 1)
    return -2


def _is_supported_scalar_printf_format(fmt: bytes) -> bool:
    """Match RuntimeWrapper: literals plus ``%d`` / ``%u`` / ``%f`` / ``%%``."""
    i = 0
    n = len(fmt)
    while i < n:
        if fmt[i] != ord("%"):
            i += 1
            continue
        if i + 1 >= n:
            return False
        spec = fmt[i + 1]
        if spec not in (ord("%"), ord("d"), ord("u"), ord("f")):
            return False
        i += 2
    return True


def _format_scalar_body(fmt: bytes, args: bytes) -> str | None:
    """Format a scalar printf TLV body.

    Returns the body string on success, ``None`` if argument slots are
    malformed, or ``""`` if the format string is unsupported.
    """
    if not _is_supported_scalar_printf_format(fmt):
        return ""
    parts: list[str] = []
    arg_offset = 0
    i = 0
    n = len(fmt)
    while i < n:
        if fmt[i] != ord("%"):
            parts.append(chr(fmt[i]))
            i += 1
            continue
        spec = fmt[i + 1]
        i += 2
        if spec == ord("%"):
            parts.append("%")
            continue
        if arg_offset + _PRINT_SLOT_BYTES > len(args):
            return None
        if spec == ord("d"):
            (slot,) = struct.unpack_from("<Q", args, arg_offset)
            parts.append(str(ctypes.c_int32(slot & 0xFFFFFFFF).value))
        elif spec == ord("u"):
            (slot,) = struct.unpack_from("<Q", args, arg_offset)
            parts.append(str(ctypes.c_uint32(slot & 0xFFFFFFFF).value))
        elif spec == ord("f"):
            (value,) = struct.unpack_from("<f", args, arg_offset)
            parts.append(f"{float(value):f}")
        else:
            return ""
        arg_offset += _PRINT_SLOT_BYTES
    if arg_offset != len(args):
        return None
    return "".join(parts)


def _print_scalar_tlv(record: memoryview, total: int, core: int) -> bool:
    if total < _PRINT_TLV_SIZE:
        _write_stdout(
            f"TLA printf: core={core} malformed scalar printf TLV (short record)\n"
        )
        return True
    _typ, _length, block_idx, _reserved, fmt_offset = struct.unpack_from(
        "<IIIIQ", record, 0
    )
    if fmt_offset < _PRINT_SLOT_BYTES or (fmt_offset % _PRINT_SLOT_BYTES) != 0:
        _write_stdout(
            f"TLA printf: core={core} malformed scalar printf TLV (bad fmtOffset)\n"
        )
        return True
    fmt_start = _PRINT_FMT_OFFSET_BASE + fmt_offset
    if fmt_start >= total:
        _write_stdout(
            f"TLA printf: core={core} malformed scalar printf TLV "
            f"(fmtOffset out of bounds)\n"
        )
        return True
    args_start = _PRINT_FMT_OFFSET_BASE + _PRINT_SLOT_BYTES
    if fmt_start < args_start or ((fmt_start - args_start) % _PRINT_SLOT_BYTES) != 0:
        _write_stdout(
            f"TLA printf: core={core} malformed scalar printf TLV (argument layout)\n"
        )
        return True
    fmt_bytes = bytes(record[fmt_start:total]).split(b"\0", 1)[0]
    if not _is_supported_scalar_printf_format(fmt_bytes):
        return False
    body = _format_scalar_body(fmt_bytes, bytes(record[args_start:fmt_start]))
    if body is None:
        body = "<malformed scalar printf TLV: missing argument slot>"
    elif body == "":
        return False
    _write_stdout(f"TLA printf: core={core} block={block_idx} {body}\n")
    return True


def _print_scalar_records(host: memoryview, fifo: _FifoData) -> None:
    printed = False
    for i in range(fifo.record_count):
        record = host[i * fifo.block_length : (i + 1) * fifo.block_length]
        magic = struct.unpack_from("<H", record, 16)[0]
        if magic != _MAGIC:
            continue
        ring = record[fifo.ring_buffer_offset :]
        buf_offset = struct.unpack_from("<Q", ring, fifo.ring_buffer_bytes + 8)[0]
        written = min(int(buf_offset), fifo.ring_buffer_bytes)
        offset = 0
        while offset + 8 <= written:
            typ, length = struct.unpack_from("<II", ring, offset)
            total = 8 + length
            if length == 0 or total > written - offset:
                if typ == _FIFO_SCALAR:
                    _write_stdout(
                        f"TLA printf: core={i} malformed scalar printf TLV "
                        f"(length out of bounds)\n"
                    )
                    printed = True
                break
            if typ == _FIFO_SCALAR:
                printed = (
                    _print_scalar_tlv(ring[offset : offset + total], total, i)
                    or printed
                )
            elif typ == _FIFO_TENSOR:
                # SIMD VF dumps share the scalar debug FIFO and may be
                # interleaved with formatted scalar records in program order.
                chunk = ring[offset : offset + total]
                shape, logical, subblock = _tensor_record_metadata(
                    chunk, total, fifo, vf_record=True
                )
                _render_tensor(chunk, shape, logical, subblock=subblock)
                printed = True
            offset += total
    if not printed:
        _write_stdout("TLA debug: no records captured\n")


def _format_tensor_value(data_type: int, payload: bytes, index: int) -> str:
    dtype = _print_tensor_dtype(data_type)
    assert dtype is not None
    _name, width = dtype
    off = index * width
    chunk = payload[off : off + width]
    if data_type == 0:
        return f"{struct.unpack_from('<f', chunk, 0)[0]:.9g}"
    if data_type == 1:
        bits = struct.unpack_from("<H", chunk, 0)[0]
        return f"{_decode_float16(bits):.9g}"
    if data_type == 2:
        return str(struct.unpack_from("<b", chunk, 0)[0])
    if data_type == 3:
        return str(struct.unpack_from("<i", chunk, 0)[0])
    if data_type == 4:
        return str(struct.unpack_from("<B", chunk, 0)[0])
    if data_type == 6:
        return str(struct.unpack_from("<h", chunk, 0)[0])
    if data_type == 7:
        return str(struct.unpack_from("<H", chunk, 0)[0])
    if data_type == 8:
        return str(struct.unpack_from("<I", chunk, 0)[0])
    raise AscDebugFifoError(f"unsupported tensor dtype code {data_type}")


def _render_tensor(
    tlv: memoryview,
    shape_tlv: memoryview,
    logical_block: int,
    *,
    subblock: int | None = None,
) -> None:
    data_type = struct.unpack_from("<I", tlv, 12)[0]
    desc = struct.unpack_from("<I", tlv, 16)[0]
    position = struct.unpack_from("<H", tlv, 24)[0]
    dump_size = struct.unpack_from("<I", tlv, 68)[0]
    shape_dim = struct.unpack_from("<I", shape_tlv, 8)[0]
    shape = struct.unpack_from("<8I", shape_tlv, 12)
    dtype = _print_tensor_dtype(data_type)
    if dtype is None:
        return
    name, width = dtype
    count = dump_size // width
    payload = bytes(tlv[_PRINT_TENSOR_TLV_SIZE : _PRINT_TENSOR_TLV_SIZE + dump_size])
    pos_name = {
        _GLOBAL_MEMORY_POSITION: "GM",
        _UNIFIED_BUFFER_POSITION: "UB",
        _LEVEL1_MEMORY_POSITION: "L1",
        _LEVEL0C_MEMORY_POSITION: "L0C",
        _REGISTER_POSITION: "REG",
    }[position]
    parts = [f"DumpTensor: call={desc & 0xFFFF}, block={logical_block}, "]
    if subblock is None:
        subblock = _decode_subblock(desc)
    if subblock >= 0:
        parts.append(f"subblock={subblock}, ")
    shape_text = ",".join(str(shape[i]) for i in range(shape_dim))
    parts.append(f"data_type={name}, position={pos_name}, shape=[{shape_text}] ")
    parts.append(f"dump_size={count} [")
    parts.append(
        ", ".join(_format_tensor_value(data_type, payload, i) for i in range(count))
    )
    parts.append("]\n")
    _write_stdout("".join(parts))


def _embedded_shape_tlv(tlv: memoryview) -> memoryview | None:
    """Shape TLV equivalent for a tensor TLV that describes its own shape.

    CANN 9.2 writes dim/shape straight into the tensor TLV and emits no
    standalone shape TLV. Rebuild the 48-byte shape TLV the rest of this module
    expects, or return None when the tensor carries no usable shape (a corrupt
    record).
    """

    if len(tlv) < _PRINT_TENSOR_TLV_SIZE:
        return None
    dim = struct.unpack_from("<I", tlv, 28)[0]
    if not 1 <= dim <= _MAX_SHAPE_DIM:
        return None
    extents = struct.unpack_from("<8I", tlv, 32)
    if any(extent == 0 for extent in extents[:dim]):
        return None
    shape_tlv = bytearray(_PRINT_SHAPE_TLV_SIZE)
    struct.pack_into("<III", shape_tlv, 0, _FIFO_SHAPE, _PRINT_SHAPE_TLV_SIZE - 8, dim)
    struct.pack_into("<8I", shape_tlv, 12, *extents)
    return memoryview(shape_tlv)


def _validate_tensor_tlv(tlv: memoryview, total: int, shape_tlv: memoryview) -> None:
    """Mirror RuntimeWrapper.cpp ``validate_tensor_tlv`` checks.

    The tensor TLV carries its own dim/shape (CANN 9.2 layout); ``shape_tlv``
    is the equivalent 48-byte shape record rebuilt from those fields.
    """
    if total < _PRINT_TENSOR_TLV_SIZE:
        raise AscDebugFifoError("malformed tensor print FIFO: truncated tensor header")
    length = struct.unpack_from("<I", tlv, 4)[0]
    if length != total - 8:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor length does not match record"
        )
    data_type = struct.unpack_from("<I", tlv, 12)[0]
    desc = struct.unpack_from("<I", tlv, 16)[0]
    position = struct.unpack_from("<H", tlv, 24)[0]
    dim = struct.unpack_from("<I", tlv, 28)[0]
    tensor_shape = struct.unpack_from("<8I", tlv, 32)
    dump_size = struct.unpack_from("<I", tlv, 68)[0]
    dtype = _print_tensor_dtype(data_type)
    if dtype is None:
        raise AscDebugFifoError("malformed tensor print FIFO: unsupported tensor dtype")
    _name, width = dtype
    if position not in (
        _GLOBAL_MEMORY_POSITION,
        _UNIFIED_BUFFER_POSITION,
        _LEVEL1_MEMORY_POSITION,
        _LEVEL0C_MEMORY_POSITION,
        _REGISTER_POSITION,
    ):
        raise AscDebugFifoError(
            "malformed tensor print FIFO: unsupported tensor position"
        )
    if dim == 0:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor dimension must be set"
        )
    if _decode_subblock(desc) < -1:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: invalid tensor descriptor namespace"
        )
    if (
        dump_size == 0
        or dump_size % width != 0
        or dump_size > _MAX_TENSOR_ELEMENTS * width
    ):
        raise AscDebugFifoError("malformed tensor print FIFO: invalid tensor dump size")
    expected_total = _PRINT_TENSOR_TLV_SIZE + _align_up(
        dump_size, _TENSOR_PAYLOAD_ALIGNMENT
    )
    if total != expected_total:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor payload size does not match record"
        )
    if any(extent == 0 for extent in tensor_shape[:dim]):
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor shape extents must be set"
        )
    if len(shape_tlv) != _PRINT_SHAPE_TLV_SIZE:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: missing or invalid tensor shape record"
        )
    shape_type, shape_length, shape_dim = struct.unpack_from("<III", shape_tlv, 0)
    shape_extents = struct.unpack_from("<8I", shape_tlv, 12)
    shape_reserved = struct.unpack_from("<I", shape_tlv, 44)[0]
    if (
        shape_type != _FIFO_SHAPE
        or shape_length != _PRINT_SHAPE_TLV_SIZE - 8
        or shape_reserved != 0
        or shape_dim not in (1, 2)
    ):
        raise AscDebugFifoError(
            "malformed tensor print FIFO: missing or invalid tensor shape record"
        )
    # CANN leaves shape slots beyond the declared rank untouched.
    shape_elements = 1
    for index, extent in enumerate(shape_extents):
        if index < shape_dim:
            if extent == 0:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: tensor shape contains a zero extent"
                )
            shape_elements *= extent
    if shape_elements < dump_size // width:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor shape is smaller than the dump size"
        )


def _tensor_record_metadata(
    chunk: memoryview, total: int, fifo: _FifoData, *, vf_record: bool = False
) -> tuple[memoryview, int, int]:
    shape = _embedded_shape_tlv(chunk)
    if shape is None:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor record missing shape"
        )
    _validate_tensor_tlv(chunk, total, shape)
    desc = struct.unpack_from("<I", chunk, 16)[0]
    block_idx = struct.unpack_from("<H", chunk, 26)[0]
    subblock = _decode_subblock(desc)
    logical = int(block_idx)
    if vf_record:
        position = struct.unpack_from("<H", chunk, 24)[0]
        if (
            position not in (_REGISTER_POSITION, _UNIFIED_BUFFER_POSITION)
            or subblock != -1
        ):
            raise AscDebugFifoError(
                "malformed VF print FIFO: invalid position or descriptor"
            )
        # CANN VF records encode block * subblockdim + subblockid, without
        # the descriptor tag used by the legacy native Tensor helper.
        if fifo.mixed_handoff:
            logical, subblock = divmod(block_idx, fifo.aiv_subblock_count)
    elif fifo.mixed_handoff and subblock >= 0:
        if block_idx % fifo.aiv_subblock_count != subblock:
            raise AscDebugFifoError(
                "malformed tensor print FIFO: tensor block index does not match mixed AIV subblock"
            )
        logical = block_idx // fifo.aiv_subblock_count
    if logical >= fifo.launch_block_count:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor block index exceeds launch block_num"
        )
    return shape, logical, subblock


def _decode_tensor_records(host: memoryview, fifo: _FifoData) -> None:
    records: list[tuple[memoryview, memoryview, int]] = []
    for i in range(fifo.record_count):
        record = host[i * fifo.block_length : (i + 1) * fifo.block_length]
        magic = struct.unpack_from("<H", record, 16)[0]
        if magic != _MAGIC:
            raise AscDebugFifoError("malformed tensor print FIFO: invalid record magic")
        ring = record[fifo.ring_buffer_offset :]
        write_type, write_len, buf_offset = struct.unpack_from(
            "<IIQ", ring, fifo.ring_buffer_bytes
        )
        if write_type != _FIFO_BUF_IN or write_len != 16:
            raise AscDebugFifoError(
                "malformed tensor print FIFO: invalid write-control record"
            )
        if buf_offset > fifo.ring_buffer_bytes:
            raise AscDebugFifoError(
                "malformed tensor print FIFO: ring write offset exceeds 1 MiB capacity"
            )
        offset = 0
        written = int(buf_offset)
        while offset < written:
            if offset + 8 > written:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: truncated TLV header"
                )
            typ, length = struct.unpack_from("<II", ring, offset)
            total = 8 + length
            if length == 0 or offset + total > written:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: TLV length exceeds captured bytes"
                )
            chunk = ring[offset : offset + total]
            # CANN 9.2 writes dim/shape straight into the tensor TLV and emits no
            # standalone shape record. That is the only layout supported here.
            if typ != _FIFO_TENSOR:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: unexpected record type"
                )
            shape_tlv, logical, _ = _tensor_record_metadata(chunk, total, fifo)
            records.append((shape_tlv, chunk, logical))
            offset += total
    if not records:
        _write_stdout("TLA debug: no records captured\n")
        return
    for shape_tlv, tlv, logical in records:
        _render_tensor(tlv, shape_tlv, logical)


def read_print_fifo(fifo: _FifoData) -> None:
    """Decode a shared FIFO record by record, without taking resource ownership."""
    host_ptr = _acl_malloc_host(fifo.region_size)
    try:
        _acl_memcpy(
            host_ptr, fifo.device_ptr, fifo.region_size, _ACL_MEMCPY_DEVICE_TO_HOST
        )
        host = (ctypes.c_char * fifo.region_size).from_address(host_ptr)
        view = memoryview(host).cast("B")
        printed = False
        for core in range(fifo.record_count):
            record = view[core * fifo.block_length : (core + 1) * fifo.block_length]
            if struct.unpack_from("<H", record, 16)[0] != _MAGIC:
                raise AscDebugFifoError("malformed print FIFO: invalid magic")
            ring = record[fifo.ring_buffer_offset :]
            control, length, written = struct.unpack_from(
                "<IIQ", ring, fifo.ring_buffer_bytes
            )
            if (
                control != _FIFO_BUF_IN
                or length != 16
                or written > fifo.ring_buffer_bytes
            ):
                raise AscDebugFifoError("malformed print FIFO: invalid write control")
            offset = 0
            while offset < written:
                if written - offset < 8:
                    raise AscDebugFifoError(
                        "malformed print FIFO: truncated record header"
                    )
                kind, length = struct.unpack_from("<II", ring, offset)
                total = length + 8
                if not length or total > written - offset:
                    raise AscDebugFifoError(
                        "malformed print FIFO: record length out of bounds"
                    )
                chunk = ring[offset : offset + total]
                if kind == _FIFO_SCALAR:
                    printed = _print_scalar_tlv(chunk, total, core) or printed
                elif kind == _FIFO_TENSOR:
                    if total < _PRINT_TENSOR_TLV_SIZE:
                        raise AscDebugFifoError(
                            "malformed print FIFO: truncated tensor"
                        )
                    descriptor = struct.unpack_from("<I", chunk, 16)[0]
                    shape, block, subblock = _tensor_record_metadata(
                        chunk, total, fifo, vf_record=bool(descriptor & _VF_RECORD_TAG)
                    )
                    _render_tensor(
                        chunk,
                        shape,
                        block,
                        subblock=subblock if subblock >= 0 else None,
                    )
                    printed = True
                else:
                    raise AscDebugFifoError(
                        "malformed print FIFO: unexpected record type"
                    )
                offset += total
        truncated = [
            core
            for core in range(fifo.record_count)
            if struct.unpack_from(
                "<I",
                view,
                fifo.block_length * fifo.record_count + core * _FIFO_CONTROL_BYTES,
            )[0]
        ]
        if truncated:
            _write_stdout(
                f"TLA print: output truncated; capacity exhausted on core(s) {truncated}\n"
            )
        if not printed:
            _write_stdout("TLA debug: no records captured\n")
    finally:
        _acl_free_host(host_ptr)


def close_fifo(
    fifo: _FifoData,
    *,
    tensor_only: bool,
) -> None:
    host_ptr = 0
    primary_error: Exception | None = None
    cleanup_error: Exception | None = None
    try:
        host_ptr = _acl_malloc_host(fifo.region_size)
        _acl_memcpy(
            host_ptr,
            fifo.device_ptr,
            fifo.region_size,
            _ACL_MEMCPY_DEVICE_TO_HOST,
        )
        host = (ctypes.c_char * fifo.region_size).from_address(host_ptr)
        view = memoryview(host).cast("B")
        if tensor_only:
            _decode_tensor_records(view, fifo)
        else:
            _print_scalar_records(view, fifo)
    except Exception as exc:
        primary_error = exc
    if host_ptr:
        host_ptr_to_free, host_ptr = host_ptr, 0
        try:
            _acl_free_host(host_ptr_to_free)
        except Exception as exc:
            cleanup_error = exc
    try:
        destroy_fifo(fifo)
    except Exception as exc:
        if cleanup_error is None:
            cleanup_error = exc
    if primary_error is not None:
        if cleanup_error is not None:
            raise primary_error from cleanup_error
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error


def _synchronize_launch(stream: int) -> None:
    import acl

    check_acl_errors(
        acl.rt.synchronize_stream(int(stream)),
        "acl.rt.synchronize_stream(AscDebugFifo)",
        error_cls=AscDebugFifoError,
    )


def launch_with_debug_fifo(
    *,
    launch_kernel: Callable[[bytes], None],
    payload: bytes,
    block_num: int,
    device: int,
    stream: int,
    print_config: _PrintLaunchConfig,
) -> None:
    """Open FIFO if needed, rewrite args, invoke ``launch_kernel``, then close."""
    uses_scalar_print = print_config.uses_scalar_print
    uses_tensor_print = print_config.uses_tensor_print
    aiv_subblock_count = print_config.aiv_subblock_count
    print_tensor_position = print_config.print_tensor_position
    is_mixed = print_config.mixed_core
    args = payload
    expects_debug_fifo = uses_scalar_print
    expects_print_tensor = uses_tensor_print
    expects_debug_tunnel = print_tensor_position in {"L1", "L0C"}
    needs_fifo = bool(expects_debug_fifo) or bool(expects_print_tensor)
    if not needs_fifo:
        launch_kernel(args)
        return

    needs_debug_bus = print_tensor_position == "L1"
    fifo = open_fifo(
        block_num,
        # Decoding needs the launch mode even when VF records use the scalar
        # FIFO. Argument rewriting below still uses the native Tensor protocol.
        mixed_handoff=is_mixed,
        aiv_subblock_count=aiv_subblock_count,
        needs_debug_bus=needs_debug_bus,
        device=device,
    )
    debug_tunnel_state: _DebugTunnelHostState | None = None
    try:
        if expects_debug_tunnel:
            debug_tunnel_state = _DebugTunnelHostState.open(block_num)
        rewritten = prepare_launch_args(
            args,
            print_config=print_config,
            fifo_device_ptr=fifo.device_ptr,
            debug_tunnel_device_ptr=(
                debug_tunnel_state.state_ptr if debug_tunnel_state is not None else 0
            ),
        )
    except Exception as primary_error:
        cleanup_error: Exception | None = None
        if debug_tunnel_state is not None:
            try:
                debug_tunnel_state.close()
            except Exception as exc:
                cleanup_error = exc
        try:
            destroy_fifo(fifo)
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None:
            raise primary_error from cleanup_error
        raise

    launch_error: Exception | None = None
    try:
        launch_kernel(rewritten)
    except Exception as exc:
        launch_error = exc

    try:
        _synchronize_launch(stream)
    except Exception as sync_error:
        _UNQUIESCED_LAUNCH_RESOURCES.append((fifo, debug_tunnel_state))
        if launch_error is not None:
            raise launch_error from sync_error
        raise AscDebugFifoError(
            f"{sync_error}; launch resources were retained because stream "
            "completion is unknown; reset the device context before continuing"
        ) from sync_error

    cleanup_error: Exception | None = None
    if debug_tunnel_state is not None:
        try:
            debug_tunnel_state.close()
        except Exception as exc:
            cleanup_error = exc

    if launch_error is not None:
        try:
            destroy_fifo(fifo)
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None:
            raise launch_error from cleanup_error
        raise launch_error

    try:
        close_fifo(fifo, tensor_only=bool(expects_print_tensor))
    except Exception as exc:
        if cleanup_error is None:
            cleanup_error = exc
    if cleanup_error is not None:
        raise cleanup_error


__all__ = [
    "AscDebugFifoError",
    "AscDebugUnsupportedError",
    "DEBUG_PRINT_WORKSPACE_SENTINEL",
    "PRINT_TENSOR_WORKSPACE_SENTINEL",
    "DEBUG_TUNNEL_STATE_SENTINEL",
    "launch_with_debug_fifo",
    "prepare_launch_args",
]
