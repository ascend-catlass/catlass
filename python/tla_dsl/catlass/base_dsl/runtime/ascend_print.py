"""One launch lifecycle for the CANN FIFO and the bounded SIMT log."""

from __future__ import annotations

import ctypes
import struct
from contextlib import ExitStack
from collections.abc import Callable

from . import ascend_debug_fifo as fifo
from . import ascend_simt_print as simt
from .print_config import _PrintLaunchConfig

# Keep cleanup callbacks and their resources alive when device completion is unknown.
_UNQUIESCED_SESSIONS: list[ExitStack] = []


def bind_arguments(
    payload: bytes, layout: tuple[str, ...], addresses: dict[str, int]
) -> bytes:
    """Insert contexts at the positions explicitly emitted by the compiler."""
    if (
        layout.count("user") != 1
        or len(layout) != len(set(layout))
        or set(layout) != {"user", *addresses}
    ):
        raise fifo.AscDebugFifoError(
            "print context layout does not match allocated resources"
        )
    result = bytearray()
    for role in layout:
        if role == "user":
            result.extend(payload)
        else:
            result.extend(bytes((-len(result)) % 8))
            address = addresses[role]
            if not address:
                raise fifo.AscDebugFifoError(f"null {role} print context")
            result.extend(struct.pack("<Q", address))
    return bytes(result)


def launch_with_print(
    *,
    launch_kernel: Callable[[bytes], None],
    payload: bytes,
    print_config: _PrintLaunchConfig,
    block_num: int,
    stream: int,
    device: int,
) -> None:
    layout = print_config.print_argument_layout
    is_mixed = print_config.mixed_core
    aiv_subblock_count = print_config.aiv_subblock_count
    tensor_position = print_config.print_tensor_position
    simt_formats = print_config.simt_print_formats
    addresses: dict[str, int] = {}
    debug_fifo = None
    simt_host = None
    simt_device = 0
    with ExitStack() as resources:
        if "fifo" in layout:
            debug_fifo = fifo.open_fifo(
                block_num,
                mixed_handoff=is_mixed,
                aiv_subblock_count=aiv_subblock_count,
                needs_debug_bus=tensor_position == "L1",
                device=device,
            )
            resources.callback(fifo.destroy_fifo, debug_fifo)
            addresses["fifo"] = debug_fifo.device_ptr
        if "tunnel" in layout:
            tunnel = fifo._DebugTunnelHostState.open(block_num)
            resources.callback(tunnel.close)
            addresses["tunnel"] = tunnel.state_ptr
        if "simt" in layout:
            if simt_formats is None:
                raise fifo.AscDebugFifoError("SIMT print format metadata is missing")
            size = simt.HEADER_BYTES + simt.DATA_CAPACITY
            initial = bytearray(size)
            struct.pack_into("<IIQ", initial, 0, 0, simt.DATA_CAPACITY, 0)
            simt_host = ctypes.create_string_buffer(bytes(initial))
            simt_device = fifo._acl_malloc(size)
            resources.callback(fifo._acl_free, simt_device)
            fifo._acl_memcpy(
                simt_device,
                ctypes.addressof(simt_host),
                size,
                fifo._ACL_MEMCPY_HOST_TO_DEVICE,
            )
            addresses["simt"] = simt_device
        arguments = bind_arguments(payload, layout, addresses)
        launch_error: BaseException | None = None
        completed = False
        # Once launch may have submitted work, every exit must either establish
        # stream completion or retain the device-owned buffers, including an
        # interruption between launch and synchronize.
        try:
            try:
                launch_kernel(arguments)
            except BaseException as error:
                launch_error = error
            try:
                fifo._synchronize_launch(stream)
            except BaseException as error:
                if not isinstance(error, Exception):
                    raise
                if launch_error is not None:
                    raise launch_error from error
                raise fifo.AscDebugFifoError(
                    "print stream completion is unknown; resources retained"
                ) from error
            else:
                completed = True
        finally:
            if not completed:
                _UNQUIESCED_SESSIONS.append(resources.pop_all())
        if launch_error is not None:
            raise launch_error
        if debug_fifo is not None:
            fifo.read_print_fifo(debug_fifo)
        if simt_host is not None:
            size = simt.HEADER_BYTES + simt.DATA_CAPACITY
            fifo._acl_memcpy(
                ctypes.addressof(simt_host),
                simt_device,
                size,
                fifo._ACL_MEMCPY_DEVICE_TO_HOST,
            )
            for line in simt.decode_log(simt_host.raw[:size], simt_formats):
                fifo._write_stdout(line)
