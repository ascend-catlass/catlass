"""Per-launch bounded SIMT logs; private ABI shared with bc/Vector/simt_print.cpp."""

from __future__ import annotations

import struct

from . import ascend_debug_fifo as fifo

DATA_CAPACITY = 1 << 20
HEADER_BYTES = 64
COMPLETE = 0x544C4150
CLOSED = 1 << 31


def decode_log(
    data: bytes, formats: bytes, *, capacity: int = DATA_CAPACITY
) -> list[str]:
    """Validate complete records before rendering. No ordering across threads."""
    if len(data) < HEADER_BYTES + capacity:
        raise fifo.AscDebugFifoError("SIMT print buffer is truncated")
    cursor, declared, dropped = struct.unpack_from("<IIQ", data)
    used = cursor & ~CLOSED
    if declared != capacity or used > capacity or used % 8:
        raise fifo.AscDebugFifoError("SIMT print cursor/capacity is invalid")
    lines: list[str] = []
    offset = HEADER_BYTES
    end = offset + used
    while offset < end:
        if end - offset < HEADER_BYTES:
            raise fifo.AscDebugFifoError("SIMT print record header is truncated")
        fields = struct.unpack_from("<14I", data, offset)
        complete, size, kind, metadata = fields[:4]
        block, thread = fields[4:7], fields[7:10]
        count, rows, columns, rank = fields[10:14]
        if (
            complete != COMPLETE
            or size < HEADER_BYTES
            or size % 8
            or size > end - offset
        ):
            raise fifo.AscDebugFifoError(
                "SIMT print record is incomplete or out of bounds"
            )
        prefix = f"block={block} thread={thread}"
        payload = data[offset + HEADER_BYTES : offset + size]
        if kind == 1:
            if (
                size != 128
                or count > 8
                or metadata >= len(formats)
                or (metadata > 0 and formats[metadata - 1] != 0)
            ):
                raise fifo.AscDebugFifoError("SIMT scalar record is invalid")
            terminator = formats.find(b"\0", metadata)
            if terminator < 0:
                raise fifo.AscDebugFifoError("SIMT print format is unterminated")
            text = formats[metadata:terminator]
            if not fifo._is_supported_scalar_printf_format(text):
                raise fifo.AscDebugFifoError("SIMT scalar format is invalid")
            body = fifo._format_scalar_body(text, payload[: count * 8])
            if body is None:
                raise fifo.AscDebugFifoError("SIMT scalar argument count is invalid")
            lines.append(f"TLA printf: {prefix} {body}\n")
        elif kind == 2:
            dtype = fifo._print_tensor_dtype(metadata)
            if (
                dtype is None
                or count == 0
                or rank not in (1, 2)
                or (rank == 1 and rows != 1)
                or rows == 0
                or columns == 0
                or count > rows * columns
            ):
                raise fifo.AscDebugFifoError("SIMT tensor record metadata is invalid")
            name, width = dtype
            expected = HEADER_BYTES + ((count * width + 7) // 8) * 8
            if size != expected:
                raise fifo.AscDebugFifoError("SIMT tensor record length is invalid")
            values = ", ".join(
                fifo._format_tensor_value(metadata, payload, index)
                for index in range(count)
            )
            shape = (columns,) if rank == 1 else (rows, columns)
            lines.append(
                f"DumpTensor: {prefix} dtype={name} shape={shape} [{values}]\n"
            )
        else:
            raise fifo.AscDebugFifoError("SIMT print record kind is invalid")
        offset += size
    if dropped:
        lines.append(
            f"TLA SIMT print: dropped {dropped} print call(s); log capacity exhausted\n"
        )
    return lines
