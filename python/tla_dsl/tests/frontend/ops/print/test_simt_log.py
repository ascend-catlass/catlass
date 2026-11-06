import struct

import pytest

from catlass.base_dsl.runtime import ascend_debug_fifo as fifo
from catlass.base_dsl.runtime import ascend_simt_print as simt


def _log(record=b"", *, capacity=128, dropped=0):
    data = bytearray(64 + capacity)
    struct.pack_into(
        "<IIQ",
        data,
        0,
        len(record) | (simt.CLOSED if dropped else 0),
        capacity,
        dropped,
    )
    data[64 : 64 + len(record)] = record
    return data


def _scalar(value=7):
    record = bytearray(128)
    struct.pack_into(
        "<13I", record, 0, simt.COMPLETE, 128, 1, 0, 2, 0, 0, 3, 0, 0, 1, 1, 1
    )
    struct.pack_into("<Q", record, 64, value)
    return record


def test_scalar_keeps_identity_and_reports_dropped_calls():
    lines = simt.decode_log(_log(_scalar(), dropped=19), b"value=%d\0", capacity=128)
    assert "block=(2, 0, 0) thread=(3, 0, 0) value=7" in lines[0]
    assert (
        lines[1] == "TLA SIMT print: dropped 19 print call(s); log capacity exhausted\n"
    )


def test_empty_and_first_oversized_record():
    assert simt.decode_log(_log(capacity=64), b"", capacity=64) == []
    lines = simt.decode_log(_log(capacity=64, dropped=1), b"", capacity=64)
    assert len(lines) == 1 and "dropped 1" in lines[0]


@pytest.mark.parametrize(
    "offset,value", [(0, 0), (4, 256), (8, 99), (12, 100), (40, 9)]
)
def test_decoder_rejects_incomplete_or_corrupt_scalar(offset, value):
    record = _scalar()
    struct.pack_into("<I", record, offset, value)
    with pytest.raises(fifo.AscDebugFifoError):
        simt.decode_log(_log(record), b"value=%d\0", capacity=128)


def test_tensor_values_and_size_validation():
    record = bytearray(72)
    struct.pack_into(
        "<13I", record, 0, simt.COMPLETE, 72, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 8
    )
    struct.pack_into("<I", record, 52, 1)  # rank-1 view
    struct.pack_into("<2f", record, 64, -1.5, float("inf"))
    assert "[-1.5, inf]" in simt.decode_log(_log(record), b"", capacity=128)[0]
    struct.pack_into("<I", record, 40, 3)
    with pytest.raises(fifo.AscDebugFifoError, match="length"):
        simt.decode_log(_log(record), b"", capacity=128)
