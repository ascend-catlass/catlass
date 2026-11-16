import struct

import pytest
import torch
import torch_npu
import torch_catlass

from common import only_on_3510


def _pow2(e: int) -> float:
    return struct.unpack("f", struct.pack("I", (e + 127) << 23))[0]


def hif8_decode(b: int) -> float:
    s = b >> 7
    t2 = (b >> 5) & 0x3
    if t2 == 3:
        d, dl = 4, 2
    elif t2 == 2:
        d, dl = 3, 2
    elif t2 == 1:
        d, dl = 2, 2
    elif (b >> 4) & 1:
        d, dl = 1, 3
    elif (b >> 3) & 1:
        d, dl = 0, 4
    else:
        m = b & 0x7
        if m == 0:
            return float("nan") if s else 0.0
        return (-1.0 if s else 1.0) * _pow2(m - 23)
    sh = 7 - dl
    em = ((b >> (sh - d)) & ((1 << d) - 1)) if d else 0
    w = 3 if d <= 2 else (2 if d == 3 else 1)
    m = b & ((1 << w) - 1)
    if d == 0:
        e = 0
    else:
        se = (em >> (d - 1)) & 1
        stored = (em & ((1 << (d - 1)) - 1)) if d > 1 else 0
        mag = (1 << (d - 1)) + stored
        e = -mag if se else mag
    sg = -1.0 if s else 1.0
    if d == 4 and mag == 15 and m == 1:
        return sg * 32768.0
    return sg * (1.0 + m / (1 << w)) * _pow2(e)


# Decode the byte domain once for the CPU reference.
_HIF8_TABLE = torch.tensor([hif8_decode(b) for b in range(256)], dtype=torch.float32)


def _decode_hif8_bytes(byte_tensor: torch.Tensor) -> torch.Tensor:
    return _HIF8_TABLE[byte_tensor.cpu().reshape(-1).to(torch.int64)].reshape(byte_tensor.shape)


def _truncate(v: float) -> float:
    bits = struct.unpack("I", struct.pack("f", v))[0] & 0xFFFFE000
    return struct.unpack("f", struct.pack("I", bits))[0]


def _encode_word(v: float) -> int:
    bits = struct.unpack("I", struct.pack("f", v))[0]
    return ((bits & 0xFFFFE000) | (1 << 46))


def _expected(a_bytes, b_bytes, bias, eff_scale):
    acc = torch.bmm(_decode_hif8_bytes(a_bytes), _decode_hif8_bytes(b_bytes))
    return (acc + bias.reshape(1, 1, -1)) * eff_scale.reshape(1, 1, -1)


@only_on_3510
@pytest.mark.parametrize(
    "quant_mode,B,m,n,k",
    [
        (0, 2, 128, 128, 256),   # per-tensor baseline
        (0, 2, 17, 19, 50),      # per-tensor native-K tail
        (1, 2, 128, 128, 50),    # per-channel native-K tail
    ],
    ids=[
        "qm0_per_tensor_K256",
        "qm0_per_tensor_M17_N19_K50",
        "qm1_per_channel_K50",
    ],
)
def test_89_ascend950_hif8_quant_batch_matmul(quant_mode, B, m, n, k):
    """hif8 (B,M,K) @ hif8 (B,K,N) -> bf16 (B,M,N)，fixpipe 随路反量化 + float bias。

    K 以原始 shape 直接下发，非 32 对齐的 K=50 用例验证 Ascend 950 HiF8 tile copy
    路径的设备侧尾块处理；M/N 也使用非对齐形状覆盖 per-tensor / per-channel。
    """
    a_bytes = torch.randint(8, 30, (B, m, k), dtype=torch.uint8)
    b_bytes = torch.randint(8, 30, (B, k, n), dtype=torch.uint8)

    bias = torch.tensor([0.5 if j % 2 == 0 else -1.0 for j in range(n)], dtype=torch.float32)

    if quant_mode == 1:
        scales = (torch.rand(n) * 1.99 + 0.01).tolist()
        words = torch.tensor([_encode_word(v) for v in scales], dtype=torch.uint64)  # (N,) CPU
        eff = torch.tensor([_truncate(v) for v in scales], dtype=torch.float32)      # (N,)
        scale_per_tensor_arg = 1.0  # per-channel 下不参与
    else:
        scale_value = 1.5
        words = torch.zeros(1, dtype=torch.uint64)  # 占位，per-tensor 下不参与
        eff = torch.full((n,), _truncate(scale_value), dtype=torch.float32)        # (N,)
        scale_per_tensor_arg = scale_value

    result = torch_catlass.ascend950_89_ascend950_hif8_quant_batch_matmul(
        a_bytes.npu(),
        b_bytes.npu(),
        words,
        scale_per_tensor_arg,
        bias.npu(),
        quantMode=quant_mode,
    )

    expected = _expected(a_bytes, b_bytes, bias, eff)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "npu"
    assert torch.allclose(result.cpu().float(), expected, rtol=1e-1, atol=1e-1), (
        f"quant_mode={quant_mode} shape=({B},{m},{n},{k}) max diff = "
        f"{(result.cpu().float() - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
