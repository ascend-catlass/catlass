import torch
from torch import Tensor


def ascend950_89_ascend950_hif8_quant_batch_matmul(
    mat1: Tensor,
    mat2: Tensor,
    perChannelScale: Tensor,
    scalePerTensor: float = 1.0,
    bias: Tensor | None = None,
    quantMode: int = 0,
    outDType: str | torch.dtype = torch.bfloat16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 quant batch matmul with fixpipe per-tensor/per-channel dequant.

    Source: example 89_ascend950_hif8_quant_batch_matmul (hifloat8 path).

    hif8 A @ hif8 B, dequantized via fixpipe with per-tensor or per-channel scale,
    float bias added in-fixpipe.

    K 维直接按原始 shape 下发；非对齐 K 的尾部由 Ascend 950 HiF8 tile copy 路径处理。

    Args:
        mat1: raw hif8 bytes as uint8 ``(B, M, K)``.
        mat2: raw hif8 bytes as uint8 ``(B, K, N)``.
        perChannelScale: per-channel scale (uint64 encoded fp32, shape ``(N,)``)
            on CPU. Ignored unless ``quantMode == 1``.
        scalePerTensor: per-tensor scale (float scalar). Used when ``quantMode == 0``.
        bias: float ``(N,)``. Zeros when unused.
        quantMode: 0 = per_tensor, 1 = per_channel.
        outDType: Output dtype (bfloat16).
        transA: Must stay False; transposed layouts are not supported.
        transB: Must stay False; transposed layouts are not supported.
        useNzA: Must stay False; NZ block layout is not supported.
        useNzB: Must stay False; NZ block layout is not supported.

    Returns:
        Output ``(B, M, N)`` bf16.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    if bias is None:
        bias = torch.zeros(mat2.shape[-1], dtype=torch.float32, device=mat1.device)

    return torch.ops.catlass.ascend950_89_ascend950_hif8_quant_batch_matmul(
        mat1, mat2, perChannelScale,
        scalePerTensor, bias, quantMode, outDType,
        transA, transB, useNzA, useNzB
    )
