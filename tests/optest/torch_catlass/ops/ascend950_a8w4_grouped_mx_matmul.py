import torch
from torch import Tensor


def ascend950_a8w4_grouped_mx_matmul(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    mx_scale_a: Tensor,
    mx_scale_b: Tensor,
) -> Tensor:
    """Run CATLASS Ascend950 grouped A8W4 MX matmul on NPU tensors.

    Source: example 74_ascend950_weight_quant_a8w4_grouped_mx_matmul.

    Computes grouped MX matmul ``C = (MxScaleA * A_fp8) @ (MxScaleB * B_fp4)``
    sliced on M. A is float8_e4m3fn (RowMajor, shared across groups). B is the
    packed FP4 prologue (int8 bytes, Weight4BitnZ layout, per group). Scales are
    float8_e8m0fnu. Output is FP16, matching example 74; the kernel accumulates in float.

    Args:
        mat1: Left input (float8_e4m3fn), shape ``(M, K)``. ``M`` is the total
            across all groups (``sum(groupList)``).
        mat2: Packed FP4 prologue bytes (int8), flat buffer spanning all groups
            in Weight4BitnZ fractal layout. Length ``G * kPadded * nPadded / 2``
            where ``kPadded = ceil(K/32)*32`` and ``nPadded = ceil(N/16)*16``.
        groupList: int64 group sizes (non-cumsum), shape ``(G,)``.
        mx_scale_a: MX scale for A (float8_e8m0fnu), shape
            ``(M, mxScaleAlignedK/2, 2)``.
        mx_scale_b: MX scale for B (float8_e8m0fnu), per group, shape
            ``(G, N, mxScaleAlignedK/2, 2)``.

    Returns:
        FP16 output tensor with shape ``(M, N)``.
    """
    return torch.ops.catlass.ascend950_a8w4_grouped_mx_matmul(
        mat1, mat2, groupList, mx_scale_a, mx_scale_b
    )
