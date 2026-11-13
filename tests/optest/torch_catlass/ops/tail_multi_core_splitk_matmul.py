import torch
from torch import Tensor

MIN_K: int = 1025
MAX_K: int = 3072
MATRIX_RANK: int = 2


def ascend950_tail_multi_core_splitk_matmul(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float32,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Ascend950 tail multi-core split-K matmul (example 69).

    Splits the tail-round K dimension to create more work-blocks so that
    more cores can participate in the tail-round block computation.
    Requires ``C < B < 1.5 * C``, where B is the
    number of basic blocks and C is the number of AI cores.

    Inputs and output must be FP32, with 1025 <= K <= 3072.
    The native entry validates the block count against the actual AI core count.

    Args:
        mat1: left operand ``(M, K)`` or ``(K, M)`` when transA.
        mat2: right operand ``(K, N)`` or ``(N, K)`` when transB.
        outDType: output data type (``"float32"`` or ``torch.float32``).
        transA:  ``True`` if mat1 is column-major (shape K × M).
        transB:  ``True`` if mat2 is column-major (shape N × K).
        formatA: ``True`` if A uses Ascend NZ-format.
        formatB: ``True`` if B uses Ascend NZ-format.

    Returns:
        Tensor of shape ``(M, N)`` with dtype ``outDType`` on NPU.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType != torch.float32:
        raise ValueError("tail split-K output must be float32")
    if mat1.ndim != MATRIX_RANK or mat2.ndim != MATRIX_RANK:
        raise ValueError("tail split-K requires rank-2 inputs")
    if mat1.dtype != torch.float32 or mat2.dtype != mat1.dtype:
        raise ValueError("tail split-K requires matching float32 inputs")
    k = mat1.shape[0 if transA else 1]
    if k <= 0 or k != mat2.shape[1 if transB else 0]:
        raise ValueError("tail split-K requires matching positive K dimensions")
    if not MIN_K <= k <= MAX_K:
        raise ValueError(f"tail split-K requires {MIN_K} <= K <= {MAX_K}; got {k}")
    return torch.ops.catlass.ascend950_tail_multi_core_splitk_matmul(
        mat1, mat2, outDType, transA, transB, formatA, formatB
    )
