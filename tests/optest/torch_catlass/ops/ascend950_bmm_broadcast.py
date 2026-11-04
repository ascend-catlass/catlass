"""Continuous floating-point BMM broadcast, example 88 (Ascend 950)."""
import torch
from torch import Tensor


def ascend950_bmm_broadcast(
    a: Tensor, b: Tensor, batch: int, bias: Tensor | None = None,
    transA: bool = False, transB: bool = False, hf32: bool = False,
) -> Tensor:
    """Compute op(A) @ op(B) + Bias with FP32 accumulation.

    A/B are contiguous physical rank-3 tensors. A transposed input is stored
    as [batchA,K,M], B as [batchB,N,K]. Input batches are 1 or ``batch``.
    Bias is None, [N], or [batch,N], with the same dtype/device as A/B.
    HF32 requires FP32 inputs and is never enabled implicitly.
    The output is always [batch,M,N], including batch=1.
    """
    return torch.ops.catlass.ascend950_bmm_broadcast(a, b, batch, bias, transA, transB, hf32)
