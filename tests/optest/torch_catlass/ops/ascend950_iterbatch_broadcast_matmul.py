import torch
from torch import Tensor


def ascend950_iterbatch_broadcast_matmul(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float32,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
    bias: Tensor | None = None,
    batch: int | None = None,
) -> Tensor:
    """Run CATLASS Ascend950 IterBatch broadcast batched matmul on NPU tensors.

    Source: experimental/matmul/85_ascend950_iterbatch_broadcast_matmul.

    Computes ``C[b] = cast(op(A[ba]) * op(B[bb]) + Bias(b))`` where a 2-D
    operand is broadcast over the batch dim (``ba``/``bb`` = 0) and a 3-D
    operand is per-batch (``ba``/``bb`` = ``b``). The batch dim is split into
    L1 groups / L0 sub-groups so broadcast operands are physically loaded
    once per group; requests that do not fit are rejected by the capacity
    plan instead of being clamped.

    Phase-1 small-matrix path: shapes must fit into the L1 tile
    (``m, n <= 128`` and ``k <= 128`` for fp16/bf16, ``k <= 64`` for fp32).

    Args:
        mat1: Left input. Shape ``(B, M, K)`` (per-batch) or ``(M, K)``
            (broadcast over batch) unless ``transA`` is true.
        mat2: Right input. Shape ``(B, K, N)`` (per-batch) or ``(K, N)``
            (broadcast over batch) unless ``transB`` is true.
        outDType: Output dtype. Accepted strings are ``float16``/``fp16``,
            ``float32``/``fp32`` and ``bf16``/``bfloat16``.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        formatA: Whether ``mat1`` is stored in the CATLASS NZ block format.
        formatB: Whether ``mat2`` is stored in the CATLASS NZ block format.
        bias: Optional bias. ``None`` for no bias, shape ``(N,)`` for a bias
            shared by all batches, shape ``(B, N)`` for a per-batch bias.
        batch: Batch count hint. Only required (and only used) when both
            ``mat1`` and ``mat2`` are 2-D broadcast operands, where the batch
            count cannot be derived from the input shapes.

    Returns:
        Output tensor with shape ``(B, M, N)`` on the active NPU device.
    """
    outDTypeRaw = outDType
    if isinstance(outDType, str):
        # torch has no fp16/fp32/bf16 attributes; map the short aliases first.
        dtype_alias = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_alias.get(dtype_lower, dtype_lower), None)
    if outDType is None:
        raise ValueError(f"{outDTypeRaw!r} is not a data type of torch")
    return torch.ops.catlass.ascend950_iterbatch_broadcast_matmul(
        mat1, mat2, outDType, transA, transB, formatA, formatB, bias,
        batch if batch is not None else -1
    )
