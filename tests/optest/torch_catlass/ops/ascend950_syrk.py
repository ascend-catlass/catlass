import torch
from torch import Tensor


def ascend950_syrk(
    mat: Tensor,
    input: Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    outDType: str | torch.dtype | None = None,
) -> Tensor:
    """Run CATLASS Ascend950 SYRK: ``out = alpha * (mat @ mat.T) + beta * input``.

    BLAS-style SYRK with ``mat`` as ``A`` and ``input`` as ``C``.

    Source: optest kernel 84_ascend950_syrk (MIX L0C→workspace + AIV AXPBY).

    Args:
        mat: Input matrix ``A`` with shape ``(M, K)`` or ``(B, M, K)`` on NPU.
        input: Square matrix ``C`` with shape ``(M, M)`` or ``(B, M, M)`` on NPU.
        alpha: Scale for ``mat @ mat.T``.
        beta: Scale for ``input``.
        outDType: Output dtype, either ``mat.dtype`` or ``float32``. Defaults to
            ``mat.dtype``. Accepted strings include ``float16`` / ``fp16``,
            ``bfloat16`` / ``bf16`` and ``float32`` / ``fp32``.

    Returns:
        Output tensor with shape ``(M, M)`` or ``(B, M, M)`` on the active NPU device.
    """
    if outDType is None:
        outDType = mat.dtype
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        if dtype_lower in ("bf16", "bfloat16"):
            outDType = torch.bfloat16
        elif dtype_lower in ("fp16", "float16"):
            outDType = torch.float16
        else:
            outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.ascend950_syrk(mat, input, alpha, beta, outDType)
