import torch
from torch import Tensor

_DTYPE_ALIASES_16 = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}
_DTYPE_ALIASES = dict(_DTYPE_ALIASES_16, fp32=torch.float32, float32=torch.float32)


def _resolve_dtype(value, aliases):
    """Map a dtype spelling such as ``"fp32"`` onto a torch dtype."""
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    resolved = aliases.get(lowered, getattr(torch, lowered, None))
    if resolved is None:
        raise ValueError(f"{value} is not a data type of torch")
    return resolved


def ascend950_basic_syrk(
    matX: Tensor,
    matXt: Tensor | None = None,
    outDType: str | torch.dtype | None = None,
    tile: int = 0,
    splitk: int = 0,
) -> Tensor:
    """Run CATLASS Ascend950 Basic SYRK: ``Y = X @ X.T``.

    Source: example 82_ascend950_basic_syrk.

    ``matXt`` is the second, transposed operand view and defaults to ``matX``;
    the lower triangle of ``matX @ matXt.T`` is mirrored for distinct operands,
    including inside diagonal tiles. Output is bit-exactly symmetric for both
    shared and distinct buffers. Both inputs may be
    row-major tensors or transposed views of row-major ``(K, M)`` / ``(B, K, M)``
    tensors, and must share layout, shape and dtype. ``M`` must be at least 32.

    Args:
        matX: Input ``X`` with shape ``(M, K)`` or ``(B, M, K)`` on NPU.
        matXt: Optional second input with the same shape as ``matX``.
        outDType: Output dtype. Defaults to ``matX.dtype``. Accepted strings
            include ``float16`` / ``fp16``, ``bfloat16`` / ``bf16`` and
            ``float32`` / ``fp32``. Inputs must be FP16 or BF16.
        tile: Square output tile edge: 0 (auto), 64, 128 or 256.
        splitk: 0 (auto) or 1; both select the single-pass kernel.

    Returns:
        Output ``Y`` with shape ``(M, M)`` or ``(B, M, M)`` on the active NPU device.
    """
    if outDType is None:
        outDType = matX.dtype
    if matXt is None:
        matXt = matX
    outDType = _resolve_dtype(outDType, _DTYPE_ALIASES)
    return torch.ops.catlass.ascend950_basic_syrk(matX, matXt, outDType, int(tile), int(splitk))
