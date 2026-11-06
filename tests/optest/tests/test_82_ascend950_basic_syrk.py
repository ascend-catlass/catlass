# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import pytest
import torch
import torch_npu
import torch_catlass

from common import only_on_3510


def _random_npu(shape, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(dtype).to("npu")


@only_on_3510
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("m,k", [(256, 256), (300, 200)])
def test_ascend950_basic_syrk(m, k, dtype):
    """Compare CATLASS Ascend950 Basic SYRK against torch.matmul(X, X.T)."""
    x = torch.randn(m, k, dtype=dtype, device="npu")

    result = torch_catlass.ascend950_basic_syrk(x)
    expected = torch.matmul(x.float(), x.float().T).to(dtype)

    assert result.shape == (m, m)
    assert result.dtype == dtype
    assert result.device.type == "npu"
    assert torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@only_on_3510
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ascend950_basic_syrk_batched_triangular_schedule(dtype):
    x = _random_npu((3, 768, 384), dtype, seed=3097)

    result = torch_catlass.ascend950_basic_syrk(x)
    expected = torch.matmul(x.float(), x.float().transpose(-2, -1)).to(dtype)

    assert result.shape == expected.shape
    torch.testing.assert_close(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@only_on_3510
@pytest.mark.parametrize("dtype,rtol", [(torch.float16, 1e-2), (torch.bfloat16, 2e-2)])
def test_ascend950_syrk_polynomial(dtype, rtol):
    batch, m, k = 2, 512, 384
    x = _random_npu((batch, m, k), dtype, seed=3097) * (k**-0.5)
    scales = torch.linspace(0.5, 1.5, k).to(dtype).to("npu")
    xt = x * scales
    residual_source = _random_npu((batch, m, k), dtype, seed=7351) * (k**-0.5)
    residual = torch.matmul(residual_source, residual_source.transpose(-2, -1))
    alpha, beta, diagonal = 2.0315, -4.7750, 3.4445

    result = torch_catlass.ascend950_syrk_polynomial(
        x, xt, residual, alpha, beta, diagonal
    )
    expected = alpha * torch.matmul(x.float(), xt.float().transpose(-2, -1))
    expected.add_(residual.float(), alpha=beta)
    expected.diagonal(dim1=-2, dim2=-1).add_(diagonal)

    torch.testing.assert_close(result.float(), expected, rtol=rtol, atol=rtol)
    assert torch.equal(result, result.transpose(-2, -1))


@only_on_3510
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ascend950_syrk_polynomial_without_residual(dtype):
    x = _random_npu((64, 96), dtype, seed=3097) * (96**-0.5)
    diagonal = 1.25

    result = torch_catlass.ascend950_syrk_polynomial(x, x, None, 0.75, 0.0, diagonal)
    expected = 0.75 * torch.matmul(x.float(), x.float().T)
    expected.diagonal().add_(diagonal)

    torch.testing.assert_close(result.float(), expected, rtol=1e-2, atol=1e-2)


@only_on_3510
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "batch,m,k,tile,splitk,transposed",
    [
        (None, 32, 129, 64, 1, False),
        (None, 300, 513, 128, 1, False),  # non-32-aligned M
        (1, 384, 1025, 128, 4, True),  # batch of one, uneven K slices, column-major X
        (5, 544, 513, 256, 4, True),  # M tail, column-major X
        (3, 1024, 257, 128, 2, False),  # K tail
        (1, 256, 16384, 0, 0, False),  # automatic configuration
    ],
)
def test_ascend950_basic_syrk_fp32_output(batch, m, k, tile, splitk, transposed, dtype):
    """Compare FP32 output for direct and split-K launches."""
    generator = torch.Generator().manual_seed(3097)
    prefix = () if batch is None else (batch,)
    physical_shape = prefix + ((k, m) if transposed else (m, k))
    physical = (torch.randn(physical_shape, generator=generator) * k**-0.5).to(dtype)
    logical = physical.transpose(-2, -1) if transposed else physical
    wide = logical.double()
    expected = wide @ wide.transpose(-2, -1)
    device = physical.to("npu")
    x = device.transpose(-2, -1) if transposed else device

    result = torch_catlass.ascend950_basic_syrk(
        x, outDType=torch.float32, tile=tile, splitk=splitk
    )
    actual = result.cpu()
    assert actual.shape == prefix + (m, m)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, actual.transpose(-2, -1))
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)


@only_on_3510
@pytest.mark.parametrize("m,k,tile,splitk", [(64, 3, 64, 1), (384, 513, 128, 4)])
def test_ascend950_basic_syrk_fp32_preserves_small_products(m, k, tile, splitk):
    # FP32 output preserves products that are rounded or flushed in FP16 output.
    x = torch.full((m, k), 2.0**-14, dtype=torch.float16, device="npu")
    expected = torch.full((m, m), k * 2.0**-28, dtype=torch.float32)
    actual = torch_catlass.ascend950_basic_syrk(
        x, outDType="fp32", tile=tile, splitk=splitk
    ).cpu()
    lowp = torch_catlass.ascend950_basic_syrk(x, tile=tile, splitk=splitk).cpu().float()
    assert torch.equal(actual, expected)
    assert not torch.equal(lowp, expected)


@only_on_3510
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("fp32_output", [False, True], ids=["lowp", "fp32"])
@pytest.mark.parametrize(
    "batch,m,k,tile,splitk,transposed",
    [
        (None, 64, 129, 64, 1, False),
        (None, 300, 513, 128, 1, True),  # partial diagonal tile, column-major
        (2, 544, 513, 256, 1, False),  # batched diagonal tails
        (1, 384, 1025, 128, 4, True),  # split-K, column-major, uneven K
        (None, 1024, 1024, 256, 0, False),  # automatic split-K
    ],
)
def test_ascend950_basic_syrk_distinct_operands(
    batch, m, k, tile, splitk, transposed, dtype, fp32_output
):
    generator = torch.Generator().manual_seed(3097)
    prefix = () if batch is None else (batch,)
    shape = prefix + ((k, m) if transposed else (m, k))
    a = (torch.randn(shape, generator=generator) * k**-0.5).to(dtype)
    b = (torch.randn(shape, generator=generator) * k**-0.5).to(dtype)
    logical_a = a.transpose(-2, -1) if transposed else a
    logical_b = b.transpose(-2, -1) if transposed else b
    full = logical_a.double() @ logical_b.double().transpose(-2, -1)
    expected = full.tril() + full.tril(-1).transpose(-2, -1)
    x, xt = a.to("npu"), b.to("npu")
    if transposed:
        x, xt = x.transpose(-2, -1), xt.transpose(-2, -1)
    result = torch_catlass.ascend950_basic_syrk(
        x, xt, torch.float32 if fp32_output else dtype, tile, splitk
    ).cpu()
    assert torch.isfinite(result).all()
    assert torch.equal(result, result.transpose(-2, -1))
    if fp32_output:
        torch.testing.assert_close(result.double(), expected, rtol=2e-5, atol=2e-6)
    else:
        torch.testing.assert_close(result.float(), expected.float(), rtol=1e-2, atol=1e-3)


@only_on_3510
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splitk", [1, 4])
def test_ascend950_syrk_polynomial_distinct_operands(dtype, splitk):
    batch, m, k = 2, 384, 513
    x = _random_npu((batch, m, k), dtype, seed=3097) * k**-0.5
    xt = _random_npu((batch, m, k), dtype, seed=7351) * k**-0.5
    residual_source = _random_npu((batch, m, 96), dtype, seed=31) * 96**-0.5
    residual = residual_source @ residual_source.transpose(-2, -1)
    result = torch_catlass.ascend950_syrk_polynomial(
        x, xt, residual, 0.75, -0.25, 1.25, dtype, 128, splitk
    ).cpu()
    full = 0.75 * (x.cpu().double() @ xt.cpu().double().transpose(-2, -1))
    full -= 0.25 * residual.cpu().double()
    full.diagonal(dim1=-2, dim2=-1).add_(1.25)
    expected = full.tril() + full.tril(-1).transpose(-2, -1)
    assert torch.isfinite(result).all()
    assert torch.equal(result, result.transpose(-2, -1))
    torch.testing.assert_close(result.float(), expected.float(), rtol=1e-2, atol=2e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
