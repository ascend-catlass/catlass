"""BMM broadcast metadata checks and numerical coverage, including HF32."""
import itertools
import math

import pytest
import torch
import torch_catlass

from common import only_on_3510

op = torch_catlass.ascend950_bmm_broadcast
PRECISIONS = [(torch.float16, False), (torch.bfloat16, False),
              (torch.float32, False), (torch.float32, True)]
REDUCTION_CASES = [
    (dtype, hf32, k)
    for dtype, hf32 in PRECISIONS
    for k in (17, 128, 1024, 3071, 3072, 3073, 4096, 7168, 16385, 32769, 65536)
]


def round_hf32(tensor):
    bits = tensor.contiguous().float().view(torch.int32).long() & 0xFFFFFFFF
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & ~0x1FFF
    return rounded.int().view(torch.float32)


def reference(a, b, bias, trans_a, trans_b, hf32):
    aa, bb = a.cpu().float(), b.cpu().float()
    if hf32:
        aa, bb = round_hf32(aa), round_hf32(bb)
    if trans_a:
        aa = aa.transpose(-1, -2)
    if trans_b:
        bb = bb.transpose(-1, -2)
    result = aa.double() @ bb.double()
    if bias is not None:
        result = result + bias.cpu().double().reshape(-1, 1, result.size(-1))
    return result.to(a.dtype)


def assert_accuracy(actual, expected, dtype, hf32):
    # Original ATK MixedToleranceV2 thresholds and its independent all-element guard.
    tolerance = {(torch.float16, False): 0.001, (torch.bfloat16, False): 0.0078,
                 (torch.float32, False): 0.00012, (torch.float32, True): 0.001}[dtype, hf32]
    absolute_limit = 0.01 if dtype == torch.float32 and not hf32 else 0.1
    actual, expected = actual.cpu(), expected.cpu()
    assert torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item()
    x, y = actual.float(), expected.float()
    assert torch.isclose(x, y, rtol=tolerance, atol=tolerance).float().mean().item() >= 0.99
    error = (x - y).abs()
    spacing = (torch.nextafter(expected, torch.full_like(expected, float("inf"))) - expected).float().abs()
    assert torch.all((error <= absolute_limit) | (error <= 32 * spacing)).item(), (
        f"dtype={dtype}, hf32={hf32}, max_abs={error.max().item()}"
    )


def check_case(dtype, hf32, trans_a, trans_b, ba, bb, mode,
               shape=(67, 131, 145), batch=3, offset=False, repeat=False):
    m, n, k = shape
    generator = torch.Generator().manual_seed(88)

    def tensor(shape):
        value = torch.empty(shape).uniform_(-5, 5, generator=generator).to(dtype).npu()
        if offset:
            storage = torch.empty(value.numel() + 1, dtype=dtype, device="npu")
            view = storage[1:].view(shape)
            view.copy_(value)
            return view
        return value

    a = tensor((ba, k, m) if trans_a else (ba, m, k))
    b = tensor((bb, n, k) if trans_b else (bb, k, n))
    bias = None
    if mode == 1:
        bias = tensor((n,))
    elif mode == 2:
        bias = tensor((batch, n))
    expected = reference(a, b, bias, trans_a, trans_b, hf32).expand(batch, m, n)
    result = op(a, b, batch, bias, trans_a, trans_b, hf32)
    assert result.shape == (batch, m, n)
    assert result.dtype == dtype and result.device == a.device and result.is_contiguous()
    assert_accuracy(result, expected, dtype, hf32)
    if repeat:
        torch.testing.assert_close(op(a, b, batch, bias, trans_a, trans_b, hf32), result, rtol=0, atol=0)


@only_on_3510
@pytest.mark.parametrize("dtype,hf32", PRECISIONS)
@pytest.mark.parametrize("trans_a,trans_b", itertools.product([False, True], repeat=2))
@pytest.mark.parametrize("ba,bb", [(1, 1), (1, 3), (3, 1), (3, 3)])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_combinations(dtype, hf32, trans_a, trans_b, ba, bb, mode):
    check_case(dtype, hf32, trans_a, trans_b, ba, bb, mode)


@only_on_3510
@pytest.mark.parametrize("dtype,hf32", PRECISIONS)
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 17, 7), (64, 128, 256), (129, 257, 513)])
def test_boundaries_and_storage_offset(dtype, hf32, shape):
    check_case(dtype, hf32, True, True, 1, 1, 2, shape, batch=1, offset=True, repeat=True)


@only_on_3510
@pytest.mark.parametrize("dtype,hf32", PRECISIONS)
@pytest.mark.parametrize("n", [128, 129, 131])
@pytest.mark.parametrize("mode", [1, 2])
def test_bias_tail_with_storage_offset(dtype, hf32, n, mode):
    batch, m, k = 3, 3, 17
    a = torch.ones((1, m, k), dtype=dtype, device="npu")
    b = torch.ones((batch, k, n), dtype=dtype, device="npu")
    shape = (n,) if mode == 1 else (batch, n)
    count = math.prod(shape)
    storage = torch.empty(count + 1, dtype=dtype, device="npu")
    bias = storage[1:].view(shape)
    values = ((torch.arange(count) % 11) - 5).reshape(shape).to(dtype)
    bias.copy_(values)
    expected = (k + values).reshape(-1, 1, n).expand(batch, m, n)
    result = op(a, b, batch, bias, hf32=hf32)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(bias.cpu(), values, rtol=0, atol=0)


@only_on_3510
@pytest.mark.parametrize("dtype,hf32,k", REDUCTION_CASES)
@pytest.mark.parametrize("trans_a,trans_b,ba,bb,mode", [
    (False, False, 1, 3, 0), (True, False, 3, 1, 1),
    (False, True, 1, 1, 2), (True, True, 3, 3, 2),
])
def test_large_k_exact_tail_broadcast_and_bias(dtype, hf32, k, trans_a, trans_b, ba, bb, mode):
    # Small exact integer sums test full-K execution, not arbitrary-input precision.
    batch, m, n = 3, 3, 7
    signs = (1 - 2 * (torch.arange(k) % 2)).to(dtype)
    a = signs.reshape(1, 1, k).expand(ba, m, k).contiguous().npu()
    b = torch.ones((bb, k, n), dtype=dtype, device="npu")
    a[:, :, -1] *= 2
    b[:, -1, :] = 3
    bias = None
    if mode:
        shape = (n,) if mode == 1 else (batch, n)
        bias = ((torch.arange(math.prod(shape)) % 11) - 5).reshape(shape).to(dtype).npu()
    if trans_a:
        a = a.transpose(-1, -2).contiguous()
    if trans_b:
        b = b.transpose(-1, -2).contiguous()
    expected = reference(a, b, bias, trans_a, trans_b, hf32).expand(batch, m, n)
    result = op(a, b, batch, bias, trans_a, trans_b, hf32)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(op(a, b, batch, bias, trans_a, trans_b, hf32), result, rtol=0, atol=0)


@only_on_3510
@pytest.mark.parametrize("dtype,hf32", PRECISIONS)
@pytest.mark.parametrize("trans_a,trans_b", itertools.product([False, True], repeat=2))
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_full_k_tail_and_bias_once(dtype, hf32, trans_a, trans_b, mode):
    batch, m, n, k = 3, 65, 129, 17
    a = torch.ones((1, m, k), dtype=dtype, device="npu")
    b = torch.ones((batch, k, n), dtype=dtype, device="npu")
    a[:, :, -1] = 2
    b[:, -1, :] = 3
    bias = None
    if mode:
        shape = (n,) if mode == 1 else (batch, n)
        bias = ((torch.arange(math.prod(shape)) % 11) - 5).reshape(shape).to(dtype).npu()
    if trans_a:
        a = a.transpose(-1, -2).contiguous()
    if trans_b:
        b = b.transpose(-1, -2).contiguous()
    expected = reference(a, b, bias, trans_a, trans_b, hf32).expand(batch, m, n)
    result = op(a, b, batch, bias, trans_a, trans_b, hf32)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(op(a, b, batch, bias, trans_a, trans_b, hf32), result, rtol=0, atol=0)


@only_on_3510
@pytest.mark.parametrize("dtype,hf32", PRECISIONS)
@pytest.mark.parametrize("pattern", ["blocked", "interleaved", "reverse"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_small_k_cancellation_diagnostics(dtype, hf32, pattern, mode):
    # Keep adversarial signs within [-5,5], including the small retained term.
    for k in range(1, 18):
        values = torch.tensor([5.0, 2**-14, -5.0, 0.0]).repeat((k + 3) // 4)[:k]
        if pattern == "blocked":
            values = values.sort(descending=True).values
        elif pattern == "reverse":
            values = values.flip(0)
        a = values.reshape(1, 1, k).expand(1, 63, k).contiguous().to(dtype).npu()
        b = torch.full((1, k, 1), 5.0, dtype=dtype, device="npu")
        bias = None
        if mode == 1:
            bias = torch.tensor([2**-14], dtype=dtype, device="npu")
        elif mode == 2:
            bias = torch.tensor([[-5.0], [0.0], [5.0]], dtype=dtype, device="npu")
        result = op(a, b, 3, bias, hf32=hf32)
        expected = reference(a, b, bias, False, False, hf32).expand(3, 63, 1)
        assert_accuracy(result, expected, dtype, hf32)


@only_on_3510
def test_hf32_switch_and_quantized_reference():
    a = torch.full((1, 1, 1), 2.0009765625, dtype=torch.float32, device="npu")
    b = a.clone()
    bias = torch.tensor([-4.00390720367431640625], dtype=torch.float32, device="npu")
    strict = op(a, b, 1, bias)
    approximate = op(a, b, 1, bias, hf32=True)
    assert_accuracy(strict, reference(a, b, bias, False, False, False), torch.float32, False)
    assert_accuracy(approximate, reference(a, b, bias, False, False, True), torch.float32, True)
    assert torch.any(strict != approximate).item()
    torch.testing.assert_close(op(a, b, 1, bias), strict, rtol=0, atol=0)
    torch.testing.assert_close(op(a, b, 1, bias, hf32=True), approximate, rtol=0, atol=0)


@only_on_3510
@pytest.mark.parametrize("dtype,hf32", PRECISIONS)
def test_subnormal_inputs_against_original_tolerance(dtype, hf32):
    k = 17
    exponent = {torch.float16: -24, torch.bfloat16: -133, torch.float32: -149}[dtype]
    a = torch.full((1, 2, k), 2.0**exponent, dtype=dtype, device="npu")
    b = torch.full((1, k, 3), 5.0, dtype=dtype, device="npu")
    a[:, :, -1] *= -1
    expected = reference(a, b, None, False, False, hf32)
    assert_accuracy(op(a, b, 1, hf32=hf32), expected, dtype, hf32)


@only_on_3510
@pytest.mark.parametrize("dtype,hf32", PRECISIONS)
@pytest.mark.parametrize("name", ["A", "B", "Bias"])
@pytest.mark.parametrize("value", [8.0, -8.0])
def test_values_outside_atk_default_range(dtype, hf32, name, value):
    a = torch.ones((1, 2, 3), dtype=dtype, device="npu")
    b = torch.ones((1, 3, 2), dtype=dtype, device="npu")
    bias = torch.zeros(2, dtype=dtype, device="npu")
    {"A": a, "B": b, "Bias": bias}[name].flatten()[0] = value
    expected = reference(a, b, bias, False, False, hf32)
    torch.testing.assert_close(op(a, b, 1, bias, hf32=hf32).cpu(), expected, rtol=0, atol=0)


@only_on_3510
@pytest.mark.parametrize("invalid", ["batch", "zero_batch", "huge_batch", "k", "dtype", "bias_shape", "bias_dtype", "stride", "rank", "empty", "hf32", "bias_stride"])
def test_reject_invalid_arguments(invalid):
    a = torch.ones((3, 2, 4), dtype=torch.float16, device="npu")
    b = torch.ones((1, 4, 5), dtype=torch.float16, device="npu")
    batch, bias, hf32 = 3, None, False
    if invalid == "batch":
        batch = 2
    elif invalid == "zero_batch":
        batch = 0
    elif invalid == "huge_batch":
        batch = 2**32
    elif invalid == "k":
        b = b[:, :3, :].contiguous()
    elif invalid == "dtype":
        b = b.float()
    elif invalid == "bias_shape":
        bias = torch.ones((3, 1, 5), dtype=a.dtype, device="npu")
    elif invalid == "bias_dtype":
        bias = torch.ones(5, dtype=torch.float32, device="npu")
    elif invalid == "stride":
        a = a.transpose(1, 2)
    elif invalid == "rank":
        a = a[0]
    elif invalid == "empty":
        a = a[:, :0, :]
    elif invalid == "hf32":
        hf32 = True
    elif invalid == "bias_stride":
        bias = torch.ones(10, dtype=a.dtype, device="npu")[::2]
    with pytest.raises(RuntimeError):
        op(a, b, batch, bias, hf32=hf32)
