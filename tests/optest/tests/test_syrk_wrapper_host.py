"""Wrapper-only checks: load the module without importing a device runtime."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def wrapper():
    path = Path(__file__).parents[1] / "torch_catlass/ops/ascend950_basic_syrk.py"
    spec = importlib.util.spec_from_file_location("syrk_wrapper_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", ["fp32", "float32", "FP32", torch.float32])
def test_syrk_fp32_wrapper_dispatch(wrapper, monkeypatch, dtype):
    calls = []
    marker = object()

    def invoke(*args):
        calls.append(args)
        return marker

    monkeypatch.setattr(torch.ops.catlass, "ascend950_basic_syrk", invoke, raising=False)
    x = torch.empty((32, 64), dtype=torch.float16)
    assert wrapper.ascend950_basic_syrk(x, outDType=dtype, tile=128, splitk=4) is marker
    assert calls == [(x, x, torch.float32, 128, 4)]


def test_syrk_default_dtype_unchanged(wrapper, monkeypatch):
    monkeypatch.setattr(torch.ops.catlass, "ascend950_basic_syrk", lambda *args: args, raising=False)
    x = torch.empty((32, 64), dtype=torch.bfloat16)
    args = wrapper.ascend950_basic_syrk(x)
    assert args[0] is x and args[1] is x
    assert args[2:] == (torch.bfloat16, 0, 0)


@pytest.mark.parametrize("dtype", ["fp16", "bfloat16", torch.float16])
def test_syrk_polynomial_wrapper_dispatch(wrapper, monkeypatch, dtype):
    calls = []
    marker = object()

    def invoke(*args):
        calls.append(args)
        return marker

    monkeypatch.setattr(torch.ops.catlass, "ascend950_syrk_polynomial", invoke, raising=False)
    x = torch.empty((32, 64), dtype=torch.float16)
    residual = torch.empty((32, 32), dtype=torch.float16)
    assert wrapper.ascend950_syrk_polynomial(x, x, residual, 1.5, -0.25, 2.0, dtype, 64, 2) is marker
    expected_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    assert calls == [(x, x, residual, expected_dtype, 1.5, -0.25, 2.0, 64, 2)]
