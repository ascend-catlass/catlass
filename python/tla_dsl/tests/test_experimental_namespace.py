"""Experimental APIs are explicitly opt-in, not stable aliases."""

import importlib

import pytest

import catlass.tla as tla


def test_experimental_exports():
    experimental = importlib.import_module("catlass.tla.experimental")
    assert tla.experimental is experimental
    assert "experimental" in dir(tla)
    assert set(experimental.__all__) == {
        "img2col",
        "Img2colParams",
        "height",
        "width",
        "mx_to",
        "exp_sub",
    }
    assert callable(experimental.img2col)
    assert callable(experimental.exp_sub)
    assert callable(experimental.mx_to)
    params = experimental.Img2colParams((8, 8), (3, 3))
    assert params.stride_hw == (1, 1)
    assert params.pad_value == 0
    assert params.repeat_load_stride == 0
    assert params.repeat_load_time == 1
    assert params.repeat_load_axis is experimental.height


@pytest.mark.parametrize(
    "module", ["catlass.tla", "catlass.runtime", "catlass.core_api"]
)
@pytest.mark.parametrize(
    "name",
    [
        "im2col",
        "Im2colParams",
        "img2col",
        "Img2colParams",
        "height",
        "width",
        "mx_to",
        "exp_sub",
    ],
)
def test_experimental_api_is_not_stable(module, name):
    namespace = importlib.import_module(module)
    assert name not in getattr(namespace, "__all__", ())
    assert name not in dir(namespace)
    with pytest.raises(AttributeError):
        getattr(namespace, name)


def test_experimental_has_no_old_spelling():
    assert not hasattr(tla.experimental, "im2col")
    assert not hasattr(tla.experimental, "Im2colParams")
    assert not hasattr(tla.experimental, "LoadRepeatParams")
