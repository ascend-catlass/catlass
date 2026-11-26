"""Host-side shape contracts for the Mixed examples."""

import argparse
from functools import lru_cache
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("catlass.tla")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

EXAMPLES = (
    Path(__file__).resolve().parents[2]
    / "examples/python/tla_dsl/tla/kernel/matmul/basic_mixed"
)
DEFAULTS = {
    "basic_mixed": (32, 32, 32),
    "basic_mixed_mutex": (32, 32, 32),
    "basic_mixed_fixpipe_nz2dn": (60, 100, 64),
    "basic_mixed_store_zN": (60, 64, 256),
    "basic_mixed_store_zNUnAlign": (60, 64, 256),
    "basic_mixed_ub2l1": (32, 32, 32),
    "basic_mixed_fixpipe_per_tensor": (128, 128, 64),
    "basic_mixed_fixpipe_per_channel": (128, 128, 64),
    "basic_mixed_l0c2ub_b8_unitflag_pad": (33, 44, 64),
}


@lru_cache
def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(shape):
    return argparse.Namespace(
        m=shape[0],
        n=shape[1],
        k=shape[2],
        dtype="f32",
        dtype_ab="f32",
        dtype_c="f16",
        layout_c="col",
        device=0,
    )


class DeviceSetupReached(Exception):
    pass


def stop_at_device_setup(*args, **kwargs):
    raise DeviceSetupReached


@pytest.mark.parametrize("name", DEFAULTS)
@pytest.mark.parametrize("axis", range(3), ids=("m", "n", "k"))
@pytest.mark.parametrize("value", (0, -1))
def test_nonpositive_dimensions_are_rejected_before_device_setup(
    monkeypatch, name, axis, value
):
    monkeypatch.setattr(torch.npu, "set_device", stop_at_device_setup)
    shape = list(DEFAULTS[name])
    shape[axis] = value
    with pytest.raises(ValueError, match="positive"):
        load_example(name).run(arguments(shape))


@pytest.mark.parametrize(
    "name,shape",
    [
        ("basic_mixed", (16, 32, 32)),
        ("basic_mixed", (32, 32, 33)),
        ("basic_mixed_mutex", (32, 16, 32)),
        ("basic_mixed_fixpipe_nz2dn", (60, 99, 64)),
        ("basic_mixed_fixpipe_nz2dn", (60, 100, 65)),
        ("basic_mixed_store_zN", (31, 64, 256)),
        ("basic_mixed_store_zN", (60, 65, 256)),
        ("basic_mixed_store_zNUnAlign", (66, 64, 256)),
        ("basic_mixed_store_zNUnAlign", (60, 64, 257)),
        ("basic_mixed_ub2l1", (31, 32, 32)),
        ("basic_mixed_ub2l1", (50, 32, 32)),
        ("basic_mixed_ub2l1", (256, 256, 512)),
        ("basic_mixed_fixpipe_per_tensor", (128, 128, 65)),
        ("basic_mixed_fixpipe_per_channel", (128, 128, 65)),
        ("basic_mixed_l0c2ub_b8_unitflag_pad", (65, 44, 64)),
    ],
)
def test_invalid_shapes_are_rejected_before_device_setup(monkeypatch, name, shape):
    monkeypatch.setattr(torch.npu, "set_device", stop_at_device_setup)
    with pytest.raises(ValueError):
        load_example(name).run(arguments(shape))


@pytest.mark.parametrize(
    "name,shape",
    [
        ("basic_mixed", (32, 32, 17)),
        ("basic_mixed_mutex", (32, 32, 17)),
        ("basic_mixed_fixpipe_nz2dn", (60, 100, 31)),
        ("basic_mixed_store_zN", (50, 31, 128)),
        ("basic_mixed_store_zNUnAlign", (64, 1, 1)),
        ("basic_mixed_ub2l1", (64, 32, 32)),
        ("basic_mixed_fixpipe_per_tensor", (129, 133, 31)),
        ("basic_mixed_fixpipe_per_channel", (129, 133, 31)),
        ("basic_mixed_l0c2ub_b8_unitflag_pad", (33, 35, 31)),
    ],
)
def test_supported_shapes_reach_device_setup(monkeypatch, name, shape):
    monkeypatch.setattr(torch.npu, "set_device", stop_at_device_setup)
    with pytest.raises(DeviceSetupReached):
        load_example(name).run(arguments(shape))
