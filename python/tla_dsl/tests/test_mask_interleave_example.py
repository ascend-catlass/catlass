from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


@pytest.fixture
def mask_interleave_example(monkeypatch):
    import importlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "examples/end_to_end/vector_ops"))
    return importlib.import_module("mask_interleave_op")


@pytest.mark.parametrize("op_name", ["mask_interleave", "mask_deinterleave"])
@pytest.mark.parametrize("dtype_name", ["i8", "f16", "f32"], ids=["b8", "b16", "b32"])
def test_example_cpu_reference(mask_interleave_example, op_name, dtype_name, monkeypatch) -> None:
    # CPU golden checks must not initialize optional torch device plugins.
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    torch = pytest.importorskip("torch")
    example = mask_interleave_example
    _, dtype, _ = example._set_kernel_config(op_name, dtype_name, (512,))
    a = torch.full((512,), 3, dtype=dtype)
    b = torch.full((512,), -2, dtype=dtype)
    low, high = example.HARNESS.config.expected(op_name, (a, b))
    lanes = example.VL_ELE
    expected_high = b.reshape(-1, lanes).clone()
    if op_name == "mask_interleave":
        # H / NOT Q: low starts alternating, then all true; high alternates.
        expected_low = a.reshape(-1, lanes).clone()
        expected_low[:, 1:lanes // 2:2] = -2
        expected_high[:, 1::2] = 3
    else:
        # Check the reference with explicit output intervals rather than
        # repeating its concatenate/slice algorithm. Odd input boundaries
        # put one extra true lane in the first half of the even output,
        # and one extra true lane in the second half of the odd output.
        expected_low = b.reshape(-1, lanes).clone()
        expected_low[:, :lanes // 4] = 3
        expected_low[:, lanes // 2 + lanes // 8 + 1:] = 3
        expected_high[:, :lanes // 4 - 1] = 3
        expected_high[:, lanes // 2 + lanes // 8:] = 3
    assert torch.equal(low, expected_low.flatten())
    assert torch.equal(high, expected_high.flatten())
    assert not torch.equal(low, high)
    for output in (low, high):
        chunks = output.reshape(-1, lanes)
        assert bool((chunks == 3).any(dim=1).all())
        assert bool((chunks == -2).any(dim=1).all())


@pytest.mark.parametrize("op_name", ["mask_interleave", "mask_deinterleave"])
@pytest.mark.parametrize("dtype_name", ["i8", "f16", "f32"], ids=["b8", "b16", "b32"])
def test_example_compiles_to_kernel_object(mask_interleave_example, op_name, dtype_name, request) -> None:
    import os
    from pathlib import Path

    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("CANN environment required: source set_env.sh")
    request.getfixturevalue("isolated_compile_cache")
    example = mask_interleave_example
    dtype, _, _ = example._set_kernel_config(op_name, dtype_name, (512,))
    tensor = make_fake_tensor(
        dtype, (512,), (1,), addrspace=tla.AddressSpace.gm,
        origin_shape=(512,), layout_tag=tla.arch.RowMajor,
    )
    artifact = tla.compile(
        example.mask_interleave_op, tensor, tensor, tensor, tensor,
        options="--npu-arch 3510",
    )
    binary = Path(artifact.kernel_binary_path)
    assert binary.is_file()
    assert binary.stat().st_size > 0


@pytest.mark.parametrize("op_name", ["mask_interleave", "mask_deinterleave"])
@pytest.mark.parametrize("dtype_name", ["i8", "f16", "f32"], ids=["b8", "b16", "b32"])
def test_example_on_npu(mask_interleave_example, op_name, dtype_name, request, monkeypatch) -> None:
    import os
    import subprocess
    import sys

    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("CANN environment required: source set_env.sh")
    # Import the optional NPU backend explicitly so a missing driver library
    # is reported as a skip, rather than a PyTorch plugin-autoload failure.
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not torch.npu.is_available():
        pytest.skip("No accessible Ascend NPU")
    request.getfixturevalue("isolated_compile_cache")
    # Run the real harness in a child process so a device failure has a bounded
    # timeout. Its exit status includes comparisons of both output tensors.
    result = subprocess.run(
        [sys.executable, "-u", mask_interleave_example.__file__, op_name,
         "--dtype", dtype_name, "--shape", "512", "--device",
         os.environ.get("CATLASS_TEST_NPU_DEVICE", "0")],
        capture_output=True, text=True, timeout=180,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "launch_ok=True" in result.stdout
    assert f"outputs equal expected {op_name}? True" in result.stdout
