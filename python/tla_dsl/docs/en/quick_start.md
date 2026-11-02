---
nav_order: 10
---

# CATLASS Python Quick Start

## Compatibility Requirements

- Supported products: Ascend 950PR&950DT series products
- Operating system: Linux supported by CANN (`aarch64` / `x86_64`)
- CANN version: `>= 9.2.0`
- Python: `>=3.10, <3.14`

## Installation

Official wheel packages are published to PyPI and can be installed directly with pip:

```bash
pip install ascend-catlass-dsl
# Check whether the installation succeeded
python -c "import catlass"
```

> When you run `import catlass` in the `/path/to/catlass/python/tla_dsl` directory, Python imports from the current directory first by default, so it imports the `catlass` module in the source tree rather than the wheel installed in the previous step. The `catlass` in the source tree contains only the Python source code and lacks the corresponding Python binary extension, so switch to another working directory to run the examples.

To try the latest features, you can also follow the [Developer Guide](dsl_development/index.md) to set up the environment and build the source code to compile the wheel yourself.

After installation, precompile the data-movement and other BC templates (about 20 seconds the first time; the result is cached automatically). Use the `CATLASS_DSL_CACHE_DIR` environment variable to set the cache directory; the default is `$HOME/.cache/catlass/bc/`.

```bash
python -m catlass.bc_compile
```

Recompile after changing the wheel package version or the CANN environment; clear the cache or ignore the cache to recompile as needed:

```bash
python -m catlass.bc_compile --clean
# Force: ignore the cache and recompile
python -m catlass.bc_compile --force
```

In most cases, CATLASS Python operators are used together with PyTorch and TorchNPU, so installing them as well is recommended. See [PyTorch Ascend Installation and Deployment](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download), and select and install the TorchNPU version that matches your CANN version.

## Writing the Simplest matmul Operator

The following code implements the simplest matmul operator, using only one cube core and one tile loop, for a quick hands-on experience with CATLASS Python.

```python
# catlass_python_matmul.py
from catlass import tla
from catlass.tla.runtime import from_dlpack
import torch
import torch_npu

M, K, N = 16, 16, 16

@tla.kernel
def matmul_kernel(a: tla.Tensor, b: tla.Tensor, c: tla.Tensor) -> None:
    l0c_data_ready = tla.flag("l0c_data_ready", tla.arch.CUBE, tla.arch.FIX)
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    l1a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(K * N, tla.Float16, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(K * N, tla.Float16, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(M * N, tla.Float32, tla.AddressSpace.l0c, 512)

    with tla.cube():
        # GM -> L1
        l1_a = tla.make_tensor_like(l1a_ptr, a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, b, tla.arch.zN)
        tla.copy(l1_a, a)
        tla.copy(l1_b, b)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        # L1 -> L0A / L0B
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, c, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a)
        tla.copy(l0_b, l1_b)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        # Cube: C = A x B
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        # L0C -> GM
        tla.set_flag(l0c_data_ready)
        tla.wait_flag(l0c_data_ready)
        tla.copy(c, l0_c)

def main() -> int:
    torch.npu.set_device(0)
    torch.manual_seed(0)
    a = torch.rand(M, K, dtype=torch.float16, device="cpu") * 10.0 - 5.0
    b = torch.rand(K, N, dtype=torch.float16, device="cpu") * 10.0 - 5.0
    c = torch.rand(M, N, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    ref = a.float() @ b.float()

    a = a.contiguous().npu()
    b = b.contiguous().npu()
    c = c.contiguous().npu()
    a_tensor = from_dlpack(a, layout_tag=tla.arch.RowMajor, origin_shape=(M, K))
    b_tensor = from_dlpack(b, layout_tag=tla.arch.RowMajor, origin_shape=(K, N))
    c_tensor = from_dlpack(c, layout_tag=tla.arch.RowMajor, origin_shape=(M, N))

    artifact = tla.compile(
        matmul_kernel,
        a_tensor,
        b_tensor,
        c_tensor,
        options="--npu-arch 3510",
    )
    artifact(a_tensor, b_tensor, c_tensor, block_num=1)
    torch.npu.synchronize()

    passed = torch.allclose(c.cpu(), ref)
    print("Passed." if passed else "Failed.")

main()
```

Run:

```bash
python catlass_python_matmul.py
Passed.
```

## Running Repository Examples

After installation, you can also directly run the end-to-end examples in the repository:

```bash
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py --device 0
```

If the output contains `passed=True`, it means the environment and toolchain are ready. For more end-to-end examples, see `examples/python/tla_dsl/tla/kernel/` and `examples/python/tla_dsl/dsl_tutorials/`.
