---
nav_order: 10
---

# CATLASS Python 快速开始

## 兼容性要求

- 支持产品：Ascend 950PR&950DT系列产品
- 操作系统：CANN 支持的 Linux（`aarch64` / `x86_64`）
- CANN 版本：`>= 9.2.0`
- Python：`>=3.10, <3.14`

## 安装

正式版本的 wheel 包会上传到 PyPI，可直接通过 pip 安装：

```bash
pip install ascend-catlass-dsl
# 检查是否安装成功
python -c "import catlass"
```

> 在 `/path/to/catlass/python/tla_dsl` 目录下执行 `import catlass` 时，Python 默认优先从当前目录导入，命中的是源码树中的 `catlass`，而非上一步安装的 wheel。源码树中的 `catlass` 仅包含 Python 部分源码，缺少对应的 Python 二进制扩展，因此运行示例需切换到其他工作目录。

如需体验最新特性，也可参考[开发者指南](dsl_development/index.md)完成环境配置与源码构建，自行编译 wheel 包。

安装完成后需预编译数据搬运等 BC 模板（首次约 20 秒，结果自动缓存）；缓存目录可通过 `CATLASS_DSL_CACHE_DIR` 环境变量指定，默认值为 `$HOME/.cache/catlass/bc/`。

```bash
python -m catlass.bc_compile
```

更换 wheel 包版本或 CANN 环境后需重新编译，可按需清理缓存或忽略缓存重新编译：

```bash
python -m catlass.bc_compile --clean
# 强制：忽略缓存，重新编译
python -m catlass.bc_compile --force
```

多数情况下，CATLASS Python 算子会结合 PyTorch 和 TorchNPU 使用，因此建议同时安装。可参考[PyTorch Ascend 安装部署](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download)，选择并安装所需的、与当前 CANN 版本兼容的 TorchNPU 版本。

## 编写一个最简单的 matmul 算子

下列代码实现了一个最简单的 matmul 算子，仅使用一个 cube 核、一次 tile 循环，用于快速体验 CATLASS Python。

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

执行：

```bash
python catlass_python_matmul.py
Passed.
```

## 运行仓库示例

安装完成后，也可以直接运行仓库内的端到端示例：

```bash
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py --device 0
```

输出包含 `passed=True` 即表示环境与工具链就绪。更多端到端示例见 `examples/python/tla_dsl/tla/kernel/` 与 `examples/python/tla_dsl/dsl_tutorials/`。
