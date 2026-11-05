# Python 包层

## 包结构

```text
torch_catlass/
├── __init__.py          # 入口：库加载、版本、架构检测
├── _version.py          # 构建时自动生成的版本字符串 (由 build.sh)
├── ops/
│   ├── __init__.py      # 重新导出所有算子
│   ├── basic_matmul.py  # BasicMatmul Python 包装器
│   └── quant_optimized_matmul_tla.py  # 量化 Matmul Python 包装器
```

## 初始化序列

```text
import torch_catlass
  │
  ├── _version.py ──→ os.environ["TORCH_CATLASS_VERSION"]
  │
  ├── _load_kernel_libs()
  │   ├── _find_pkg_dir() ──→ 定位 lib/ 目录
  │   ├── ctypes.CDLL(libcatlass_kernel_jit_compiler.so, RTLD_GLOBAL)
  │   │     └── 提供：JitCompiler 单例、workspace 分配器符号
  │   ├── ctypes.CDLL(libcatlass_kernel_jit.so, RTLD_GLOBAL)
  │   │     └── 提供：JIT 内核入口函数 (BasicMatmul 等)
  │   ├── get_npu_arch() ──→ 2201 或 3510 (设备名 → 架构 ID)
  │   └── ctypes.CDLL(libcatlass_kernel_{arch}_*.so, RTLD_GLOBAL)
  │         └── 提供：预编译的架构相关内核
  │
  └── _load_main_lib()
      └── torch.ops.load_library(libcatlass_torch.so)
            └── 注册 torch.ops.catlass.*
```

## 关键函数

### `get_npu_arch()`

将 TorchNPU 设备名映射到 CATLASS 架构 ID：

| 设备名      | 架构 ID |
| ----------- | ------- |
| Ascend910B* | 2201    |
| Ascend 950*  | 3510    |

无支持设备时抛出 `RuntimeError`。架构 ID 影响：

- 预编译内核选取（加载哪个 `.so`）
- JIT 编译器的 `--npu-arch` 标志
- 模板中的 `CATLASS_ARCH` 预处理器定义

### `enable_mssanitizer()`

设置 `MS_SANITIZE_MEMORY=1`，为后续 JIT 编译启用 Ascend 内存消毒器。库重新加载后恢复普通编译。

### `_load_kernel_libs()`

单次初始化守卫（`_catlass_loaded` 标志）。加载顺序重要：

1. JIT 编译器先加载 — 提供 `JitCompiler` 单例和 workspace 分配器符号
2. JIT 内核入口 — 包含算子分发函数（如 `BasicMatmul`）
3. 预编译架构特定内核 — 架构优化后的 `.so`

所有 `.so` 以 `RTLD_GLOBAL` 加载，确保符号跨库边界可见。

### `_load_main_lib()`

加载 `libcatlass_torch.so`，此库通过 `TORCH_LIBRARY` 注册所有 `torch.ops.catlass.*` 算子。必须在内核库加载后执行。

## Op 包装器

每个算子遵循相同模式：

```python
def operator_name(mat1, mat2, ..., outDType, transA, transB, formatA, formatB) -> Tensor:
    # 1. 规范化 outDType (str → torch.dtype)
    # 2. 委托给 torch.ops.catlass.operator_name(...)
```

它们直接调用 C++ 桥接层，由桥接层分发到 JIT 编译器。
