# Torch Catlass

`tests/optest` 是 CATLASS 示例算子接入 PyTorch 的测试框架。框架将 AscendC/CATLASS kernel 通过 C++ extension 注册为 `torch.ops.catlass.*`，并由 Python 包 `torch_catlass` 提供调用入口。

详细设计见 [docs/design.md](docs/design.md)。

## 架构总览

框架按职责分为五层：

```text
Python API (`torch_catlass.ops.*`)
  -> Python loader (`torch_catlass/__init__.py`)
  -> PyTorch C++ extension (`src/`)
  -> Kernel adapter (`src/include/template/`)
  -> Kernel 实现 (`kernels/`)
```

- Python 层负责用户接口与动态库加载。
- C++ extension 层负责 `torch.ops.catlass.*` 注册和 NPU dispatch。
- adapter 层负责 Tensor 参数到 kernel ABI 参数的转换。
- kernel 层负责 prebuilt kernel 与 JIT kernel 的实现和执行。

## 算子接入清单

### 已接入

- [x] 00_basic_matmul
- [x] 01_batched_matmul
- [x] 02_grouped_matmul_slice_m
- [x] 03_matmul_add
- [x] 04_padding_matmul
- [x] 05_grouped_matmul_slice_k
- [x] 06_optimized_matmul
- [x] 07_grouped_matmul_slice_m_per_token_dequant_moe
- [x] 08_grouped_matmul
- [x] 09_splitk_matmul
- [x] 10_grouped_matmul_slice_m_per_token_dequant
- [x] 11_grouped_matmul_slice_k_per_token_dequant
- [x] 12_quant_matmul
- [x] 13_basic_matmul_tla
- [x] 14_optimized_matmul_tla
- [x] 15_gemm
- [x] 16_group_gemm
- [x] 17_gemv_aiv
- [x] 18_gemv_aic
- [x] 19_mla
- [x] 20_matmul_bias
- [x] 21_basic_matmul_preload_zN
- [x] 22_padding_splitk_matmul
- [x] 23_flash_attention_infer
- [x] 40_flash_attention_infer_tla
- [x] 25_matmul_full_loadA
- [x] 26_matmul_relu
- [x] 27_matmul_gelu
- [x] 28_matmul_silu
- [x] 29_a2_fp8_e4m3_matmul
- [x] 30_w8a16_matmul
- [x] 31_small_matmul
- [x] 32_w4a8_matmul
- [x] 34_single_core_splitk_matmul
- [x] 37_streamk_matmul
- [x] 38_w4a4_per_token_per_channel_dequant
- [x] 41_sparse_matmul_tla
- [x] 42_quant_optimized_matmul_tla
- [x] 43_ascend950_basic_matmul
- [x] 44_quant_matmul_full_loadA_tla
- [x] 45_strided_batched_matmul_tla
- [x] 46_ascend950_matmul_fixpipe_opti (Ascend950)
- [x] 47_ascend950_grouped_slice_m_dequant (Ascend950)
- [x] 48_ascend950_grouped_slice_m_pt_pc_dequant (Ascend950)
- [x] 50_ascend950_basic_matmul_gemv (Ascend950)
- [x] 49_ascend950_flash_attention_infer (Ascend950)
- [x] 51_ascend950_quant_per_group_per_block (Ascend950)
- [x] 52_quant_multi_core_splitk_matmul_tla
- [x] 53_ascend950_fp8_mx_matmul (Ascend950)
- [x] 54_ascend950_fp4_mx_matmul (Ascend950)
- [x] 55_ascend950_mx_grouped_slice_m (Ascend950)
- [x] 57_ascend950_matmul_full_dequant (Ascend950)
- [x] 58_ascend950_fp8_mx_batch_matmul (Ascend950)
- [x] 59_ascend950_a8w4_mx_matmul (Ascend950)
- [x] 60_ascend950_grouped_matmul_slice_m (Ascend950)
- [x] 67_ascend950_batched_matmul (Ascend950)
- [x] 73_ascend950_matmul_full_loadA (Ascend950)
- [x] 74_ascend950_weight_quant_a8w4_grouped_mx_matmul (Ascend950)
- [x] 78_x_attention
- [x] 79_ascend950_matmul_gather_scatter (Ascend950)
- [x] 82_ascend950_basic_syrk (Ascend950)
- [x] 84_ascend950_syrk (Ascend950)

### 暂未接入

- [ ] 24_conv_bias
- [ ] 33_basic_conv2d
- [ ] 56_ascend950_basic_conv2d_tla
- [ ] 70_ascend950_flash_attention_chunk_prefill (Ascend950)
- [ ] 81_ascend950_rain_fusion_attention (Ascend950)
- [ ] 102_dynamic_optimized_matmul
- [ ] 103_dynamic_optimized_quant_matmul_per_token_basic

## 目录结构

```text
tests/optest/
├── CMakeLists.txt
├── build.sh
├── pyproject.toml
├── docs/
│   └── design.md
├── include/
│   ├── catlass_kernel.h
│   ├── catlass_kernel_jit.h
│   └── catlass_kernel_prebuilt.h
├── src/
│   ├── catlass_torch.cpp
│   └── include/common/
├── kernels/
│   ├── 00_basic_matmul/
│   ├── include/
│   └── jit/
├── torch_catlass/
│   ├── __init__.py
│   └── ops/
├── tests/
│   └── test_00_basic_matmul.py
└── utils/
```

## 昇腾与架构识别说明

本项目面向昇腾 NPU 的 CANN/Ascend C 开发栈，相关术语和 SoC 信息以昇腾公开开发文档为准。

- Python loader 侧：通过 `torch_npu.npu.get_device_name()` 映射 arch id，用于加载 `torch_catlass/lib/<arch>/` 下的预构建库。
- JIT 编译侧：通过 `GetCurrentNPUArch()`（AscendC 平台 API）获取当前 SoC 对应架构，用于设置编译参数。
- 当前代码中的映射：
  `Ascend910B.*` / `Ascend910_93` -> `2201`，`Ascend950` -> `3510`。

## 公共 ABI 与接口分层

`include/catlass_kernel.h` 仅作为聚合头，接口按模块拆分为：

- [catlass_kernel_jit.h](include/catlass_kernel_jit.h)：matmul/gemm/gemv 等 JIT 接口，参数语义为 `TParams + Params`。
- [catlass_kernel_prebuilt.h](include/catlass_kernel_prebuilt.h)：flash-attention/conv/mla 等 prebuilt 接口。

其中：

- `TParams` 表示编译期模板参数（dtype/layout/transpose 等）。
- `Params` 表示运行期参数（shape、buffer address 等）。

## 构建与测试

### 环境要求

- Python 3.11+
- PyTorch + TorchNPU（与本机 CANN/驱动版本匹配）
- CMake 3.16+
- 可用昇腾 NPU 设备与环境

### 编译

```bash
bash build.sh
```

### 测试

```bash
pytest tests/ -v
```

## 环境变量

### 外部配置（用户可设置）

| 变量                     | 作用                                                  | 可接受值                      | 默认值                       |
| ------------------------ | ----------------------------------------------------- | ----------------------------- | ---------------------------- |
| `ASCEND_HOME_PATH`       | CANN 安装根目录，查找 compiler(`ccec`) 和 runtime 库  | 绝对路径                      | —（必设）                    |
| `CATLASS_JIT_CACHE_DIR`  | JIT 编译产物 `.so` 磁盘缓存根目录，版本号作为二级目录 | 绝对路径                      | `~/.cache/catlass/jit_cache` |
| `CATLASS_JIT_LOG_LEVEL`  | JIT 编译日志等级                                      | `0`=None, `1`=Info, `2`=Debug | `0`                          |
| `MS_SANITIZE_MEMORY`     | 启用 Ascend memory sanitizer 调试                     | `1`                           | —                            |
| `CATLASS_JIT_AIC_AS_MIX` | 强制 AIC kernel 以 `__mix__(1,0)` 编译                | 任意非空                      | —（默认 `__cube__`）         |
| `CATLASS_JIT_AIV_AS_MIX` | 强制 AIV kernel 以 `__mix__(0,1)` 编译                | 任意非空                      | —（默认 `__vector__`）       |
| `CATLASS_JIT_MIX_CV_11`  | 强制 MIX kernel 以 `__mix__(1,1)` 编译                | 任意非空                      | —（默认 `__mix__(1,2)`）     |
| `PYTHON`                 | `build.sh` 使用的 Python 解释器路径                   | `which python3` 的输出        | `$(which python3)`           |

### 包内注入（import 时自动设置，用户不直接改）

| 变量                  | 作用                                                                           | 来源                                                    |
| --------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------- |
| `CATLASS_JIT_VERSION` | 注入 JIT 编译的版本标识 `-DCATLASS_VERSION_FULL`，同时作为缓存目录的二级目录名 | `torch_catlass/__init__.py` 从 catlass git 推导         |
| `CATLASS_JIT_PKG_DIR` | Python 包安装目录，JIT 依据此路径定位 templates 和 include                     | `torch_catlass/__init__.py` 通过 `_find_pkg_dir()` 解析 |

### 构建选项

| 变量 / 参数                 | 作用                                   | 可接受值                     |
| --------------------------- | -------------------------------------- | ---------------------------- |
| `build.sh --build-type`     | CMake 构建类型                         | `Release`（默认）, `Debug`   |
| `build.sh --skip-wheel`     | 跳过 wheel 打包，使用 editable install | —                            |
| `build.sh --clean`          | 清理 JIT 缓存 + 构建产物               | —                            |
| `CATLASS_ARCH_LIST` (CMake) | 限制 kernel 编译/验证的 NPU 架构（prebuilt 编译 + JIT 模板构建期验证） | 分号分隔列表，如 `2201;3510` |

> prebuilt kernel 默认为每个 arch 编译两份：普通版本和 `_ms` (sanitizer) 版本，无需额外选项。

## 使用示例

```python
import torch
import torch_catlass

a = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
b = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
c = torch_catlass.basic_matmul(a, b, outDType="float16")
```

## 开发指南

### 新增算子流程

1. 在 `include/catlass_kernel_jit.h` 或 `include/catlass_kernel_prebuilt.h` 预留 ABI（带 example 编号说明）。
2. 在 `kernels/` 实现 kernel（JIT template 或 prebuilt）。
3. 在 `src/` 增加 C++ op 适配和注册。
4. 在 `torch_catlass/ops/` 增加 Python wrapper。
5. 在 `tests/` 增加 pytest 用例。
