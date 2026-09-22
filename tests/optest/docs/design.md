<!--
This program is free software, you can redistribute it and/or modify.
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
the software repository for the full text of the License.
-->

# torch-catlass 测试框架设计文档

## 总览

`tests/optest` 是 CATLASS 示例算子接入 PyTorch 的端到端测试框架。框架将 CATLASS AscendC kernel 封装为 `torch.ops.catlass.*` 算子，并通过 Python 包 `torch_catlass` 提供测试入口。

框架按职责分为五层：

```text
Python API
  torch_catlass.ops.*
        |
        v
Python package loader
  load kernel libs and libcatlass_torch.so
        |
        v
PyTorch C++ extension
  register torch.ops.catlass.*
        |
        v
Kernel adapter
  convert Tensor arguments to CATLASS kernel params
        |
        v
Kernel implementation
  prebuilt kernel or JIT compiled template
```

核心原则：

- Python 层只负责用户接口、动态库加载和轻量参数转换。
- C++ extension 层负责 PyTorch op 注册和 NPU dispatch。
- adapter 层负责 Tensor 到 kernel ABI 的转换。
- kernel 层负责 AscendC/CATLASS 代码执行。
- JIT 子系统负责模板参数宏生成、编译、缓存和动态加载。

## 目录模块

```text
tests/optest/
├── pyproject.toml
├── CMakeLists.txt
├── build.sh
├── docs/
│   └── design.md
├── include/
│   ├── catlass_kernel.h
│   └── catlass_torch.h
├── torch_catlass/
│   ├── __init__.py
│   ├── _version.py
│   └── ops/
├── src/
│   ├── catlass_torch.cpp
│   ├── common/
│   └── include/
├── utils/
│   ├── CMakeLists.txt
│   ├── include/
│   ├── kernel_utils.cpp
│   ├── torch_utils.cpp
│   └── type_utils.hpp
├── kernels/
│   ├── CMakeLists.txt
│   ├── common/
│   ├── include/
│   ├── jit/
│   └── 00_basic_matmul/
└── tests/
    └── test_00_basic_matmul.py
```

| 模块             | 责任                                                       |
| ---------------- | ---------------------------------------------------------- |
| `torch_catlass/` | Python 包入口、动态库加载、用户侧 op wrapper               |
| `include/`       | 框架和 kernel 共享的公共 ABI 声明                          |
| `src/`           | PyTorch C++ extension 和 torch op 注册                     |
| `utils/`         | dtype/layout/Tensor 工具函数，拆分 torch 依赖和纯 ACL 依赖 |
| `kernels/`       | kernel 构建、JIT compiler、JIT template、kernel entry      |
| `tests/`         | pytest 集成测试                                            |

## Python 包模块

### 包初始化

`torch_catlass/__init__.py` 在 import 时完成运行时初始化：

1. 从 `_version.py` 读取构建期版本信息。
2. 设置 `TORCH_CATLASS_VERSION`，供 JIT 编译阶段注入版本宏。
3. 设置 `TORCH_CATLASS_PKG_DIR`，供 JIT compiler 定位安装后的 headers 和 templates。
4. 加载 JIT compiler 和 JIT kernel entry 库。
5. 根据当前 NPU 架构加载 arch-specific kernel 库。
6. 调用 `torch.ops.load_library()` 加载 `libcatlass_torch.so`，注册 `torch.ops.catlass.*`。

动态库加载顺序：

```text
lib/jit/libcatlass_kernel_jit_compiler.so
lib/jit/libcatlass_kernel_jit.so
lib/<arch>/*.so
lib/libcatlass_torch.so
```

JIT compiler 和 kernel entry 必须先于 PyTorch extension 加载，保证 extension 中引用的 kernel 符号可解析。

### 架构识别

Python loader 通过 `torch_npu.npu.get_device_name()` 识别设备并映射为 CATLASS arch id：

| 设备名                        | arch   |
| ----------------------------- | ------ |
| `Ascend910B.*`                | `2201` |
| `Ascend910_93`                | `2201` |
| `Ascend950PR` / `Ascend950DT` | `3510` |

当 `torch_npu.npu.device_count()` 为 0 时，loader 抛出明确错误。测试代码在无 NPU 环境下通过 pytest skip 处理，避免在 collection 阶段触发 TorchNPU 内部错误。

### Python op wrapper

`torch_catlass/ops/` 保存 Python 用户接口。以 `basic_matmul` 为例：

```python
torch_catlass.basic_matmul(
    mat1,
    mat2,
    outDType="float16",
    transA=False,
    transB=False,
    formatA=False,
    formatB=False,
)
```

Python wrapper 只做用户友好的轻量转换，例如 dtype 字符串白名单解析。shape 推导、输出分配、stream 获取和 kernel launch 都在 C++ 层完成，避免 Python 和 C++ 维护两套语义。

## PyTorch C++ Extension 模块

### 注册入口

`src/catlass_torch.cpp` 是 PyTorch extension 的注册入口：

```cpp
using BasicMatmulOp = MatmulLike<CatlassKernel::BasicMatmul>;
static auto& basic_matmul = BasicMatmulOp::Run;
REGISTER_TORCH_FUNC(basic_matmul);
```

`REGISTER_TORCH_FUNC` 位于 `src/include/common/register.h`，注册流程为：

1. 创建或复用 `torch::Library`，namespace 固定为 `catlass`。
2. 通过 PyTorch schema inference 从 C++ 函数签名生成 schema。
3. 将实现注册到 `c10::DispatchKey::PrivateUse1`。

`PrivateUse1` 是 TorchNPU 使用的 NPU dispatch key。Python 调用 `torch.ops.catlass.basic_matmul` 时，PyTorch 根据输入 Tensor device 走到该 backend 实现。

### kernel launch 包装

`RUN_NPU_FUNC` 位于 `src/include/common/run_npu_func.h`。它通过 TorchNPU 的 `OpCommand::RunOpApiV2` 执行 kernel launch：

- launch 前检查函数指针是否为空。
- 将 C++ 异常转为 ACL error code。
- 将 kernel 调用交给 TorchNPU runtime 管理。

## Matmul Adapter 模块

`src/include/template/matmul.h` 提供 `MatmulLike<KernelFunc>`。该模板封装 matmul 类算子的通用流程：

```text
Run()
  ├─ GetKernelInfo()
  ├─ AllocOutput()
  ├─ get current NPU stream
  ├─ get AIC core count
  └─ RUN_NPU_FUNC(KernelFunc, ...)
```

### 参数拆分

Matmul 参数拆分为两类：

| 参数结构       | 用途                                | 是否参与 JIT 编译 |
| -------------- | ----------------------------------- | ----------------- |
| `TParams`      | dtype、layout、transpose 等模板参数 | 是                |
| `MatmulParams` | M/N/K、输入输出地址等运行时参数     | 否                |

这种拆分保证 dtype/layout 变化会生成新的模板实例，而 shape 和 Tensor 地址变化不会导致重复编译。

### `GetKernelInfo`

`GetKernelInfo()` 负责将 PyTorch Tensor 转为 kernel 参数：

- 将 `torch::Dtype` 转为 `aclDataType`。
- 根据 `transA` 和 `transB` 推导 `m/n/k`。
- 检查两个输入矩阵的 K 维一致。
- 填充 `TParams`。
- 填充 `MatmulParams`。
- 将输入 Tensor storage 地址写入 `params.inputAddr`。

### `AllocOutput`

`AllocOutput()` 根据 `params.m`、`params.n` 和 `tParams.elementC` 创建输出 Tensor，并将其 storage 地址写入 `params.outputAddr[0]`。输出 Tensor 生命周期由 PyTorch 管理，kernel ABI 只接收裸地址。

## 公共 ABI 模块

`include/catlass_kernel.h` 定义 C++ wrapper 和 kernel 实现之间共享的数据结构和函数声明。

### matmul ABI

`TParams` 表示编译期参数：

- `elementA`
- `elementB`
- `elementC`
- `transA`
- `transB`
- `transC`
- `useNzA`
- `useNzB`
- `useNzC`

`MatmulParams` 表示运行期参数：

- `m`
- `n`
- `k`
- `inputAddr`
- `outputAddr`

kernel entry 签名保持固定：

```cpp
void BasicMatmul(
    uint32_t blockNum,
    aclrtStream stream,
    const TParams& tParams,
    const MatmulParams& params);
```

### 扩展 ABI

`catlass_kernel.h` 中还保留了 grouped matmul、quant matmul、conv、flash attention 等参数结构和函数声明。新增算子时优先复用已有 ABI 结构；当参数语义明显不同，再新增独立结构。

## Utils 模块

`utils/` 将工具函数拆成两个 target：

| target                 | 文件               | 依赖                  | 用途                                            |
| ---------------------- | ------------------ | --------------------- | ----------------------------------------------- |
| `catlass_kernel_utils` | `kernel_utils.cpp` | ACL                   | JIT compiler 使用的 dtype 到 bisheng type 转换  |
| `catlass_torch_utils`  | `torch_utils.cpp`  | ACL、torch、torch-npu | PyTorch wrapper 使用的 Tensor/dtype/layout 工具 |

### dtype 映射

`utils/type_utils.hpp` 维护 dtype 映射表：

- canonical string name
- `torch::Dtype`
- `aclDataType`
- bisheng C++ type token

映射表按 index 对齐，`TypeCast<S, T>()` 通过查表完成不同表示之间的转换。

部分 dtype 在 TorchNPU 随包 ACL 头中没有枚举名，但 ABI 数值与 CANN 定义一致。此类 dtype 使用 `static_cast<aclDataType>(value)` 表达，避免混用系统 CANN ACL 头和 TorchNPU 随包 ACL 头导致重复定义。

### Tensor 工具

`torch_utils.cpp` 提供：

- `GetOutputTensor()`：在当前 NPU device 上创建 ND 格式输出 Tensor。
- `TypeStrToTorchDtype()`：字符串到 torch dtype。
- `TorchDtypeToAclDtype()`：torch dtype 到 ACL dtype。
- `AclDtypeToTorchDtype()`：ACL dtype 到 torch dtype。
- `GetTransposeStatus()`：根据 tensor stride 和 NPU format 判断矩阵布局。

## JIT 子系统

JIT 子系统由四部分组成：

| 组件            | 文件                                            | 责任                                      |
| --------------- | ----------------------------------------------- | ----------------------------------------- |
| JIT entry       | `kernels/00_basic_matmul/basic_matmul.cpp`      | 稳定 kernel 入口，负责获取 JIT 函数并调用 |
| JIT template    | `kernels/00_basic_matmul/basic_matmul_impl.cpp` | 被运行时编译的 CATLASS kernel 模板        |
| JIT compiler    | `kernels/jit/jit_compiler.cpp`                  | 编译、缓存、加载 `.so`                    |
| macro generator | `kernels/include/jit_macro_generator.h`         | 将模板参数转为 `-D` 宏                    |

### JIT entry

JIT entry 固定编译进 `libcatlass_kernel_jit.so`。以 `BasicMatmul` 为例：

```cpp
auto* entry = JitCompiler::instance().getKernel(
    "basic_matmul_impl.cpp",
    JitMacroGenerator<TParams>::generate("basic_matmul", tParams));
if (entry) {
    entry(blockNum, stream, &params);
}
```

entry 的职责是连接 stable ABI 和 runtime-compiled template，不承载具体 GEMM 模板逻辑。

### JIT template

JIT template 使用宏注入类型和布局：

| 宏                        | 语义                 |
| ------------------------- | -------------------- |
| `CATLASS_JIT_ELEMENT_A`   | A 元素类型           |
| `CATLASS_JIT_ELEMENT_B`   | B 元素类型           |
| `CATLASS_JIT_ELEMENT_C`   | C 元素类型           |
| `CATLASS_JIT_LAYOUT_A`    | A layout             |
| `CATLASS_JIT_LAYOUT_B`    | B layout             |
| `CATLASS_JIT_LAYOUT_C`    | C layout             |
| `CATLASS_JIT_KERNEL_NAME` | device kernel 符号名 |

模板导出稳定 C ABI：

```cpp
extern "C" void run(
    uint32_t blockNum,
    aclrtStream stream,
    const CatlassKernel::MatmulParams* params);
```

JIT loader 固定解析 `run` 符号。device kernel 名只用于编译产物可读性和 profiling。

template 内部通过 `Catlass::RunKernel<Kernel>(arguments, stream, blockNum)` 启动内核（来自 `common/kernel_runner.h`），不使用 `device_gemm.hpp`。

### 宏生成

`JitMacroGenerator<TParams>` 是模板策略类。默认模板不生成任何宏，具体参数类型通过特化实现。

`JitMacroGenerator<TParams>` 生成：

- `CATLASS_KERNEL_NAME`
- `CATLASS_JIT_ELEMENT_A`
- `CATLASS_JIT_ELEMENT_B`
- `CATLASS_JIT_ELEMENT_C`
- `CATLASS_JIT_LAYOUT_A`
- `CATLASS_JIT_LAYOUT_B`
- `CATLASS_JIT_LAYOUT_C`
- `CATLASS_JIT_KERNEL_NAME`

新增非 matmul JIT kernel 时，应新增对应参数结构的 `JitMacroGenerator` 特化。

### 编译和缓存

`JitCompiler` 是进程级单例。初始化内容包括：

- JIT cache 目录。
- bisheng/ccec 路径。
- 当前 NPU arch。
- JIT template 根目录。

缓存分两层：

1. 内存缓存：`loaded_` 保存 `SharedLib` 和 `run` 指针。
2. 磁盘缓存：保存编译后的 `.so`。

cache key 通过 SHA256 对 (key=value&) 拼接串做哈希生成 UUID，文件名格式为 `{uuid}.so`。编译命令中所有参数通过单引号 `shellQuote()` 转义后由 `popen` 调度 bisheng 执行，防止 `__mix__(1,2)` 等宏值中的特殊字符被 shell 误解析。

宏按 key 排序后拼接，保证 `unordered_map` 遍历顺序不影响缓存路径。

### 环境变量

| 环境变量                  | 作用                                                                                   | 可接受值                      | 默认值                       |
| ------------------------- | -------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------- |
| `ASCEND_HOME_PATH`        | 查找 Ascend compiler (`ccec`) 和 runtime 库                                            | 绝对路径                      | —（必设）                    |
| `TORCH_CATLASS_CACHE_DIR` | JIT 编译产物 `.so` 磁盘缓存目录                                                        | 绝对路径                      | `~/.cache/catlass/jit_cache` |
| `CATLASS_JIT_LOG_LEVEL`   | JIT 编译日志等级                                                                       | `0`=None, `1`=Info, `2`=Debug | `0`                          |
| `MS_SANITIZE_MEMORY`      | 启用 Ascend memory sanitizer 调试；设为 `1` 时 JIT 编译器追加 `--cce-enable-sanitizer` | `1`                           | —                            |
| `CATLASS_JIT_AIC_AS_MIX`  | 强制 AIC kernel 以 `__mix__(1,0)` 发射（覆盖默认 `__cube__`）                          | 任意非空                      | —                            |
| `CATLASS_JIT_AIV_AS_MIX`  | 强制 AIV kernel 以 `__mix__(0,1)` 发射（覆盖默认 `__vector__`）                        | 任意非空                      | —                            |
| `CATLASS_JIT_MIX_CV_11`   | 强制 MIX kernel 以 `__mix__(1,1)` 发射（覆盖默认 `__mix__(1,2)`）                      | 任意非空                      | —                            |
| `TORCH_CATLASS_VERSION`   | 注入 `-DCATLASS_VERSION_FULL` 到 JIT 编译                                              | 包内自动设置                  | catlass git describe         |
| `TORCH_CATLASS_PKG_DIR`   | JIT 依据此路径定位 `jit/templates/` 和 include                                         | 包内自动设置                  | `_find_pkg_dir()` 解析       |

JIT 编译使用的 NPU arch 只通过 AscendC platform API 获取，即 `GetCurrentNPUArch()`。运行时不支持通过环境变量覆盖 arch，避免编译参数和实际设备不一致。

环境变量分为两类：

- **外部配置**：`ASCEND_HOME_PATH`、`TORCH_CATLASS_CACHE_DIR`、`CATLASS_JIT_LOG_LEVEL`、`MS_SANITIZE_MEMORY`、`CATLASS_JIT_{AIC,AIV,MIX}_*` — 用户按需设置。
- **包内注入**：`TORCH_CATLASS_VERSION`、`TORCH_CATLASS_PKG_DIR` — 由 Python loader 在 import 时自动设置，用户不直接修改。

## Kernel 构建模块

`kernels/CMakeLists.txt` 提供 `add_kernel()`，统一 JIT 和 prebuilt kernel 的构建入口。

### JIT kernel

```cmake
add_kernel(
    NAME basic_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/basic_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/basic_matmul_impl.cpp)
```

JIT kernel 构建流程：

1. entry 源文件加入统一 `libcatlass_kernel_jit.so`。
2. template 文件安装到 `jit/templates/`。
3. `jit_verify_template()` 在构建期检查 template 可被 bisheng 编译。
4. 运行时由 `JitCompiler` 根据模板参数编译具体 `.so`。

### prebuilt kernel

prebuilt kernel 按 arch 构建独立动态库：

```text
lib/<arch>/libcatlass_kernel_<arch>_<name>.so
```

prebuilt 模式用于固定参数组合或无需运行时模板编译的 kernel。默认每个 arch 同时编译普通版本和 `_ms` (sanitizer) 版本，无需额外选项。

## 顶层构建模块

### Python 构建入口

`build.sh` 是主要构建入口：

- 推导 package 版本。
- 写入 `torch_catlass/_version.py`。
- 自动探测当前 Python 环境中的 Torch CMake 目录。
- 调用 pip/scikit-build-core 驱动 CMake 构建。

常用命令：

```bash
bash build.sh --skip-wheel
bash build.sh --build-type Debug --skip-wheel
bash build.sh --clean
```

### CMake target

顶层 `CMakeLists.txt` 负责：

- 查找 ASC、Python、Torch。
- 设置 C++17、PIC、compile commands。
- 安装 public headers。
- 安装 CATLASS headers 到 Python 包内的 JIT include tree。
- 安装 `kernels/common/` 头文件（`kernel_runner.h`、`common.h`、`tile_shape_scaler.h`、`workspace_alloc.h` 等）到 `jit/common/`，供 JIT 运行时编译使用。
- 添加 `kernels`、`utils`、`src` 子目录。

关键 target：

| target                        | 输出       | 说明                             |
| ----------------------------- | ---------- | -------------------------------- |
| `catlass_kernel_utils`        | static lib | JIT compiler 依赖的纯 ACL 工具   |
| `catlass_torch_utils`         | static lib | torch wrapper 依赖的 Tensor 工具 |
| `catlass_kernel_jit_compiler` | shared lib | JIT 编译器                       |
| `catlass_kernel_jit`          | shared lib | JIT entry 集合                   |
| `catlass_torch`               | shared lib | PyTorch extension                |

## 测试模块

pytest 集成测试验证 Python API 到 kernel 执行的完整链路。

`tests/test_00_basic_matmul.py` 测试流程：

1. 检查是否存在可用 Ascend NPU；无设备时跳过。
2. 构造 NPU fp16 输入。
3. 调用 `torch_catlass.basic_matmul()`。
4. 用 `torch.matmul()` 生成参考结果。
5. 校验 shape、dtype、device。
6. 用 `torch.allclose()` 校验数值。

本地静态检查：

```bash
python3 -m py_compile torch_catlass/__init__.py torch_catlass/ops/basic_matmul.py tests/test_00_basic_matmul.py
python3 -m ruff check torch_catlass tests
```

完整集成测试：

```bash
bash build.sh --skip-wheel
pytest tests/test_00_basic_matmul.py -v -s
```

## 扩展流程

### 新增 matmul 类 JIT 算子

1. 在 `kernels/<nn_name>/` 下添加 entry `.cpp` 和 template `.cpp`。
2. 在 entry 中调用 `JitCompiler::instance().getKernel()`。
3. 复用或扩展 `JitMacroGenerator<TParams>`。
4. **在 template 中使用 `Catlass::RunKernel<Kernel>()`（来自 `common/kernel_runner.h`）启动 kernel，禁止引入 `catlass/gemm/device/device_gemm.hpp`。**
5. 模板若引用 `kernels/common/` 下头文件，确认已在顶置 `CMakeLists.txt` 安装到 `jit/common/`。
6. 在 `kernels/<nn_name>/CMakeLists.txt` 中调用 `add_kernel()`。
7. 在 `include/catlass_kernel.h` 声明 kernel entry。
8. 在 `src/catlass_torch.cpp` 使用 `MatmulLike<Kernel>` 注册 torch op。
9. 在 `torch_catlass/ops/` 添加 Python wrapper。
10. 在 `tests/` 添加 pytest，与 PyTorch 参考实现比对。

### 新增非 matmul 算子

非 matmul 算子应新增独立 adapter，而不是扩展 `MatmulLike`：

```text
src/include/template/<op_family>.h
  ├─ GetKernelInfo()
  ├─ AllocOutput()
  └─ Run()
```

同时新增对应的参数结构和 `JitMacroGenerator` 特化，保持参数解析、宏生成和 kernel ABI 各自独立。

### 新增 dtype 或 layout

新增 dtype/layout 时需要同步更新：

- `utils/type_utils.hpp`
- `JitMacroGenerator` 对应特化
- JIT template 中的默认宏和类型别名
- pytest 参数覆盖

dtype 映射应优先使用当前编译环境可见的 enum；当 TorchNPU 随包 ACL 头未暴露 enum 名但 ABI 数值稳定时，可使用 `static_cast<aclDataType>(value)` 保持兼容。
