---
name: "catlass-example-to-pytest"
description: "Integrate numbered CATLASS examples from examples/ into tests/optest with consistent ABI, torch.ops registration, Python wrappers, and pytest verification."
---

# Catlass Example to Pytest Skill

将 `examples/` 下带编号算子接入 `tests/optest/` 测试框架，产出可构建、可导入、可测试的完整链路：

```text
example source
  -> kernel entry (jit/prebuilt)
  -> ABI declaration (include/catlass_kernel_*.h)
  -> torch C++ adapter (src/)
  -> python wrapper (torch_catlass/ops/)
  -> pytest (tests/)
```

## Scope

- 工作目录：`tests/optest/`
- 输入：`examples/<nn>_<name>/`
- 输出：
  - 头文件预留接口（按模块）
  - C++/Python 调用链接入
  - pytest 用例
  - README / docs 必要更新（仅与接入相关）

## Non-negotiable Rules

1. 参数语义统一使用 `TParams` + `Params`。
2. `matmul/gemm/gemv` 归入 JIT 接口头：`include/catlass_kernel_jit.h`。
3. 其他（如 conv / flash-attention / mla）归入 prebuilt 接口头：`include/catlass_kernel_prebuilt.h`。
4. 不大量引入 `using XxxParams = ...` 去伪装兼容；优先使用统一结构体或结构体继承表达兼容关系。
5. grouped quant 场景避免把运行时 dtype 再塞回 `Params`；dtype 应放在 `QuantTParams` 这类模板参数结构。
6. 架构识别逻辑遵循现有实现：JIT 侧从 `GetCurrentNPUArch()` 获取，不新增环境变量分支。
7. 任何已有用户改动不可回滚；发现冲突先停并问用户。

## Phase 0: Intake & Classification

### 解析 example

- 读取：
  - `examples/<dir>/CMakeLists.txt`
  - `examples/<dir>/README*`（若存在）
  - 主 `.cpp` / `.h`
- 提取：
  - 算子家族：`matmul/gemm/gemv` 或 `other`
  - 入口函数名（建议 PascalCase C++ + snake_case Python）
  - 参数特征：是否包含 bias / quant / grouped / rope / mask / workspace

### 分类规则

- dirname 含 `matmul|gemm|gemv` -> JIT 类
- dirname 含 `conv|flash_attention|mla|fai` -> prebuilt 类
- 不确定时：
  - 先按源码依赖（是否强依赖模板宏生成/JIT）判断
  - 仍不明确则问用户

### 重名检测

检查以下是否已存在：

- `include/catlass_kernel_jit.h` / `include/catlass_kernel_prebuilt.h` 中同名声明
- `src/catlass_torch.cpp` 中同名注册
- `torch_catlass/ops/<name>.py`
- `tests/test_<nn>_<name>.py` 或 `tests/test_<name>.py`

重复时默认"增量更新不覆盖"，除非用户明确要求覆盖。

### 自动提取 kernel 类型

从 example 的 `CMakeLists.txt` 读取 `cube`/`mix`，自动填入入口 cpp：

```bash
python scripts/gen_entry.py <nn> <name>
```

对应关系：`cube` → `JitKernelType::AIC`，`mix` → `JitKernelType::MIX`。

## Automation Scripts

`scripts/` 目录提供代码生成，减少手工模板填充：

| Script | 作用 | 用法 |
|--------|------|------|
| `gen_entry.py` | 生成 JIT 入口 cpp，自动读取 example CMakeLists 的 kernel type | `python scripts/gen_entry.py <nn> <name> [--macros none\|scheduler\|apply_opt]` |
| `gen_cmake.py` | 生成 kernel CMakeLists.txt | `python scripts/gen_cmake.py <nn> <name>` |
| `gen_python.py` | 生成 Python wrapper | `python scripts/gen_python.py <nn> <name>` |
| `gen_test.py` | 生成 pytest | `python scripts/gen_test.py <nn> <name> [--transB] [--padding] [--small]` |

### gen_entry 宏模式

`--macros` 控制入口是否带条件宏：

| 模式 | 适用 | 示例 |
|------|------|------|
| `none`（默认） | 无运行时条件分支 | 21, 31, 37, 39 |
| `scheduler` | swizzle 方向依 m>n | 13, 14, 25 |
| `apply_opt` | ApplyOptMacros（scheduler 数字 "30"/"31" + padding） | 04, 06 |

### 接入修改清单（自动生成外需手动）

1. `kernels/CMakeLists.txt`：补 `add_subdirectory(<nn>_<name>)`
2. `src/catlass_torch.cpp`：补 `MatmulLike<...>::Run` + `REGISTER_TORCH_FUNC`
3. `torch_catlass/ops/__init__.py`：补 import + `__all__`
4. `README.md`：更新接入清单
5. `<name>_impl.cpp`：**模板须手动编写**（遵循原始 example kernel 结构）

## Phase 1: ABI Reservation First

先预留 ABI，再写实现，避免后续接口漂移。

### JIT ABI（matmul family）

在 `include/catlass_kernel_jit.h`：

- 复用或扩展统一结构体：
  - `TParams`
  - `MatmulParams`
  - `GroupedMatmulParams`
  - `QuantTParams`
  - `GemmParams`
- 添加函数声明，并写明 example 编号和目录名的 docstring：
  - `@brief Reserved JIT interface for example <nn>_<name>.`

### Prebuilt ABI（others）

在 `include/catlass_kernel_prebuilt.h`：

- 复用或扩展统一结构体：
  - `PrebuiltParams`
  - `ConvParams`
  - `FlashAttentionParams`
  - `MlaParams`
- 添加函数声明并保留编号 docstring。

### 聚合头

- `include/catlass_kernel.h` 只做聚合 include，不承载具体参数定义。

## Phase 2: Kernel Integration

### JIT 类接入

在 `kernels/<nn>_<name>/` 创建或更新：

1. `<name>.cpp`：JIT 入口（调用 `JitCompiler::instance().getKernel(...)`）
2. `<name>_impl.cpp`：模板实现，支持 `CATLASS_JIT_*` 宏
3. `CMakeLists.txt`：`add_kernel(... jit ... TEMPLATE ...)`

要求：

- 模板默认宏齐全（dtype/layout/transpose 等）。
- 运行入口签名与 ABI 声明一致。
- **JIT 模板中 kernel 启动统一使用 `Catlass::RunKernel<Kernel>()`（来自 `common/kernel_runner.h`），禁止引入 `catlass/gemm/device/device_gemm.hpp`。**
- **L1/L0 TileShape 统一使用 `CatlassKernel::TileShapeScaler<ElementA, half, BaseShape>::type`（或 TLA 变体 `TileShapeScalerTLA`），确保元素类型变化时 K 维度按字节宽度等比缩放。不硬编码 `GemmShape<...>`。**
- **⚠️ 累加器类型（CType）不可用 `CATLASS_JIT_ELEMENT_C` 宏推导**：Split-K/Padding 等 kernel 的 CType 是累加器类型（如 `float`），而 `CATLASS_JIT_ELEMENT_C` 由 adapter 的 `outDType` 设置，会覆盖为 `half`。必须用 `#define CATLASS_JIT_ELEMENT_C float`（非 `#ifndef`）固定，防止 JIT 宏覆盖导致 workspace 步长错乱产生 inf。模板修改后须清理 JIT 缓存（`rm -rf ~/.cache/catlass/jit_cache/`），否则旧 `.so` 仍被命中。
- **`CATLASS_JIT_BLOCK_SCHEDULER` 使用数字编码：末位=direction，前位=offset。`GemmIdentityBlockSwizzle<3, 0>` → 默认值 `30`。模板中通过 `(CATLASS_JIT_BLOCK_SCHEDULER / 10)` 取 offset、`(CATLASS_JIT_BLOCK_SCHEDULER % 10)` 取 direction。**
- 模板引用 `kernels/common/` 下的头文件（如 `kernel_runner.h`、`tile_shape_scaler.h`、`common.h`、`workspace_alloc.h`）时，需确认对应头文件已通过顶置 `CMakeLists.txt` 安装到 `jit/common/`。
- **Device 内存分配统一使用 workspace 全局函数**（`#include "common/workspace_alloc.h"`），详见 [WORKSPACE.md](WORKSPACE.md)：
  - kernel 写入的 buffer → `g_catlassWorkspaceAlloc(size)`
  - 需 H2D 的 struct 数组 → `g_catlassWorkspaceAllocFromHost(hostData, size)`
  - **禁止** `aclrtMalloc` + `aclrtMemcpy`（torch tensor 内存不支持）、**禁止** `memset`（device 内存不能 host memset）
  - **无需手动 free**，workspace pool 自动管理
- **Padding 路径需为 `deviceWA`/`deviceWB` 单独分配缓冲区**，通过 `g_catlassWorkspaceAlloc`，不可复用原始 `deviceA`/`deviceB` 指针。模板变更后须清除 JIT 缓存目录（`torch_catlass.clear_jit_cache()`），否则旧 `.so` 仍被命中。
- 不把 example 的命令行/数据生成逻辑带入内核模板。

### Prebuilt 类接入

在 `kernels/<nn>_<name>/`：

- 使用 prebuilt 方式编译并导出与 ABI 一致的入口函数。
- 若需 workspace，参数和实际 launch 必须一致（shape、dtype、flags 全对齐）。

### 构建入口

在 `kernels/CMakeLists.txt` 补 `add_subdirectory(<nn>_<name>)`（保持编号顺序）。

## Phase 3: Torch Adapter & Python Wrapper

### C++ 注册

在 `src/catlass_torch.cpp`：

- JIT matmul 类：优先复用 `MatmulLike<...>` 或同类通用适配器。
- 非 matmul：按参数语义新增轻量 adapter。
- 使用既有 `REGISTER_TORCH_FUNC(...)` 注册到 `torch.ops.catlass.*`。

### Python 包装

在 `torch_catlass/ops/<name>.py`：

- 参数名与 C++ adapter 对齐。
- dtype 字符串解析保持白名单/显式映射，不使用隐式 `eval`。
- 填充 `TParams` 使用直接 map 访问（如 `tParams.element["A"] = ...`）。
- `transA`/`transB`/`useNzA`/`useNzB` 全部声明为 `bool = False`，不再使用 `Optional`。
  调用 `torch.ops.catlass.*` 时直接传参。
- docstring 写清：
  - 来源 example 编号名称
  - 参数语义
  - 输出语义

并在 `torch_catlass/ops/__init__.py` 导出。

## Phase 4: Tests

### pytest 用例

新增 `tests/test_<nn>_<name>.py`（建议保留编号，避免重名）。

最小断言：

1. shape 正确
2. dtype 正确
3. device 为 NPU
4. 与基线（`torch.matmul` 或 reference kernel）数值对齐（给定 `rtol/atol`）

### 无设备处理

- 用 `pytest.mark.skipif(torch_npu.npu.device_count() <= 0, ...)` 检查是否有可用的NPU卡。
- 根据算子适用架构，用`@only_on_2201`或者`@only_on_3510`做隔离，确保测试脚本只在支持的硬件环境下工作。
- 跳过只用于环境不可用，不用于掩盖接口错误。

## Phase 5: Verify & Fix Loop

每次接入至少执行：

1. `bash build.sh`（必要时 `--skip-wheel`）
2. `pytest tests/test_<nn>_<name>.py -v`
3. `python -c "import torch_catlass"`

失败处理：

- 编译错误：先修 ABI/签名/CMake 路径问题。
- 导入错误：先查 `torch_catlass/__init__.py` 动态库加载顺序与符号。
- 运行错误：核对 `TParams/Params` 填充及 dtype/layout 映射。
- JIT 编译报 `Syntax error: "(" unexpected`：检查编译器参数是否含 shell 特殊字符（如 `__mix__(1,2)`），`RunProcessCapture` 已通过单引号转义处理，新增宏值若含特殊字符需验证转义生效。

## Checklists

### Interface Checklist

- [ ] example 分类正确（jit or prebuilt）
- [ ] 头文件声明落在正确模块（jit/prebuilt）
- [ ] docstring 含 example 编号名称
- [ ] 参数模型与 `TParams + Params` 保持一致
- [ ] 无不必要 `using XxxParams` 别名滥用
- [ ] Python 接口 `transA/B`/`useNzA/B` 声明为 `bool = False`（非 `Optional[bool]`）
- [ ] JIT 模板使用 `common/kernel_runner.h` 启动 kernel，未引入 `device_gemm.hpp`
- [ ] L1/L0 TileShape 使用 `TileShapeScaler`（非硬编码）
- [ ] `CATLASS_JIT_BLOCK_SCHEDULER` 使用数字编码（末位 direction，前位 offset）
- [ ] ⚠️ 累加器 CType 须用 `#define` 固定（非 `#ifndef`），防止 JIT 宏覆盖
- [ ] 模板依赖的 `kernels/common/` 头文件已通过 CMake install rule 安装到 `jit/common/`
- [ ] 内存分配仅用 `g_catlassWorkspaceAlloc` / `g_catlassWorkspaceAllocFromHost`，无裸 `aclrtMalloc`/`aclrtMemcpy`/`memset`/`aclrtFree`
- [ ] 需 H2D 的 struct 数组用 `g_catlassWorkspaceAllocFromHost`，纯 workspace 用 `g_catlassWorkspaceAlloc`

### Runtime Checklist

- [ ] `torch.ops.catlass.<name>` 可见
- [ ] `import torch_catlass` 成功
- [ ] pytest 通过或因无 NPU 合理 skip
- [ ] 与参考实现数值一致
- [ ] JIT 编译无 shell 转义错误（`__cube__`/`__mix__` 等含特殊字符的宏值正确编译）

## Output Contract

完成后必须给出：

1. 修改文件列表（按模块分组）
2. ABI 变更摘要（新增/复用了哪些 `TParams/Params`）
3. 构建与测试结果（命令 + 结论）
4. 若有环境阻塞，明确区分“代码问题”与“环境问题”
