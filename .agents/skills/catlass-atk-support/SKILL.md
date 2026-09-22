---
name: catlass-atk-support
description: 将 CATLASS 算子接入 ATK 测试框架。当需要把 torch_catlass optest 测试（test_NN_name.py）迁移到 ATK 框架、产出用例定义（NN_name.yaml + generator_NN_name.py + node.yaml）与运行时（execute_NN_name.py）交付物，或在 atk/ 下新增一个 ATK 用例目录时使用本技能。完整的可运行样例见本技能的 template/ 目录（基于 00_basic_matmul）。
metadata:
  depends-on: atk-quality-guard, catlass-example-to-pytest
---
## 依赖加载(必须先做)
开始执行前，先调用 skill 工具加载：
1. `atk-quality-guard` —— **本技能的基类**，权威来源：
   - 精度标准（`mixed_tolerance_bm`）、Tensor 值域分布、Attr 覆盖规则；
   - constraint 约束生成器写法（`after_case_config`）；
   - Bug-Hunting 用例设计（找 bug 而非追求 100% 通过率）、INT32 溢出看护；
   - Phase A（生成交付物）与 Phase B（NPU 精度比对）的执行方式与精度报告判读。
   本技能只补充 CATLASS 专属的 matmul 用例定义 / 运行时写法，凡涉及精度标准、数值约束、
   精度比对标准一律以 `atk-quality-guard` 的 SKILL.md 及其中 `references/experimental_standard.md` 为准，
   **禁止依赖记忆或本文件内硬编码的 rtol/atol 数值**。
   **`standard` 选择规则**：优先 `acc: mixed_tolerance_bm`（预期高版本 ATK）；
   若已装 ATK 的 accuracy 插件不存在该标准（报 `KeyError: '<std>'`），**回退到 `acc: single_bm`**——
   不要回退到 `cv_fused_double_benchmark`。
   `skill({ name: "atk-quality-guard" })`
2. `catlass-example-to-pytest` —— 支持将 `CATLASS` C++ 算子接入到 `optest` 测试框架下（本技能的源测试来源）。
   `skill({ name: "catlass-example-to-pytest" })`

加载后按其内容初始化基础上下文。

> **完整可运行模板**：本技能 `template/` 目录提供基于 `00_basic_matmul` 的 4 个模板文件，
> 是「用例定义 + 运行时」四件套的**规范参考实现**：
> - `template/00_basic_matmul.yaml`（用例规格）
> - `template/generator_00_basic_matmul.py`（generator，helper 已内联，自包含）
> - `template/execute_00_basic_matmul.py`（execute，helper 已内联，自包含）
> - `template/node.yaml`（执行拓扑）
> 下文所有模板代码均以这 4 个文件为准，本 SKILL 不再内联重复；只保留规则、映射表与各族配方。

## 工作流

把一个 CATLASS 算子接入 ATK，固定按下面 8 步执行。先建算子事实表，再决定交付物如何组织，
最后验证与排障。所有模板、映射表与配方见第 3/4 节，不要从头发明。

### 端到端总览

```text
前置：依赖加载（第 0 节：atk-quality-guard + catlass-example-to-pytest）
步骤 1：读源测试 test_NN_name.py → 建立算子事实表
步骤 2：确定命名与注册锚点（NN / name / CamelCaseName / generate）
步骤 3：按家族选型（1.3 决策表）→ 确定 yaml 字段、generator 与 execute 配方
步骤 4：产出 4 个交付物（NN_name.yaml + generator_NN_name.py + execute_NN_name.py + node.yaml）
步骤 5：一致性自检（第 5 节）
步骤 6：atk case 验证生成合法 case
步骤 7：atk task 执行 + 精度比对
步骤 8：按层排障与迭代
```

### 步骤明细

#### 步骤 1：读源测试，建立算子事实表
按顺序识别并记录（这是所有交付物的唯一依据，映射见第 3/4 节）：
- **wrapper 调用**：`torch_catlass.<wrapper>(arg0, arg1, ..., "out_dtype", transA, transB, ...)`
  及**位置参数顺序**；
- **张量输入**：每个 `torch.randn / torch.randint` 的 `name`、形状、dtype、ndim 与取值范围；
- **属性参数**：字符串 `out_dtype`、布尔 `transA/transB/formatA/formatB/use_nz_*` 等；
- **维度关系**：M/K/N（以及 `batch_count` / `group_count` / `scale` / `bias` / `per_token_scale` 形状）之间的依赖；
- **golden 语义**：测试中 `expected = ...` 的计算（参考第 4 节「CPU golden」）。

#### 步骤 2：确定命名与注册锚点
- `NN` = 源测试的两位数字前缀；`name` = 描述性后缀；目录名 == `name` == 文件后缀 `NN_name`。
- `CamelCaseName` = yaml `api_type:` == execute `@register(...)`（如 `executeBasicMatmul`）。
- `generator_NN_name` = yaml `generate:` == generator `@GENERATOR_REGISTRY.register(...)`。
  这三个锚点决定了第 5 节一致性规则能否成立，**先定锚点再写文件**。

#### 步骤 3：按家族选型
依据步骤 1 的事实，从 1.3 决策表确定算子家族，然后直接复用对应族的
yaml 字段 / generator 配方（第 3 节）/ execute 配方（第 4 节）。判定依据必须全部成立，
不确定时先询问用户，不要臆造家族归属。

#### 步骤 4：产出 4 个交付物
依次编写（目录结构见第 2 节；基础 matmul 直接对照 `template/` 四件套改写）：
1. `NN_name.yaml` —— 声明 inputs/attrs（对照 `template/00_basic_matmul.yaml` + 映射表）；
2. `generator_NN_name.py` —— 在 `after_case_config` 修正形状/dtype/属性
   （对照 `template/generator_00_basic_matmul.py`）；
3. `execute_NN_name.py` —— cpu golden + npu wrapper 双分支
   （对照 `template/execute_00_basic_matmul.py`）；
4. `node.yaml` —— 执行拓扑（对照 `template/node.yaml`）。
Grouped 家族额外遵守「GroupedMatmul 固定字段约定」（第 3 节）。

#### 步骤 5：一致性自检
对照第 5 节全部规则逐条核对（name / register / generate / inputs 顺序 / 存储形状）。
任何一条不成立都返回步骤 4 修正。

#### 步骤 6：验证 `atk case`
```bash
atk case -f NN_name.yaml -p generator_NN_name.py
```
确认：yaml 可解析、generator 被 `-p` 加载、生成的 JSON case 全部合法
（无非法 shape / dtype / attr 组合）。发现非法 case → 问题在 yaml 或 generator，回到步骤 4。

#### 步骤 7：执行 `atk task`
```bash
atk pytorch result/NN_name/json/all_NN_name.json --task accuracy --plugin execute_NN_name.py
atk aclnn   result/NN_name/json/all_NN_name.json --task accuracy --plugin execute_NN_name.py
```
或走固定节点拓扑：
```bash
atk task -c result/NN_name/json/all_NN_name.json -n node.yaml --task accuracy -p execute_NN_name.py
```
确认：cpu golden 与 npu result 均执行成功、精度比对通过（CPU 行
「执行成功用例个数 == 总用例个数」）。效率/其它任务时把 `--task accuracy` 换成对应任务名，
并同步调整 `node.yaml` 的 `task:` 与 yaml 的 `standard`。精度比对标准（rtol/atol/匹配率）
按 `atk-quality-guard` 基类执行，不要自行硬编码。

#### 步骤 8：按层排障与迭代
失败时按层定位，**不要表面打补丁**：
1. **环境问题**：torch_npu / CANN / 设备不可见 / 算子 `.so` 未部署；
2. **yaml 或配置问题**：`api_type` / `generate` / `inputs` 名不一致、`standard` 与任务不符；
3. **生成器问题**：跨输入 shape/dtype 依赖未修正、case 非法、内存超限；
4. **执行器或接口适配问题**：wrapper 位置参数顺序、golden dtype 落盘、NZ/转置、int32 累加；
5. **算子语义或基准语义问题**：golden 与测试 `expected` 不一致，需回到步骤 1 核对事实表。

### 家族决策表

| 源测试特征 | 家族 | yaml/generator 配方 | execute 配方 |
|---|---|---|---|
| `A(m,k) @ B(k,n)`，带 transA/transB/formatA/formatB | 基础/optimized（00/06/21） | **template 四件套** | **template execute** |
| 3D 张量 + `bmm` | Batched（01） | 「Batched」配方 | `torch.bmm` |
| matmul + 残差/bias 张量（D / bias） | Epilogue 加张量（03/20） | `assign_matmul_output_shape` / `assign_matmul_bias_shape` | `+ bias.cpu()` 透传 |
| matmul + 激活（relu/gelu/silu） | Epilogue 激活（26/27/28） | 「基础」配方 | 激活函数 |
| int8 输入 + scale/per_token_scale | 量化（12） | `[n]` / `[m]` scale | 先 `.to(int32)` 累加 |
| group_list / groupType / Tiling | Grouped（02/05/08/10/11） | GroupedMatmul 固定字段约定 | `init_by_input_data` 还原 Tiling |

**判定依据（必须全部成立才归入对应族）**：
- 有 `group_list` / `Tiling` / `groupType` → **Grouped**；
- 有 scale + int8 输入 → **量化**；
- 有额外输出/残差/bias 张量 → **Epilogue 加张量**；
- 3D 张量 + `bmm` → **Batched**；
- 否则 → **基础/optimized**。
归属拿不准时先询问用户，不要臆造。

## 交付物总览

将一个 CATLASS 算子接入到 ATK 测试框架中，基于测试件 optest `test_NN_name.py` 转换为一套完整的 ATK 用例，需要同时产出**用例定义**与**运行时**两半：

| 交付物 | 角色 | 模板文件 |
|--------|------|----------|
| `NN_name.yaml` | 用例规格：声明 inputs/attrs（用例定义） | `template/00_basic_matmul.yaml` |
| `generator_NN_name.py` | CaseConfig 后处理器：把随机采样的张量形状/dtype 修正为一次合法算子调用（用例定义） | `template/generator_00_basic_matmul.py` |
| `node.yaml` | 执行拓扑：backends/tasks（用例定义） | `template/node.yaml` |
| `execute_NN_name.py` | `BaseApi` 子类，ATK 对每个用例调用它一次（运行时）：`cpu` 分支算 golden，`npu` 分支调 `torch_catlass.<wrapper>` | `template/execute_00_basic_matmul.py` |

四者缺一不可。`NN` = 与源测试相同的两位数字前缀；`name` = 该测试的描述性后缀。

### 源 / 目标目录结构

源文件是 optest 测试 `test_NN_name.py`（pytest，调用 `torch_catlass.<wrapper>(...)`，由用户提供）。
目标目录是 atk 测试脚本下的 `NN_name/`（目录名 == yaml 中的 `name`），在其中产出 4 个交付物：

```
NN_name/                                 # 目标目录
  ├─ NN_name.yaml                        # 用例规格（见第 3 节，对照 template）
  ├─ generator_NN_name.py                # CaseConfig 后处理器（见第 3 节，对照 template）
  ├─ execute_NN_name.py                  # 运行时 + golden（见第 4 节，对照 template）
  └─ node.yaml                           # backends/tasks（见第 3 节，对照 template）
```

## 用例定义交付物：`NN_name.yaml` + `generator_NN_name.py` + `node.yaml`

把一个 optest `test_NN_name.py` 转换为 ATK 用例的**用例定义**那一半：
`NN_name.yaml`（声明 inputs/attrs）+ `generator_NN_name.py`（对随机采样得到的
`CaseConfig` 做后处理，使张量的形状/dtype 构成一次合法的算子调用）。
运行时那一半（`execute_NN_name.py`）见第 4 节。要得到一个完整用例，始终需要同时产出两者。

### 编写顺序

1. **阅读源测试。** 按顺序识别：
   - wrapper 调用 `torch_catlass.<wrapper>(arg0, arg1, ..., "out_dtype", transA, transB, ...)`；
   - 每个张量输入（`torch.randn` / `torch.randint`）及其形状、dtype、ndim 与取值范围；
   - 每个标量/属性参数（字符串 out_dtype，布尔 transA/transB/formatA/formatB/use_nz_*）；
   - M/K/N（以及 B / G / group_list / scale 形状）之间的关系。
2. **编写 `NN_name.yaml`** —— 每个张量、每个属性对应一个 `inputs` 条目
   （**直接对照 `template/00_basic_matmul.yaml`**，结合下方映射表调整 dtype/shape/属性）。
3. **编写 `generator_NN_name.py`** —— 继承 `CaseGenerator`，注册它，并在
   `after_case_config` 中根据采样得到的维度重新计算形状，应用内存上限，按 transA/transB
   设置存储形状，分配依赖形状（scale/bias/grouped），并对齐依赖的 dtype
   （**直接对照 `template/generator_00_basic_matmul.py`**）。
4. **编写 `node.yaml`**（**对照 `template/node.yaml`**，几乎总是相同）。
5. **对照检查**：完成前，与同族的某个现有兄弟用例对照（00 basic、01 batched、
   03 epilogue-with-extra-tensor、12 quant）。

> **关于 `standard`**：精度标准以 `atk-quality-guard` 基类（`mixed_tolerance_bm`）为准，
> 模板 `template/00_basic_matmul.yaml` 取 `acc: mixed_tolerance_bm` + `perf: not_key`，
> `dtype_numbers: 50`。**不要**硬编码 rtol/atol（按 atk-quality-guard 动态取）。
> **回退规则**：若已装 ATK 的 accuracy 插件不识别 `mixed_tolerance_bm`
> （`atk task` 报 `KeyError: 'mixed_tolerance_bm'`），把 `standard` 改为 `acc: single_bm`；
> **不要**回退到 `cv_fused_double_benchmark`。

### 张量 dtype 映射（测试 dtype → yaml `dtypes.values`）
| 测试中的 torch dtype           | yaml token |
|--------------------------------|------------|
| `torch.float16`                | `fp16`     |
| `torch.bfloat16`               | `bf16`     |
| `torch.float32`                | `fp32`     |
| `torch.int8`                   | `int8`     |
| `torch.int32`                  | `int32`    |

### 属性类型映射（wrapper 标量参数 → yaml）
| 参数                             | `type` | `dtypes.values` | ranges（valid==invalid） |
|----------------------------------|--------|-----------------|--------------------------|
| `out_dtype`（字符串）            | attr   | `string`        | `[ [ "float16", "bfloat16" ] ]`（按测试实际支持，见 template） |
| `transA`/`transB`（bool）        | attr   | `attr_bool`     | `[ false, true ]`        |
| `formatA`/`formatB`/`use_nz_*`   | attr   | `attr_bool`     | `[ false ]`（NZ 通常关闭） |

### 张量取值范围经验法则
- 用 `randn` 采样的浮点激活/权重 → `[ -5, 5 ]`。
- `int8` 量化张量（`randint(-8,8)` / `randint(-127,127)`）→ `[ -127, 127 ]`。
- 正的 scale（`randn(...).abs()*0.1`）→ `[ 0, 0.1 ]`。
- **反量化类算子（dequant，如 `10/11_grouped_matmul_slice_*_per_token_dequant`）的取值范围（覆盖上面两条通用规则）**：
  - `int8` 输入/权重 → `[ -5, 5 ]`（**不要**再用 `[ -127, 127 ]` / `[ -8, 8 ]`）。
  - `bf16` / `fp16` / `fp32` 的 per-channel `scale` 与 per-token `per_token_scale` → `[ -5, 5 ]`（**不要**再用 `[ 0, 0.1 ]`）。
  - `uint64` 张量 → `[ 0, 1 ]`。
- `dim_numbers` = 测试中该张量的 ndim（`(m,k)` 为 2，`(B,m,k)` 或 `(G,k,n)` 为 3，scale/per_token 为 1）。
- 除非该族需要特殊尺寸，否则保持标准的 `dim_values` 列表（见 template：`[1,128,512, 2560,4096,5120,131073, [ 1,65535 ]]`）。

### generator：helper 函数（模板已内联，自包含）

`template/generator_00_basic_matmul.py` 把以下 helper **内联**在文件内，保持 generator 自包含
（实现以 template 为准，**不要**自造变体）：
- `check_memory_limit(m, k, n, dtype)` → 收缩某个轴，直到 `M*K+K*N+M*N` 满足 10 GiB 上限。
- `assign_matmul_storage_shapes(cc, m, k, n, inputs_name="inputs", weights_name="weights")`
  → 设置 `inputs.shape = [k,m] if transA else [m,k]` 以及 `weights.shape = [n,k] if transB else [k,n]`。
  对 A/B 操作数**始终使用此函数**，以使存储布局匹配 kernel 的转置语义。

以下 helper 模板未用到，但各族配方会用到（按同样风格在文件内内联实现）：
- `assign_matmul_output_shape(cc, m, n, output_name="D")` → 用于额外的输出/残差张量 `[m, n]`。
- `assign_matmul_bias_shape(cc, n, bias_name="bias")` → `[n]`。
- `ensure_min_dims(m,k,n, min_m=, min_k=, min_n=, align_k=)` → 为有约束的 kernel 做钳制/对齐。

#### 为什么需要 generator
ATK 对每个张量独立地采样 `dim_values`，因此原始形状**不会**对齐成一个合法的 matmul。
`after_case_config` 会重写它们。`inputs[i]` 的下标与 yaml 中 `inputs:` 的顺序一致；
张量块暴露 `.shape` 与 `.dtype`，属性块暴露 `.range_values`。
模板 generator 用 `case_config.inputs[2].range_values` 随机选一个 `out_dtype`，
再据此对齐两个张量的 dtype（`attr2dtype` 映射）并固定该属性的 `range_values`。

### 各族特定的 generator 配方
- **基础 / 仅 epilogue 的 matmul**（`00 basic`、`06 optimized`、`26/27/28 relu/gelu/silu`）：
  从 inputs[0]/inputs[1] 推导 m,k,n；`assign_matmul_storage_shapes`；对齐 weights dtype；
  固定 `out_dtype` range。**直接对照 `template/generator_00_basic_matmul.py`**。
- **matmul + 额外张量**（`03 matmul_add`、`20 matmul_bias`）：额外调用
  `assign_matmul_output_shape(cc, m, n, output_name="D")`（残差）或
  `assign_matmul_bias_shape(cc, n)`（bias），并设置该张量的 `.dtype = inputs[0].dtype`。
- **Batched**（`01 batched_matmul`）：`b, m, k = inputs[0].shape; n = inputs[1].shape[2]`；
  对 m,k,n 设上限；然后 `inputs[0].shape = [b, m, k]; inputs[1].shape = [b, k, n]`。
- **量化 per-channel/per-token**（`12 quant_matmul`）：推导 m,k,n；设置存储形状；
  `inputs[2].shape = [n]`（scale），`inputs[3].shape = [m]`（per_token_scale）。
- **Grouped / GroupedMatmul**（`02/05/07/08/10/11`）：见下方专门的
  **「GroupedMatmul 固定字段约定」**。generator 固定 A/B 形状、依赖的 scale/per_token 形状，
  并按 08 风格把逐组 `group_list` 写入 `Tiling` 字符串（由 generator 构造、execute 从 `Tiling` 还原）。

## 运行时交付物：`execute_NN_name.py`

把一个 optest `test_NN_name.py` 转换为 ATK 用例的**运行时**那一半：
`execute_NN_name.py`，一个 `BaseApi` 子类，ATK 对每个生成的用例调用它一次。它运行在
**两个 backend** 上：
- `self.device == "cpu"` → 计算 torch **golden**（镜像测试中的 `expected`）。
- `self.device == "npu"` → 调用真实的 `torch_catlass.<wrapper>`（镜像测试中的 `result`）。

用例定义那一半（`NN_name.yaml` + `generator_NN_name.py` + `node.yaml`）见第 3 节。
要得到一个完整用例，始终需要同时产出两者。

> **模板**：完整可运行的 execute 见 `template/execute_00_basic_matmul.py`（自包含：
> `out_dtype_to_torch` / `_OUT_DTYPE_MAP` / `apply_npu_nz_format` 已内联）。

### 源测试如何映射进来

一个典型的 optest 形如：

```python
@only_on_2201
def test_xxx():
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")
    result   = torch_catlass.basic_matmul(a, b, "float16", False, False, False, False)  # -> npu 分支
    expected = torch.matmul(a, b)                                                        # -> cpu golden
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
```

映射关系：
| 源测试元素                                  | execute 交付物元素                                            |
|---------------------------------------------|---------------------------------------------------------------|
| 每个 `torch.randn/randint` 张量             | `input_data.kwargs["<name>"]`（name == yaml 输入名）          |
| 传给 wrapper 的每个标量/布尔参数            | `input_data.kwargs["<attr>"]`（配合 `.get(name, default)`）   |
| `expected = <torch math>`                   | `if self.device == "cpu":` golden 分支                        |
| `result = torch_catlass.<wrapper>(...)`     | `if self.device == "npu":` 分支，**相同的位置参数顺序**       |
| `assert ... allclose`                       | 由 yaml 中的 ATK `standard` 处理 —— 此处**不要** assert        |

### 硬性规则
0. **`__call__` 里所有实际计算逻辑必须显式包裹在 `if self.device == "cpu":` / `if self.device == "npu":` / `if self.device == "gpu":` 之中**（每个分支各自 `return`）。
   **禁止**用隐式 `else`（即 `if cpu: return` 之后裸写 npu 逻辑）来承载 npu 计算——npu 计算也必须显式写成 `if self.device == "npu":`。
   （`gpu` 分支可选，见 `template/execute_00_basic_matmul.py` 的 `cuda:{self.device_id}` 写法。）
1. **`@register("...")` 必须等于 yaml 的 `api_type:`**（例如 `executeBasicMatmul`）。这是 ATK 找到运行时的唯一关联。
2. **kwargs 名字必须与 yaml 的 `inputs:` 名字完全一致**。必需张量用 `kwargs["x"]`，属性用 `kwargs.get("attr", default)`。
3. **npu wrapper 调用必须使用与源测试相同的位置参数顺序。**
4. **不要 assert / 比较** —— ATK 通过 yaml 的 `standard` 比较 cpu-golden 与 npu-result。execute 只返回张量。
5. **两个分支都要返回一个 torch 张量。**
6. **在 npu 分支内部惰性 `import torch_catlass`**（cpu 节点没有 NPU 运行时）。模块顶部 `import torch` 没问题；`import torch_npu` 在需要处惰性导入。
7. 在真实 kernel 调用前后用 `torch.npu.synchronize()` 包裹。
8. **变量命名禁止自造缩写**：从 `kwargs` 取出的张量/属性，其局部变量名必须与 kwarg 名保持一致，
   **不得**改写成缩写。例如 `per_token_scale` **不得**写成 `pts`、`weights` 不得写成 `w`、
   `inputs` 不得写成 `x`、`group_list` 不得写成 `gl`。确需做布局变换的中间量，保留完整基名并加语义后缀
   （如把 `inputs` 由 `(m,k)` 转成 `(k,m)` 命名为 `inputs_km`），**不得**用单字母缩写
   （`template/execute_00_basic_matmul.py` 全程使用 `inputs / weights` 完整名）。
9. **`out_dtype` 字符串 → torch dtype 一律走 `_OUT_DTYPE_MAP`**（见下「out_dtype 映射」），
    **禁止**用 `if / elif` 链或三元表达式逐个判断，也**禁止**硬编码 `torch.float16` 之类字面量。

### 模块 doc-string 约定

每个 `execute_NN_name.py` 的模块级 doc-string 必须包含以下字段（从上至下）：

- **描述**：算子数学表达（如 `A(M_total, K) @ B(G, K, N) -> (M_total, N)`）。
- **特征**：GMM 类必须给出 `groupType` / `groupListType` 取值。
- **约束**：见下方「约束三类」。
- **精度标准**：以 `atk-quality-guard` 基类为准，如 `mixed_tolerance_bm (L2)`；
  已装 ATK 不识别时回退写 `single_bm (L2)`。
- **来源测试**：对应的源测试文件路径（用户提供的 `test_NN_name.py`）。
- **执行**：`atk task -c ... -p execute_NN_*.py -n node.yaml ...` 命令示例。

#### 约束三类（必须按 1-2-3 次序从上至下，用 `# <N ...` 标注类别）

GMM 类 execute 的 `约束` 部分按如下三类、严格 1→2→3 自上而下书写；每条用行尾注释
`# <1 ...` / `# <2 ...` / `# <3 ...` 标明类别：

1. **<1 M/K 分轴约束**：因分轴方式固定下来的属性。
   - slice-m（groupType=0，按 M 分轴）→ `transA = False` / `useNzA = False`（A 为 RowMajor）。
   - slice-k（groupType=2，按 K 分轴）→ `transA = True`（A 为 ColumnMajor）/ `useNzA = False`。
2. **<2 严格模式约束**：如「各组 `Mi`（slice-m）/ `Ki`（slice-k）不能为零」等强制非空约束。
3. **<3 私有约束**：算子特有的其它约束（如 K 对齐、特定 dtype 组合限制等）。

**铁律**：
- **不得虚构不存在的约束项**——某一类没有约束就**不写**该类。
- 三类的先后顺序固定为 1→2→3。
- **对 <2 / <3 是否存在拿不准时，先询问用户，不要臆造。**

### CPU golden —— 精确镜像测试中的 `expected`
在 `.cpu()` 张量上重建测试的参考计算得到 `out`（建议用 `.float()` 累加更稳），
然后**手动**落到输出 dtype：
- `template/execute_00_basic_matmul.py` 的写法：`golden = torch.matmul(a, b)` 后
  `return golden.to(out_dtype_to_torch(out_dtype))`。
- 若走 **benchmark** 任务需保持高精度浮点（不落盘到输出 dtype）——基础 accuracy 用例
  `self.task_result.is_benchmark_task` 恒为 `False`，可省略该判断；需要时按
  `is_benchmark_task` 为真则返回 golden 原值、否则 `golden.to(out_dtype)` 的逻辑内联实现。

- `dtype` = 该用例的输出 torch dtype：
  - 非量化用例（输入即输出 dtype）→ `dtype = inputs.dtype`；
  - 反量化用例（int8 输入、fp16/bf16 输出）→ 取显式输出 dtype，
    **统一用 `_OUT_DTYPE_MAP[out_dtype]` 转换**（见下「out_dtype 映射」），
    **不要**硬编码 `torch.float16` / `torch.bfloat16`，也不要 `if/elif`、三元表达式。
- golden 用**本文件内的本地函数**实现，**不要**导入任何仓库公共模块，
  也**不要**为导入它而 `import os/sys` + `sys.path.insert(...)`。

### out_dtype 映射（统一用 `_OUT_DTYPE_MAP`）

反量化等需要显式输出 dtype 的用例，**必须**在模块顶部定义并复用如下映射（见
`template/execute_00_basic_matmul.py`），把 yaml/generator 固定的 `out_dtype` 字符串转成
torch dtype：

### 量化/反量化 golden 的整数累加（int8@int8 必须先转 int32）

凡是**量化/反量化**类算子（int8 / int4 量化输入，输出 fp16/bf16/fp32），cpu golden 在
做 matmul **之前必须**把 A、B 先 `.to(torch.int32)` 再相乘，以镜像 kernel 的 **int32 累加器**、
避免 int8@int8 直接相乘溢出（`torch.matmul` 对 int8 输入会以 **int8 累加并回绕**）：

```python
inputs = inputs.cpu().to(torch.int32)
weights = weights.cpu().to(torch.int32)
scale = scale.cpu().to(torch.float32)
per_token_scale = per_token_scale.cpu().to(torch.float32)
...
part = torch.matmul(inputs[start:end], weights[i])              # int32 累加, 不溢出
part = part.to(torch.float32) * scale[i] * per_token_scale[start:end].unsqueeze(1)
```

- **不要**用 `.float()` 做整数 GEMM：float32 只有 24-bit 尾数，K 较大时 int32 累加会丢精度。
- `.to(torch.int32)` 后再 `.t()` / `.permute(...)` 切片均可，`torch.matmul` 正常支持 int32（CPU 已验证）。
- **例外**：仅当 kernel **本身就以浮点做 matmul** 时（如 w4a4 fp16 累加、w8a16 fp16 激活、
  fp8 / per-block 预缩放），golden 才按对应浮点 dtype 计算，**不**强行转 int32。

### MX 反量化 golden 的 scale 构造（灾难性抵消看护）

MX 类算子（fp8/fp4 输入 + MX e8m0 scale，如 53/54/55/58/65/71/74 等）的 cpu golden 需要构造
MX scale 做反量化 `dequant = quant * scale`。**scale 必须用「真实 per-block scale」（基于数据
max_abs），禁止用「确定性 2^整数幂 scale」且指数范围过大（如 `2^((arange % 16) - 8)`，最大
`2^7 = 128`）**，否则会触发**灾难性抵消（catastrophic cancellation）**导致精度误判。

**现象**：全量 case 中极少数（通常 1~2 个）元素点报
`存在 N 个元素点的 abs_error 同时大于 max_abs_error_limit 阈值 0.01 和 1000000 倍 ULP 值`。
定位该元素可见其 golden 结果接近 0（如 0.5625），npu 结果差 0.06 左右（如 0.5）。

**根因**：scale 过大（2^7=128）把 dequant 值放大 128 倍，matmul 中间累加量级可达 ±1e6~1e7。
当某个输出元素的结果恰好接近 0（正负项相消）时，fp32 累加的舍入误差（~ulp(中间和) ≈ 0.06~1）
相对接近 0 的结果太大；npu kernel 与 cpu `torch.matmul` 累加顺序不同，该元素即超差。
这是 fp32 数值的固有现象，**不是算子 bug**，但会误判精度。

**触发三条件（同时满足才发生）**：
1. 输入是 fp8（E4M3，3 位尾数）——乘积项尾数位多，累加有舍入；
2. 输出 fp32 且 atol 较小（0.01）——接近 0 的结果容不下 ~0.06 的舍入误差；
3. scale 范围过大（含 2^7 等大值）——放大 dequant 与舍入误差。

**豁免（不触发，或更难触发）**：
- 输出被量化（bf16 / fp8 输出）：抵消误差被输出量化过程淹没（且 atol≥0.0625）。
- atol 足够大（0.1 及以上）：能容纳 ~0.06~1 的抵消误差。
- fp4（E2M1，1 位尾数）输入：乘积项在 fp32 中精确表示，**只有累加有舍入**，比 fp8 更难触发；
  但 **K 较大（如 129）且 scale 过大时仍会触发**（实测 54 算子 k=129 超差 1 个元素）。
  **不能据此豁免 fp4**——只要输入是 fp4 且 scale 用确定性大范围，仍要改真实 per-block scale。

**正确写法（真实 per-block scale，与官方 gen_data.py 一致）**：
```python
_EPS = 1e-12
_FP8_E4M3_EMAX = 8    # E4M3 emax（以 gen_data.py _FP8_FORMATS["E4M3"]["emax"] 为准）

def _per_block_scale(max_abs, emax):
    exp = torch.floor(torch.log2(max_abs.clamp(min=_EPS))) - emax
    exp = torch.where(max_abs < _EPS, torch.zeros_like(exp), exp)
    exp = exp.clamp(-128, 127)
    return torch.exp2(exp)   # 2^整数幂，e8m0 无损表示
```
- 每 32-K block 取 `max_abs`，再 `scale = _per_block_scale(max_abs, emax)`，pad 到 `ks`。
- 真实 scale 最大只有 `2^0=1`，不再放大 dequant；且仍是 2^整数幂，e8m0 无损，
  cpu golden 与 npu kernel 两端一致。
- A 的 scale 沿 K 轴 → 逻辑 `(m, blocks)`；B 的 scale 沿 K 轴 → 逻辑 `(blocks, n)` 或
  `(g, blocks, n)`（grouped）。

**排障口诀**：MX 类算子的精度比对若出现「极少数（1~2 个）元素点超差」且这些元素结果接近 0，
先查 scale 是否是确定性大范围（`2^((arange % 16) - 8)`），是则改为真实 per-block scale。

### NZ / 格式处理
如果算子暴露 `formatA/formatB` 或 `useNzA/useNzB` 属性，则**仅在 npu 分支内联**做 NZ 转换
（实现见 `template/execute_00_basic_matmul.py` 的 `apply_npu_nz_format`）：
`x = torch_npu.npu_format_cast(x, 29)`（按对应 format/useNz 为真时），**不**做物理 permute ——
kernel 会基于原始存储形状来应用转置标志，因此在 NPU 上绝不要在 wrapper 之前 `.permute`。
（在 cpu golden 分支上，转置才用 `.permute(1,0)` 应用，以计算参考值。）
**不要**依赖任何仓库公共模块的 NZ 工具——按 `template/execute_00_basic_matmul.py` 内联即可，
保持 execute 自包含。

### 各族特定的 execute 配方
- **基础 / optimized matmul**（`00`、`06`、`21`、...）：**直接对照 `template/execute_00_basic_matmul.py`**。
- **Epilogue 算子**（`26 relu`、`27 gelu`、`28 silu`、`03 matmul_add`、`20 matmul_bias`）：
  cpu golden = `matmul(...)` 后应用激活（`torch.relu`、GELU 公式
  `x / (1 + exp(-1.595769121 * 0.044715 * inner))`，其中 `inner = x/0.044715 + x*x*x`、
  silu = `x*torch.sigmoid(x)`）或加上残差/bias 张量（`+ bias.cpu()`）；npu 分支把
  额外张量透传给 wrapper。
- **Batched**（`01`）：cpu golden = `torch.bmm(inputs.cpu(), weights.cpu())`；无转置属性。
- **量化 per-channel/per-token**（`12`）：A、B 先 `.to(torch.int32)`，
  cpu `golden = matmul(inputs.cpu().to(torch.int32), weights.cpu().to(torch.int32)).to(torch.float32) * scale.cpu() * per_token_scale.cpu().unsqueeze(1)`（int32 累加防溢出）；
  npu `torch_catlass.quant_matmul(inputs, weights, scale, per_token_scale)`。保持 `out_dtype="float16"`。
- **Grouped**（`02/05/08/10/11`）：`group_list` 由 generator 写入 `Tiling`（第 3 节约定），在
  `init_by_input_data(self, input_data)` 中还原
  `self.group_list = torch.tensor([int(s) for s in kwargs["Tiling"].split("_")], dtype=torch.int64)`。
  然后 cpu golden 逐组循环（按前缀和切分 A，与 `b[g]` 做 matmul，应用 scale），
  npu 分支把 `group_list` 传给 wrapper。

## 一致性规则（必须全部成立）
- `name` == 目录名 == 文件后缀 `NN_name`。
- yaml 中的 `generate:` == `@GENERATOR_REGISTRY.register("...")` 字符串。
- yaml 中的 `api_type:` == execute 交付物中的 `@register("...")`。
- yaml `inputs:` 顺序 == generator 中使用的 `case_config.inputs[i]` 下标 == execute 交付物中
  读取的 kwarg 名字 == `torch_catlass.<wrapper>` 的位置参数顺序。
- 每个作为 matmul 操作数的张量都必须通过 `assign_matmul_storage_shapes` 设置其形状
  （绝不要信任原始采样得到的形状）。

## 可参照拷贝的实现
本技能可参照的实现**只有**自身目录下可见的内容（技能加载时其余仓库路径不可见，禁止引用）：
- `template/00_basic_matmul.yaml` + `template/generator_00_basic_matmul.py` +
  `template/execute_00_basic_matmul.py` + `template/node.yaml` —— 基础 matmul（00/06/21 族）的
  规范参考：最小的 A@B，带 transA/transB/formatA/formatB、NZ 转换、自包含 helper。
- `scripts/gmm_grouplist.py` —— GroupedMatmul group_list 生成工具（内联照抄用）。
- 其它族（Batched/Epilogue/量化/Grouped）无模板文件，按第 3/4 节「各族配方」结合
  `template/` 基础结构改写；拿不准时询问用户。

## 参考资料

- `atk-quality-guard` SKILL.md —— 本技能基类：精度标准 `mixed_tolerance_bm`、constraint 写法、
  Bug-Hunting 设计、INT32 溢出看护、Phase A/B 执行与精度比对判读。
  （`mixed_tolerance_bm` 需高版本 ATK；已装 ATK 不识别时回退 `single_bm`。）
- `template/` —— 基于 00_basic_matmul 的完整可运行模板四件套（规范参考实现）。
- `scripts/gmm_grouplist.py` —— GroupedMatmul group_list 生成工具（内联照抄用）。
