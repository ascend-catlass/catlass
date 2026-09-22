# 精度问题定位

## 写在前面

该文档主要说明在CATLASS样例开发过程中，当精度比对不通过时，如何系统性地定位精度问题的根因。文档首先帮助开发者判断问题类型，然后通过决策树和模块化二分法逐步缩小问题范围，最后结合症状-原因速查表和诊断模式找到具体原因并修复。

阅读本文档前，请先阅读[精度分析基础](./precision_analysis_basics.md)，了解CATLASS的精度比对方式和Golden函数使用方法。

## 精度问题分类

在开始定位之前，首先需要判断当前精度问题属于哪一类。不同类别的问题，排查路径完全不同。

### 完全算错

**特征**：NPU输出与标杆结果差异巨大，几乎所有元素都不匹配，或输出呈现明显的异常模式（全0、全NaN、全Inf、随机乱值等）。

**常见原因**：

- 数据搬运未完成就开始计算（流水线同步缺失）
- 输入数据未正确传入（内存拷贝错误、地址偏移错误）
- 样例逻辑存在根本性错误（公式写错、维度搞反）
- 编译缓存未清理，实际运行的是旧版本二进制

**排查优先级**：先清理缓存，然后排查流水线同步和数据搬运问题，再排查逻辑错误。

### 精度误差

**特征**：大部分元素通过比对，仅少数元素超出误差阈值；或整体误差偏大但数值趋势正确（如输出与标杆在同一个数量级）。

**常见原因**：

- 低精度数据类型（FP16/BF16）的舍入误差累积
- 中间计算未使用升精度累加
- 数值溢出（FP16最大值仅65504）
- 减法抵消（两个相近数相减导致有效位丢失）
- 特定API的精度特性与预期不符

**排查优先级**：先检查累加器精度和溢出情况，再排查特定API行为。

### 快速判断方法

运行精度比对后，观察`CompareData`返回的错误索引数量和分布：

| 现象                                                       | 判断              | 下一步                                                             |
| ---------------------------------------------------------- | ----------------- | ------------------------------------------------------------------ |
| 错误数接近总元素数，或输出全0/NaN/Inf                      | 完全算错          | 进入[前置检查](#%E5%89%8D%E7%BD%AE%E6%A3%80%E6%9F%A5) → [流水线同步缺失排查](#%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%90%8C%E6%AD%A5%E7%BC%BA%E5%A4%B1) |
| 错误数占总元素数比例较小（如 < 10%），且错误值在合理范围内 | 精度误差          | 进入[诊断模式](#%E8%AF%8A%E6%96%AD%E6%A8%A1%E5%BC%8F)                                             |
| 错误集中在特定位置（如矩阵边缘、特定分组）                 | 边界/分组处理问题 | 进入[模块化二分定位](#%E6%A8%A1%E5%9D%97%E5%8C%96%E4%BA%8C%E5%88%86%E5%AE%9A%E4%BD%8D)                                 |

## 前置检查

在深入排查之前，先完成以下前置检查项。这些检查成本低但能排除大量常见问题。

### 验证标杆代码正确性

**这是CATLASS精度调试的第一步，也是最容易被忽略的一步。** 如果标杆本身就错了，后续所有比对都没有意义。

检查要点：

1. **标杆函数选择是否正确**：确认使用的Golden函数与样例功能匹配。例如矩阵乘应使用`ComputeMatmul`而非`ComputeGemm`（后者包含额外的alpha/beta缩放）。

2. **标杆计算的数据类型是否正确**：浮点标杆必须使用升精度计算。确认`ElementGolden`模板参数为`float`而非`half`：

```cpp
// ✅ 正确：标杆使用 float 升精度
std::vector<float> hostGolden(lenC);
golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

// ❌ 错误：标杆使用 half，CPU侧也会引入精度损失
std::vector<fp16_t> hostGolden(lenC);
golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
```

1. **Layout参数是否正确**：确认`layoutA`、`layoutB`、`layoutC`与样例实际使用的内存排布一致。行优先（RowMajor）和列优先（ColumnMajor）搞反是最常见的标杆错误。

2. **标杆输入数据与NPU输入一致**：确认传入标杆的`hostA`/`hostB`与拷贝到Device的数据完全相同。如果在拷贝后修改了host数据再传标杆，会导致标杆与NPU计算不同的数据。

3. **简单用例交叉验证**：用一个极简单的用例（如M=N=K=2，数据全为1.0）手动计算预期结果，与标杆输出对比，确认标杆逻辑正确。

### 清理编译缓存

编译缓存未清理会导致修改后的代码未生效，反复调试同一段旧代码：

```bash
rm -rf build/
rm -rf output/
```

或者在编译样例时增加`--clean`编译选项。

清理后重新编译运行，确认问题是否仍然存在。

### 固定最小可复现用例

将问题缩小到最小的可复现规模：

- 使用最小的M、N、K使问题复现（如M=N=K=16或32）
- 固定随机种子或使用固定数据替代随机数据，确保每次运行结果一致
- 简化Layout组合（优先使用RowMajor + RowMajor）

最小用例的好处：减少调试数据量、缩短编译运行周期、排除多核对多核交互的干扰。

### 确认修改已生效

在代码中插入明显的输出（如`std::cout << "check" << std::endl;`），确认修改后的二进制确实被执行。

## 模块化二分定位

### CATLASS样例分类：有无Tiling

首先需要明确一个关键区别：**CATLASS的大多数样例没有独立的Tiling计算步骤**，只有少数FlashAttention和动态Matmul样例包含显式的Tiling阶段。在开始二分定位前，先确认你开发的样例属于哪一类。

| 样例类型             | 是否有独立Tiling | 典型样例                                                                                                                                                    | 调用链路                                |
| -------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| 基础/量化Matmul      | ❌ 无            | `00_basic_matmul`、`01_batched_matmul`、`44_quant_matmul_full_loadA_tla`                                                                                    | 直接组装Block组件 → Kernel → DeviceGemm |
| FlashAttention / MLA | ✅ 有            | `19_mla`（`mla_tiling.h/cpp`，`GetMLATilingParam()`）、`23_flash_attention_infer`（`fai_tiling.cpp`，`GetFATilingParam()`）、`40_flash_attention_infer_tla` | Tiling计算 → Kernel启动                 |
| 动态Matmul           | ✅ 有            | `102_dynamic_optimized_matmul`（`DoTiling` + `SelectKernel`）、`103_dynamic_optimized_quant_matmul_per_token_basic`                                         | DoTiling → SelectKernel → Launch        |

> **关键结论**：
>
> - **对于无Tiling的样例**（占大多数），精度排查直接进入组件二分（3.3节），无需考虑Tiling问题。
> - **对于有Tiling的样例**（FA/MLA/动态Matmul），需要先区分Tiling问题还是Kernel问题，再进入组件二分。

### CATLASS模板化分层架构

CATLASS采用模板化分层设计，一个完整的样例由 **Device → Kernel → Block → Tile** 四个层级组装而成。理解这一分层结构是精准定位问题组件的前提。

以`44_quant_matmul_full_loadA_tla`为例，其完整组件层级如下：

```text
Device层:  DeviceGemm<MatmulKernel>
    ↓ 组装
Kernel层:  QuantMatmulFullLoadATla<BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>
    ↓ 组装
Block层:
    ├── BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>
    │       ↓ 内部使用的Tile组件
    │   Tile层:
    │       ├── TileCopy = PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias>
    │       │              ← 数据搬运Tile（负责A/B矩阵从GM到L1的搬运）
    │       └── (隐式) MMAD计算Tile          ← 矩阵乘累加Tile（负责L1→L0的Cube计算）
    │
    └── BlockEpilogue<EpilogueDispatchPolicy, ElementC, ElementScale, ElementPerTokenScale, ElementD,
                       TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul,
                       EpilogueTileCopy, TileScheduler>
            ↓ 内部使用的Tile组件
        Tile层:
            ├── TileRowBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>
            │              ← 行广播乘Tile（scale沿行方向广播后与结果相乘）
            ├── TileBroadcastOneBlkTla<ArchTag, ElementCompute, EpilogueTileShape::ROW>
            │              ← 单块广播Tile
            ├── TileOneBlkColumnBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>
            │              ← 列广播乘Tile（per-token scale沿列方向广播后相乘）
            ├── TileCopyDequantTla<ArchTag, ElementC, LayoutTagC, ElementScale, LayoutTagScale,
            │                       ElementPerTokenScale, LayoutTagPerTokenScale, ElementD, LayoutTagD>
            │              ← 反量化拷贝Tile（将累加结果反量化并转换为输出类型）
            └── EpilogueHorizontalTileSwizzle  ← Tile调度器（控制Tile的执行顺序和切换）
```

**核心认知**：精度问题的根因可能存在于任意层级——Kernel层的流水同步和组装逻辑、Block层的计算逻辑和组件使用方式、或Tile层的具体实现。二分定位的目标是**逐层缩小排查范围**，但需注意：**每一层本身都可能存在独立的问题**（如Kernel层的流水同步缺失、Block层的组件使用逻辑错误），而非仅存在于最底层的Tile中。因此二分时不能假设"问题一定在下一层"，应先验证当前层本身是否存在问题。

### 二分策略

#### 第一步（仅FA/MLA/动态样例）：区分Tiling问题还是Kernel问题

仅当样例包含显式Tiling阶段（见3.1节分类）时才执行此步：

- 用已知正确的Tiling参数替换当前Tiling计算结果 → 精度恢复？排查Tiling计算（TileShape划分、地址偏移计算等）
- 精度仍然失败 → 问题在Kernel实现中，进入第二步

对于无Tiling的样例，直接进入第二步。

#### 第二步：Block级二分（BlockMmad vs BlockEpilogue）

这是CATLASS精度定位的核心步骤。将Kernel拆分为两个Block组件分别验证：

**方法A：替换BlockEpilogue为直通（Identity Epilogue）**

将Epilogue替换为最简单的直通实现（不做任何后处理，直接将BlockMmad的累加结果写回），如果精度恢复，问题在BlockEpilogue；否则问题在BlockMmad。

```cpp
// 将复杂的BlockEpilogue替换为最简单的直通Epilogue
// 原代码（以44样例为例）：
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, ElementC, ElementScale, ElementPerTokenScale, ElementD,
    TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul,
    EpilogueTileCopy, TileScheduler>;
```

```cpp
// 替换为直通版本（仅做数据类型转换和写回，无量化/广播操作）：
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    SimpleDispatchPolicy, ElementC, void, void, ElementD,
    IdentityTile, IdentityTile, IdentityTile, SimpleTileCopy, SimpleScheduler>;
```

**方法B：替换BlockMmad为已知正确的实现**

用一个经过验证的BlockMmad（如从`00_basic_matmul`中提取的`Gemm::Block::BlockMmad`）替换当前BlockMmad，如果精度恢复，问题在BlockMmad；否则问题在BlockEpilogue。

#### 第三步：Tile级二分（在出问题的Block内部）

定位到具体Block后，进一步二分其内部的Tile组件：

**如果问题在BlockMmad**：

```text
BlockMmad
    ├── TileCopy（数据搬运Tile）
    │   └── 替换为简单DataCopy验证搬运逻辑是否正确
    │       - 检查搬运的地址偏移、数据量、Layout是否正确
    │       - 检查L1→L0的搬运流水线（SetFlag/WaitFlag）是否同步
    │
    └── MMAD计算Tile
        └── 检查累加器精度（是否使用FP32累加）
            - 检查mask参数（是否遗漏了尾块处理）
            - 检查repeatTime、stride等Cube指令参数
            - 检查L0 C的分块策略（l0CStages）
```

**如果问题在BlockEpilogue**（以44样例的量化Epilogue为例）：

```text
BlockEpilogue
    ├── TileRowBroadcastMulTla → 替换为逐元素Mul验证广播逻辑
    │   - 检查广播维度是否正确（沿ROW方向广播scale）
    │
    ├── TileBroadcastOneBlkTla → 检查广播的数据块是否正确
    │
    ├── TileOneBlkColumnBroadcastMulTla → 替换为逐元素Mul验证列广播
    │   - 检查广播维度是否正确（沿COL方向广播per-token scale）
    │
    ├── TileCopyDequantTla → 替换为普通DataCopyPad验证反量化逻辑
    │   - 检查反量化公式：output = accum * scale * per_token_scale
    │   - 检查Cast操作（FP32累加 → BF16/FP16输出）的RoundMode
    │
    └── EpilogueHorizontalTileSwizzle → 检查Tile调度顺序
        - 验证Tile的执行顺序是否导致数据覆盖或遗漏
```

### 组件替换示例

以`44_quant_matmul_full_loadA_tla`为例，展示如何替换具体组件来缩小排查范围：

| 层级                    | 组件                          | 替换方法                                                             | 验证目标                               |
| ----------------------- | ----------------------------- | -------------------------------------------------------------------- | -------------------------------------- |
| Block                   | BlockEpilogue                 | 替换为最简单的直通Epilogue（无量化、无广播，仅类型转换+写回）        | 判断问题在Epilogue还是Mmad             |
| Block                   | BlockMmad                     | 替换为`Gemm::Block::BlockMmad`（非TLA版本，从`00_basic_matmul`提取） | 验证基本Matmul逻辑是否正确             |
| Tile（BlockMmad内）     | TileCopy（PackedTileCopyTla） | 替换为`Gemm::Tile::SimpleTileCopy`                                   | 验证数据搬运逻辑（地址、Layout、同步） |
| Tile（BlockEpilogue内） | TileCopyDequantTla            | 替换为普通`DataCopyPad`（不做反量化）                                | 验证反量化公式是否为问题源             |
| Tile（BlockEpilogue内） | TileRowBroadcastMulTla        | 替换为逐元素Mul（不做广播）                                          | 验证广播维度是否正确                   |

对于`102_dynamic_optimized_matmul`等动态样例，其模块划分如下：

| 模块       | 文件/函数                        | 可替换性                     | 验证方法                   |
| ---------- | -------------------------------- | ---------------------------- | -------------------------- |
| Tiling计算 | `include/do_tiling_b16.h`        | 可替换为手工计算的Tiling参数 | 对比TilingParams各字段     |
| Kernel选择 | `include/select_kernel_b16.h`    | 可替换为固定Kernel           | 强制指定TilingKey          |
| 外围Launch | `impl/wrapper/*.cpp`（自动生成） | 可替换为直接调用Kernel模板   | 绕过launch_map直接实例化   |
| Kernel实现 | `impl/kernel/*.h`                | 可替换为简单Kernel           | 用basic_matmul的Kernel替代 |

### 二分定位流程

```text
精度比对失败
    │
    ├─ 用简单用例（M=N=K=16）验证标杆 → 标杆错误？修复标杆
    │
    ├─ [仅FA/MLA/动态样例] 用已知正确的Tiling参数替换 → 精度恢复？
    │   └─ 是 → 排查Tiling计算（TileShape、地址偏移等）
    │
    ├─ 模块化二分（快速方法无效时）
    │   ├─ [有Tiling的样例] 二分Tiling vs Kernel
    │   ├─ Block级二分：BlockMmad vs BlockEpilogue
    │   ├─ Tile级二分：在出问题的Block内部定位具体Tile
    │   └─ 在计算逻辑内部二分
    │
    └─ 对比法（保底手段）
        ├─ 找到正常工作的参考代码（如basic_matmul）
        └─ 逐模块、逐行对比差异
```

> **重要原则**：不要盲目试错。每次修改前先明确假设（"我认为问题在XXX模块"），修改后验证假设是否成立。如果连续多次修改无效，说明排查方向有误，应回到决策树重新判断。

## 诊断模式

以下诊断模式覆盖了CATLASS开发中最常见的精度问题场景。每种模式提供了从症状到根因的排查路径。

### FP32通过但FP16/BF16失败

**症状**：同一样例，FP32数据类型精度比对通过，但FP16或BF16版本失败。

**排查流程**：

```text
FP32通过，FP16/BF16失败
    │
    ├─ 检查累加器精度
    │   └─ BlockMmad中是否使用FP32累加？
    │       ├─ 是 → 累加器没问题，继续排查
    │       └─ 否 → 改为FP32累加，这是FP16/BF16精度的基本要求
    │
    ├─ 检查数值溢出
    │   └─ FP16最大值 ≈ 65504，BF16最大值 ≈ 3.39e38
    │       中间计算结果是否超出范围？
    │       ├─ 是 → 对输入做缩放（scale down），或使用更高精度的中间类型
    │       └─ 否 → 继续排查
    │
    ├─ 检查Epilogue中的Cast操作
    │   └─ FP32累加结果 → FP16/BF16输出的转换
    │       是否使用了正确的RoundMode？
    │       ├─ 默认RoundMode可能导致精度损失
    │       └─ 尝试不同的RoundMode（如RoundNearestEven vs RoundTowardZero）
    │
    └─ 检查减法抵消（Catastrophic Cancellation）
        └─ 两个相近的大数相减会导致有效位严重丢失
            FP16仅有10bit mantissa，BF16仅有7bit mantissa
            这种问题在FP32下不明显，但在低精度下会被放大
```

### 特定Shape或参数范围失败

**症状**：精度问题只在特定Shape（如非对齐的M/N/K、小Shape、大Shape）或特定参数组合下出现。

**排查流程**：

```text
特定Shape/参数失败
    │
    ├─ 小Shape失败（M、N、K < TileShape对应维度）
    │   └─ 可能原因：DataCopy对齐问题
    │       ├─ DataCopy要求最小搬运粒度，小Shape可能不满足
    │       └─ 解决：使用DataCopyPad替代DataCopy，或增加尾块处理逻辑
    │
    ├─ 非对齐Shape失败（M、N、K不是TileShape的整数倍）
    │   └─ 可能原因：尾块（tail block）处理逻辑有误
    │       ├─ mask参数是否正确传入？
    │       ├─ 尾块的地址偏移计算是否正确？
    │       └─ 尾块的数据初始化（清零）是否正确？
    │
    ├─ 大Shape失败
    │   └─ 可能原因：多核同步问题或内存越界
    │       ├─ BlockScheduler的调度逻辑是否正确？
    │       ├─ workspace大小是否足够？
    │       └─ 是否存在Bank Conflict或地址冲突？
    │
    └─ 特定参数组合失败（如特定batch size、特定head数）
        └─ 可能原因：参数相关的分支逻辑有误
            ├─ 检查与该参数相关的if/else分支
            └─ 检查模板特化是否正确匹配
```

### 输出呈现明显异常模式

**症状**：输出呈现可识别的异常模式，而非随机错误。

| 异常模式                       | 可能原因                                 | 排查方向                                                  |
| ------------------------------ | ---------------------------------------- | --------------------------------------------------------- |
| 输出全0                        | 累加器未初始化、数据未搬运、Kernel未执行 | 检查GlobalTensor.SetValue、检查DataCopy的SetFlag/WaitFlag |
| 输出全NaN                      | 除零、sqrt负数、无效浮点操作             | 检查Epilogue中的除法、开方等操作                          |
| 输出全Inf                      | 数值溢出                                 | 检查中间计算结果范围                                      |
| 输出为随机乱值                 | 未初始化内存、地址偏移错误               | 检查workspace初始化、检查地址计算                         |
| 输出部分区域正确、部分区域错误 | Block/Tile调度问题                       | 检查BlockScheduler、TileScheduler逻辑                     |
| 输出与标杆差一个固定倍数       | scale/bias处理遗漏                       | 检查Epilogue中的scale乘法是否遗漏                         |
| 输出矩阵转置了                 | Layout参数搞反                           | 检查RowMajor/ColumnMajor设置                              |

## 常见陷阱

以下是CATLASS开发中反复出现的精度陷阱，按出现频率排序。

### 流水线同步缺失

**症状**：输出全0或部分区域为0，或数据呈现"新旧混合"的状态。

**原因**：CATLASS使用流水线（Pipeline）机制实现数据搬运与计算的并行，通过 `SetFlag`/`WaitFlag` 控制各级流水之间的同步。例如，DataCopy 将数据从 GM 搬到 L1 后，通过 `SetFlag<MTE2_MTE1>` 通知计算单元数据已就绪；计算单元在读取 L1 数据前通过 `WaitFlag<MTE2_MTE1>` 等待搬运完成。如果遗漏了 SetFlag 或 WaitFlag，计算单元可能在数据搬运完成前就开始读取，导致读到未初始化或旧数据。

**排查方法**：

- 检查 DataCopy 完成后是否有对应的 `SetFlag` 操作（如 `SetFlag<MTE2_MTE1>`）
- 检查 MMAD 计算前是否有对应的 `WaitFlag` 操作（如 `WaitFlag<MTE2_MTE1>`）
- 检查 L0 级流水：DataCopy 从 L1 到 L0 后是否设置了 `SetFlag<MTE1_M>`，MMAD 读取 L0 前是否 `WaitFlag<MTE1_M>`
- 检查流水线的 stage 数量配置是否合理（event ID 是否与 stage 数匹配）
- 对于跨核同步场景，检查 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 是否成对出现

### DataCopy非对齐

**症状**：小Shape（M、N、K小于TileShape）时精度失败，大Shape正常。

**原因**：DataCopy有最小搬运粒度要求（通常为16Byte或32Byte对齐）。当数据量不满足对齐要求时，DataCopy可能搬运了多余的数据（读到越界数据）或遗漏了部分数据。

**排查方法**：

- 确认是否使用了DataCopyPad（带padding的版本）处理非对齐场景
- 检查尾块处理中是否正确使用了mask来限制有效数据范围

### 数值溢出

**症状**：输出中出现Inf或异常大的数值。

**原因**：

- FP16最大值约65504，BF16最大值约3.39e38
- 矩阵乘的累加结果随K维增大而增大，容易超出FP16范围
- 中间计算（如平方、指数）更容易溢出

**排查方法**：

- 估算中间结果的最大可能值
- 对输入数据做缩放（scale down）
- 使用FP32作为中间累加类型

### 累加器精度不足

**症状**：FP16/BF16精度误差偏大，尤其是大K场景。

**原因**：如果BlockMmad使用FP16或BF16作为累加器类型（而非FP32），每次累加都会引入舍入误差。K越大，误差累积越严重。

**排查方法**：

- 确认BlockMmad的ElementC模板参数为float（FP32累加）
- 如果已经是FP32累加但仍有问题，检查L0 C的分块策略（l0CStages）

### Epilogue中的量化/反量化公式错误

**症状**：输出与标杆差一个固定的比例因子，或误差分布呈现系统性偏差。

**原因**：量化Matmul的Epilogue中包含scale乘法、反量化等操作。公式错误（如遗漏某个scale因子、乘法顺序错误）会导致系统性偏差。

**排查方法**：

- 手工推导Epilogue的完整计算公式
- 与标杆的计算公式逐项对比
- 使用简单数据（如全1.0输入、全1.0 scale）验证公式

### Layout参数不匹配

**症状**：输出矩阵看起来被转置了，或错误集中在特定维度上。

**原因**：RowMajor和ColumnMajor决定了数据在内存中的排列顺序。如果样例内部假设的Layout与实际数据的Layout不一致，会导致读取/写入错误的数据。

**排查方法**：

- 确认LayoutA、LayoutB、LayoutC在标杆和样例中一致
- 检查DataCopy的LayoutTag模板参数

### 编译缓存未清理

**症状**：修改代码后问题没有任何变化，仿佛修改未生效。

**原因**：ATC编译器会缓存已编译的Kernel。如果缓存未清理，即使修改了源代码，实际运行的仍是旧版本。

**排查方法**：

```bash
rm -rf build/
rm -rf $HOME/atc_data/kernel_cache/
```

或在编译时添加`--clean`选项。

## 调试策略层级

当面对一个精度问题时，按以下层级递进使用调试策略。从成本最低的快速方法开始，逐步深入到更系统化的方法。

### 第一层：快速方法（成本最低，优先尝试）

| 方法         | 适用场景           | 操作                                         |
| ------------ | ------------------ | -------------------------------------------- |
| 清理编译缓存 | 任何修改后问题不变 | `rm -rf build/ $HOME/atc_data/kernel_cache/` |
| 固定随机种子 | 问题不稳定复现     | 使用固定种子或固定数据替代随机数据           |
| 缩小问题规模 | 大Shape场景        | 使用最小可复现Shape（如M=N=K=16）            |
| 简化数据类型 | FP16/BF16失败      | 先测试FP32版本是否通过                       |
| 简化Layout   | 复杂Layout组合     | 统一使用RowMajor                             |
| 检查标杆     | 不确定问题在哪侧   | 用简单用例手工验证标杆输出                   |

### 第二层：模块化二分（核心方法）

当快速方法无法定位问题时，使用模块化二分法。这是CATLASS精度调试的核心策略。

**二分层级（由粗到细）**：

```text
第一级：[有Tiling的样例] Tiling计算 vs Kernel实现
    └─ 用已知正确的Tiling参数替换，判断问题在哪一侧
    └─ 注意：Kernel层本身可能存在流水同步、组装逻辑等问题，不一定是Tiling的问题

第二级：Kernel层自身 vs Block层
    └─ Kernel层除了组装Block组件外，自身也包含流水同步逻辑（SetFlag/WaitFlag）、
       workspace管理、多核调度等，这些都可能引入精度问题
    └─ 先检查Kernel层的流水同步和组装逻辑是否正确，再进入Block级二分

第三级：BlockMmad vs BlockEpilogue
    └─ 替换Epilogue为直通版本，或替换BlockMmad为已知正确版本
    └─ 注意：Block层除了包含Tile组件外，自身也有组件使用逻辑（如Tile的组装顺序、
       模板参数传递、DispatchPolicy选择等），这些也可能引入问题

第四级：Tile组件二分
    └─ 在出问题的Block内部，逐个替换Tile组件

第五级：计算逻辑内部二分
    └─ 在出问题的Tile内部，二分具体的计算步骤
```

**二分原则**：

- 每次只改一个变量，确保能明确判断修改的效果
- 优先替换为最简单的实现（如直通Epilogue、简单DataCopy）
- 每次修改前明确假设，修改后验证假设

### 第三层：对比法（保底手段）

当二分法无法定位问题时，使用对比法作为保底手段。

**操作步骤**：

1. 找到一个功能相似且精度正常的参考代码（如`00_basic_matmul`）
2. 从顶层到底层，逐模块对比差异：
   - Device层：DeviceGemm的模板参数
   - Kernel层：Kernel的组装方式
   - Block层：BlockMmad和BlockEpilogue的模板参数
   - Tile层：各个Tile组件的实现细节
3. 将当前代码逐步向参考代码靠拢，每次修改一个差异点并验证精度
4. 当精度恢复时，最后修改的那个差异点就是问题根因

**对比法虽然耗时，但在面对复杂或隐蔽的精度问题时往往是最可靠的方法。**

### 策略选择决策树

```text
精度比对失败
    │
    ├─ 快速方法能否定位？
    │   ├─ 是 → 修复，完成
    │   └─ 否 → 进入模块化二分
    │
    ├─ 模块化二分
    │   ├─ [有Tiling] Tiling vs Kernel → 定位到一侧
    │   ├─ BlockMmad vs BlockEpilogue → 定位到具体Block
    │   ├─ Tile组件二分 → 定位到具体Tile
    │   └─ 计算逻辑二分 → 定位到具体代码行
    │
    └─ 二分法无法定位？
        └─ 使用对比法，与正常工作的参考代码逐模块对比
```

CATLASS的模块化架构为精度调试提供了天然的优势：每个模块都可以独立替换和验证。充分利用这一特点，配合标杆验证和决策树引导，可以高效地定位绝大多数精度问题。

## 精度标准参考

CATLASS中不同数据类型的默认精度标准如下。当样例开发Plan有明确精度要求时，以Plan为准。

| 数据类型 | rtol | atol | 说明                     |
| -------- | ---- | ---- | ------------------------ |
| FP32     | 1e-5 | 1e-6 | 高精度，容忍度最低       |
| FP16     | 1e-3 | 1e-4 | 中等精度                 |
| BF16     | 1e-2 | 1e-3 | 低精度（mantissa仅7bit） |
| INT      | -    | 0    | 要求二进制完全一致       |

CATLASS的`CompareData`函数会根据`computeNum`（通常为K维大小）动态调整rtol：

| 计算次数 | FP16/FP32 rtol | BF16 rtol |
| -------- | -------------- | --------- |
| < 2048   | 1/256          | 1/128     |
| ≥ 2048   | 1/128          | 1/64      |

## 调试检查清单

每次精度调试时，按以下清单逐项确认：

**调试前**：

- [ ] 已阅读[精度分析基础](./precision_analysis_basics.md)，了解精度比对方式
- [ ] 已固定最小可复现用例（最小Shape、固定数据）
- [ ] 已清理编译缓存（`rm -rf build/ $HOME/atc_data/kernel_cache/`）
- [ ] 已确认修改后的二进制确实被执行

**问题分类**：

- [ ] 已判断是完全算错还是精度误差
- [ ] 已观察错误元素的分布模式

**标杆验证**：

- [ ] 标杆函数选择正确（ComputeMatmul vs ComputeGemm等）
- [ ] 标杆使用升精度计算（ElementGolden = float）
- [ ] Layout参数与样例一致
- [ ] 标杆输入数据与NPU输入一致

**常见陷阱排查**：

- [ ] 已排查流水线同步问题（DataCopy后是否使用SetFlag/WaitFlag进行同步）
- [ ] 已排查DataCopy对齐问题（小Shape是否使用DataCopyPad）
- [ ] 已排查GlobalTensor.SetValue问题
- [ ] 已排查数值溢出（FP16 max ≈ 65504）

**模块化二分**：

- [ ] 已尝试替换Tiling参数（如有Tiling的样例）
- [ ] 已尝试Block级二分（BlockMmad vs BlockEpilogue）
- [ ] 已尝试Tile级二分（在出问题的Block内部定位具体Tile）
- [ ] 已尝试替换为简单Kernel

## 总结

CATLASS精度问题定位的核心思路可以概括为"**先分类、再二分、后诊断**"：

| 阶段   | 目标         | 关键动作                                                                                                        |
| ------ | ------------ | --------------------------------------------------------------------------------------------------------------- |
| 先分类 | 判断问题性质 | 区分完全算错 vs 精度误差，观察错误分布                                                                          |
| 再二分 | 缩小问题范围 | 验证标杆 → [有Tiling]二分Tiling/Kernel → Block级二分(BlockMmad vs BlockEpilogue) → Tile级二分(定位具体Tile组件) |
| 后诊断 | 定位具体原因 | 对照诊断模式和常见陷阱，找到根因并修复                                                                          |

CATLASS的模块化架构为精度调试提供了天然的优势：每个模块都可以独立替换和验证。充分利用这一特点，配合标杆验证和决策树引导，可以高效地定位绝大多数精度问题。
