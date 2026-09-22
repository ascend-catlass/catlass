# 创新样例开发流程指南

该文档面向需要在 CATLASS模板库中开发创新算子样例的开发者，以[样例44](../../../examples/44_quant_matmul_full_loadA_tla/README.md) `quant_matmul_full_loadA_tla`（量化矩阵乘A矩阵全载）为案例，说明从需求分析、方案设计到组件开发、测试合入的完整流程，帮助开发者快速掌握创新样例的开发方法。

## 什么是创新样例

CATLASS模板库中的每个样例都是在已有样例基础上的创新和扩展，都带有各自的新特性。开发新样例的核心思路是：**最大化复用已有组件，仅在必要时进行创新开发**。当你的算子需求无法通过现有组件组合完全满足时——例如需要新的数据搬运策略、新的后处理计算逻辑、或新的多核调度方式——就需要在参考已有样例的基础上，开发新的Block层组件或Kernel逻辑。

算子样例可以在[examples](../../../examples/)目录下查询。

以[样例44](../../../examples/44_quant_matmul_full_loadA_tla/README.md)为例，它并非从零开始构建，而是结合了两个已有样例的特性：量化矩阵乘的后处理逻辑（BlockEpilogue）复用了[样例42](../../../examples/42_quant_optimized_matmul_tla/README.md)（`42_quant_optimized_matmul_tla`）的per-token反量化方案，A矩阵全载策略（包括BlockMmad和BlockScheduler）参考了[样例25](../../../examples/25_matmul_full_loadA/README.md)（`25_matmul_full_loadA`）。其BlockMmad组件（`block_mmad_pingpong_full_loadA_tla.hpp`）是在已有全载组件（`block_mmad_pingpong_full_loadA.hpp`）基础上做的TLA改写，BlockScheduler（`GemmIdentityBlockSwizzleL1FullLoad`）则直接复用了[样例25](../../../examples/25_matmul_full_loadA/README.md)的全载调度策略。[样例44](../../../examples/44_quant_matmul_full_loadA_tla/README.md)没有从零新增任何组件，所有模块均通过复用或TLA改写已有组件完成。

## 设计阶段

### 需求分析

在动手编码之前，首先需要明确算子的核心需求，并**充分调研已有样例和模板组件**，判断哪些可以直接复用、哪些需要参考改写、哪些必须全新开发：

- **算子功能**：明确算子的数学定义和计算流程。样例44实现的是量化矩阵乘法 `D = (A_int8 × B_int8) * scale * perTokenScale`，输出bf16类型。
- **已有组件调研**：遍历已有样例和`include/catlass/`下的模板组件，识别可复用资源。样例44的调研结论如下：

    | 特性                         | 参考来源                               | 复用方式               |
    | ---------------------------- | -------------------------------------- | ---------------------- |
    | 量化矩阵乘 + per-token反量化 | 样例42 `42_quant_optimized_matmul_tla` | BlockEpilogue直接复用  |
    | A矩阵全载到L1                | 样例25 `25_matmul_full_loadA`          | 流水设计思路参考       |
    | 全载BlockMmad（非TLA）       | `block_mmad_pingpong_full_loadA.hpp`   | TLA改写基础            |
    | 全载场景多核调度             | 样例25 `25_matmul_full_loadA`          | BlockScheduler直接复用 |

- **创新特性**：在调研基础上，明确真正需要创新的点。样例44的核心创新在于将量化矩阵乘与A矩阵全载两种特性结合，通过复用样例25的BlockScheduler使每个核处理的基本块在N方向连续分布，最大化A矩阵的L1复用率。
- **适用场景**：明确新特性的收益场景和限制条件。A矩阵全载在N轴较大时收益显著（A矩阵可被多次复用），但要求L1空间足够容纳`L1TileShape::M × K`的数据量，否则无法使用。

### 方案设计

根据需求分析和组件调研结果，设计算子的整体实现方案。样例44的方案设计体现了"参考复用为主、创新开发为辅"的原则：

**（1）模板参数选择**

确定算子的核心模板参数，包括TileShape、数据类型、Dispatch策略等。样例44的关键参数选择如下：

| 参数           | 取值            | 说明                              |
| -------------- | --------------- | --------------------------------- |
| L1TileShape    | `128×256×512`   | L1级分块大小，M=128, N=256, K=512 |
| L0TileShape    | `128×256×128`   | L0级计算分块大小                  |
| ElementA/B     | int8_t          | 量化输入矩阵                      |
| ElementC       | int32_t         | 累加器类型                        |
| ElementD       | bfloat16_t      | 最终输出类型                      |
| DispatchPolicy | `MmadFullLoadA` | 全载A矩阵的流水策略               |

**（2）组件拆分与复用**

将算子拆分为组件模块，明确每个组件的复用/参考/新增关系：

| 组件           | 作用                                                 | 开发方式     | 参考来源                                       |
| -------------- | ---------------------------------------------------- | ------------ | ---------------------------------------------- |
| BlockMmad      | A矩阵全载 + B矩阵pingpong搬运，执行块级矩阵乘法      | **TLA改写**  | `block_mmad_pingpong_full_loadA.hpp`（样例25） |
| BlockEpilogue  | 反量化后处理（scale乘法 + per-token乘法 + 类型转换） | **直接复用** | 样例42 `42_quant_optimized_matmul_tla`         |
| BlockScheduler | 多核任务分配，全载场景下的连续分块                   | **直接复用** | 样例25 `25_matmul_full_loadA`                  |
| Kernel         | 组装上述组件，实现AIC/AIV双核流水                    | **参考改写** | 样例42的TLA Kernel结构                         |

**（3）流水设计**

设计数据搬运和计算的流水线。样例44的流水方案如下图所示：

- 在计算开始阶段，通过MTE2将A矩阵的完整数据块（`L1TileShape::M × K`）一次性搬入L1
- 之后进入主循环，pingpong搬运B矩阵的K维分块到L1，同时执行L1→L0的数据搬运和Cube计算
- A矩阵常驻L1，无需重复搬运，从而减少MTE2带宽压力

<img src="https://raw.gitcode.com/user-images/assets/7631999/1de46727-7c46-411e-936c-7a437d951a3a/3e9c799e1de0405d89f07a6bfd7d7c54.png_tplv-a9rns2rl98-image-qvalue.png" width="60%">

### 文档设计

在设计阶段就应规划好样例的文档结构。每个创新样例需要包含以下文档：

- **README.md**：算子简要说明、使用方法和快速上手指南
- **设计文档（`${id}_${op_name}.md`）**：详细说明原型设计、方案设计、组件实现和性能收益
- **代码注释**：关键接口和复杂逻辑需添加清晰的注释

## 开发阶段

### 环境准备与代码结构

参考[快速上手](./01_quick_start.md)搭建开发环境，然后按照以下目录结构组织代码：

```bash
examples/44_quant_matmul_full_loadA_tla/   # 样例目录
├── CMakeLists.txt                          # 编译配置
├── README.md                               # 使用说明
├── 44_quant_matmul_full_loadA_tla.md       # 设计文档
└── quant_matmul_full_loadA_tla.cpp         # Host侧组装代码

include/catlass/gemm/kernel/
└── quant_matmul_full_loadA_tla.hpp         # Kernel实现

include/catlass/gemm/block/
├── block_mmad_pingpong_full_loadA_tla.hpp  # TLA改写BlockMmad（基于block_mmad_pingpong_full_loadA.hpp）
└── block_swizzle.hpp                       # 复用自样例25的L1FullLoad Swizzle

include/catlass/epilogue/block/
└── block_epilogue_per_token_dequant_tla.hpp # 复用自样例42的BlockEpilogue
```

### Host侧组装

Host侧代码负责设备初始化、内存分配、组件组装和Kernel调用，详细说明可参考[Host层样例组装](./02_host_example_assembly.md)。开发时主要关注以下几点：

**（1）数据类型与Layout定义**

根据算子需求定义所有输入输出张量的数据类型和内存排布。样例44涉及5个张量：int8类型的A/B矩阵、float类型的scale和perTokenScale、bf16类型的输出D矩阵。

**（2）组件模板实例化**

按照“DispatchPolicy → BlockMmad → BlockEpilogue → BlockScheduler → Kernel → Adapter”的顺序逐层组装：

- 首先确定DispatchPolicy（如`MmadFullLoadA`），它控制各级缓存的stage数和流水行为
- 然后基于DispatchPolicy和TileShape组装BlockMmad，指定TileCopy组件处理数据搬运
- 接着组装BlockEpilogue，配置反量化所需的Tile组件（如`TileRowBroadcastMul`）
- 再根据M维度是否需要分核，选择合适的BlockScheduler（全载场景用`GemmIdentityBlockSwizzleL1FullLoad`，普通场景用`GemmIdentityBlockSwizzle`）
- 最后将BlockMmad、BlockEpilogue、BlockScheduler组装为Kernel，并通过DeviceGemm适配器完成调用

**（3）分核策略选择**

样例44根据`problemShape.M`与`L1TileShape::M`的大小关系，在Host侧选择不同的BlockScheduler：

- 当`M > L1TileShape::M`时，M方向需要分核处理，使用`GemmIdentityBlockSwizzleL1FullLoad`策略，使每个核处理的基本块连续分布，提升A矩阵全载时的块间复用率
- 当`M <= L1TileShape::M`时，M方向不需要分核，使用普通的`GemmIdentityBlockSwizzle`即可

### Kernel层开发

Kernel层是算子的执行入口，负责调度器初始化、全局张量定义和核心循环控制，详细说明可参考[Kernel代码开发](./03_kernel_development.md)。创新样例的Kernel开发要点如下：

**（1）参数结构设计**

定义`Arguments`（用户接口层）和`Params`（内核执行层）两层参数结构。样例44的Arguments包含problemShape、aicCoreNum、输入张量组（A/B/Bias）和输出张量组（Scale/PerTokenScale/D）。

**（2）AIC核实现**

AIC核负责矩阵乘法计算，核心流程为：

- 初始化BlockScheduler，获取总循环次数
- 定义指向GM的全局张量
- 循环处理每个分块：计算块坐标→计算GM偏移→调用BlockMmad执行矩阵乘法→将结果写入Workspace

**（3）AIV核实现**

AIV核负责Epilogue后处理，核心流程为：

- 初始化BlockEpilogue组件
- 从Workspace读取AIC核的计算结果
- 执行反量化计算（scale乘法 + perTokenScale乘法 + 类型转换）
- 将最终结果写回GM

**（4）AIC/AIV同步**

通过CrossCoreFlag机制实现AIC和AIV核之间的流水同步：AIC核完成一个分块的计算后设置标志，AIV核等待该标志后开始后处理，处理完成后再设置标志通知AIC核复用该Workspace阶段。

### Block层组件开发

Block层是创新样例开发的核心。以下结合样例44说明三类组件的开发要点，各组件详细开发指南请参考对应文档：

**（1）BlockMmad — 矩阵乘计算组件**（详见 [BlockMmad代码开发](./04_block_mmad_development.md)）

BlockMmad负责管理L1/L0缓存、执行数据搬运和Cube计算。样例44的`BlockMmadTla`（`block_mmad_pingpong_full_loadA_tla.hpp`）是在已有全载组件`block_mmad_pingpong_full_loadA.hpp`基础上做的TLA改写：

- L1缓存分配：A矩阵占用`L1TileShape::M × K`空间（全载），B矩阵占用`L1TileShape::N × L1TileShape::K × 2`空间（双buffer pingpong）
- 构造函数中完成L1/L0 buffer的分配和事件初始化
- operator()中实现：先一次性加载完整A矩阵到L1→进入K维循环→pingpong加载B矩阵分块→L1→L0搬运→Cube计算→L0C写回

与普通pingpong BlockMmad的核心区别在于：A矩阵只搬运一次，后续所有K维循环都复用L1中的A矩阵数据。

**（2）BlockEpilogue — 后处理组件**（详见 [Epilogue适配与开发](./07_epilogue_adaptation.md)）

BlockEpilogue负责对矩阵乘结果进行后处理。样例44的`BlockEpilogue`直接复用了样例42的per-token反量化方案：

- UB缓存分配：为C矩阵、Scale、PerTokenScale、输出D矩阵各分配双buffer空间
- 计算流程：从GM加载C和Scale到UB→执行tileRowBroadcastMul（per-channel乘）→加载PerTokenScale→执行tileOneBlkColumnBroadcastMul（per-token乘）→类型转换（int32→bf16）→写回GM
- 通过UB双buffer轮转实现计算与搬运的重叠

**（3）BlockScheduler — 多核调度组件**（详见 [BlockScheduler代码开发](./05_block_scheduler_development.md)）

BlockScheduler负责将计算任务分配到多个AIC核。样例44复用了样例25的`GemmIdentityBlockSwizzleL1FullLoad`，其与普通Swizzle的核心区别在于分块顺序：

- 普通Swizzle：分块按`0-1-2-...-19-0-1-2-...`交替分配到各核，每个核处理的基本块跳跃分布
- L1FullLoad Swizzle：分块按`0-0-...-0-1-1-...-1-2-2-...-19`连续分配，每个核处理的基本块在N方向连续，使得A矩阵全载后能被连续的基本块充分复用

### Tile层组件开发

Tile组件是CATLASS中最底层的计算和数据操作单元，直接与硬件交互，主要包括`TileMmad`（矩阵乘法计算）和`TileCopy`（各级缓存间的数据搬运）。详细开发指南请参考[Tile组件代码开发详解](./06_tile_development.md)。

在Block层组件开发过程中，需要分析BlockMmad和BlockEpilogue内部使用的Tile组件是否满足需求：

- **TileMmad**：负责L0缓存上的矩阵乘法运算。大多数场景下，已有的`TileMmad`模板（支持int8/bf16/fp16等多种数据类型组合）可直接复用。仅当需要支持新的数据类型组合或特殊计算模式时，才需要扩展TileMmad。
- **TileCopy**：负责GM↔L1、L1↔L0、L0→GM等各级缓存间的数据搬运。已有的`TileCopy`模板覆盖了常见的数据搬运模式（如`CopyGmToL1`、`CopyL1ToL0A/B`、`CopyL0CToGm`等），通常可直接复用。仅当需要新的数据排布转换或特殊搬运策略时，才需要新增TileCopy组件。

以样例44为例，其BlockMmad和BlockEpilogue内部使用的TileMmad和TileCopy均直接复用了已有组件，无需开发新的Tile组件。开发者在设计自己的样例时，应优先遍历`include/catlass/gemm/tile/`目录下的已有Tile组件，确认是否满足需求后再决定是否需要新增。

### 编译与调试

使用以下命令编译和运行样例：

```bash
# 编译
bash scripts/build.sh 44_quant_matmul_full_loadA_tla

# 运行
./output/44_quant_matmul_full_loadA_tla 512 4096 1024 0
```

开发过程中建议使用[性能调测工具](./08_evaluation.md)进行仿真验证，通过流水图检查数据搬运和计算的重叠情况是否符合预期。

## 测试阶段

### 精度测试

- 使用`examples/common/golden.hpp`中的标杆函数或自行实现标杆，对比算子输出与标杆结果的误差
- 执行至少200例泛化精度测试，覆盖不同的M/N/K组合
- 样例44使用`golden::QuantMatmul`作为精度标杆，对比bf16输出的误差

### 性能测试

- 与标杆算子进行性能对比，验证新特性的收益。样例44以`12_quant_matmul`为标杆，在N轴较大的场景下性能提升5%~15%
- 在算子设计文档中记录性能数据、测试环境和标杆信息
- 若性能未达预期，参考[性能瓶颈分析及优化手段](./evaluation/bottleneck_analysis_and_optimization.md)进行定位和优化

## 合入阶段

### 代码规范检查

- 确保代码符合项目的代码风格（参考`.clang-format`配置）
- 检查命名规范：样例目录使用`${id}_${op_name}`格式，文件名使用小写下划线风格
- 添加必要的版权声明和License头

### 文档完善

- README.md包含算子简介、编译运行步骤和参数说明
- 设计文档包含原型设计、方案设计、组件说明和性能数据
- 如有必要，补充新组件的使用说明或API文档

### 提交流程

参考[样例贡献流程](./09_example_contribution_guide.md)和[CONTRIBUTING.md](../../../CONTRIBUTING.md)：

1. 创建Issue说明新算子的设计方案和创新点
2. 提交代码到个人分支
3. 创建PR，详细填写模板信息（包括精度和性能测试结果）
4. 等待代码审查并根据反馈修改
5. 审查通过后合入主分支

## 总结

开发一个CATLASS创新样例的核心流程可以概括为以下四步：

1. **调研已有组件**：在动手开发之前，首先遍历已有样例和`include/catlass/`下的模板组件，判断哪些可以直接复用、哪些需要参考改写、哪些必须全新开发。这是最关键的一步——充分的调研能最大程度减少重复开发工作。
2. **明确创新点**：在调研基础上，确定真正需要创新的部分。如样例44的调研结论是：BlockEpilogue可直接复用样例42，BlockMmad可在已有全载组件基础上做TLA改写，BlockScheduler可直接复用样例25，所有模块均无需从零开发。
3. **逐层实现**：按照Host→Kernel→Block→Tile的顺序，自顶向下完成各层代码。优先复用和改写已有组件，仅在必要时开发新组件。
4. **验证收益**：通过精度和性能测试，确认新特性达到了预期效果。

样例44通过结合量化矩阵乘（复用样例42）和A矩阵全载（参考样例25），所有组件均通过复用或TLA改写完成，就在N轴较大的量化矩阵乘场景中实现了5%~15%的性能提升，充分体现了"最大化复用、最小化创新"的开发原则。开发者在设计自己的样例时，应首先完成全面的组件调研，再确定创新点和开发方案。
