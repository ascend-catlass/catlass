# CV融合算子性能调优案例集

## 写在前面

本文整理CATLASS中CV融合算子的性能分析和调优方法。这里的CV融合指同一个kernel中同时使用AIC和AIV，将矩阵计算、后处理、归约、Softmax或数据整理等阶段放到一条流水中执行。

本文主要覆盖三个样例：

| 样例           | 路径                                          | 融合形态                                                     |
| -------------- | --------------------------------------------- | ------------------------------------------------------------ |
| quant_matmul   | `examples/12_quant_matmul`                    | AIC执行int8 matmul，AIV执行per-token/per-channel反量化和写回 |
| streamk_matmul | `examples/37_streamk_matmul`                  | AIC执行普通块和尾轮Stream-K块，AIV归约尾轮部分和             |
| FAI            | `examples/49_ascend950_flash_attention_infer` | AIC执行QK/PV，AIV执行Online Softmax/RescaleO                 |

CV融合算子性能调优时不能只看单个阶段的算力利用率，它的调优收益来自阶段重叠，最终耗时通常由最长流水、流水空泡、核间长尾、GM workspace搬运和同步开销共同决定。

## 通用分析方法

### 获取性能数据

建议先按单算子方式执行样例并采集Profiling数据，确认AIC、AIV、MTE2、MTE3、FixPipe占比。CATLASS中性能工具的使用方式可参考[msProf&Profiling](../evaluation/performance_tools.md)。

采集数据时建议准备两组结果：

| 数据                        | 用途                                   |
| --------------------------- | -------------------------------------- |
| 融合kernel总耗时            | 判断整体收益和不同参数组合的最终效果   |
| AIC/AIV/MTE/FixPipe流水占比 | 判断是Cube/Vector、搬运/写回成为瓶颈   |
| 每核任务量或每核耗时        | 判断核间负载是否均衡                   |
| workspace大小和对齐信息     | 判断CV阶段间中间结果是否引入额外GM瓶颈 |

如果需要观察更细的流水并行关系，可以结合[性能流水仿真](../evaluation/performance_tools.md#performance-pipeline-simulation)，重点看AIC和AIV之间是否存在长时间互等。

### 建立融合收益模型

可以用下面的简化模型判断一个CV融合优化是否值得继续推进：

```text
串行耗时 = AIC阶段耗时 + AIV阶段耗时 + 搬运/写回耗时
融合耗时 ~= max(AIC流水耗时, AIV流水耗时, 搬运流水耗时) + 首尾flush开销 + 同步开销
融合收益 = (串行耗时 - 融合耗时) / 串行耗时
```

理想情况下，AIC和AIV耗时接近，且中间数据不需要反复搬入搬出GM，融合收益更明显。如果某一侧远慢于另一侧，另一侧会出现等待，融合收益会收敛到瓶颈阶段耗时。若切块过小、阶段数过多或workspace访问低效，融合后性能还可能因为调度、同步和搬运问题而下降。

### 设计优化方案

CV融合算子的调优通常按以下顺序推进：

1. 调整TileShape或基本块大小，让单块计算粒度足够大，同时满足L1/L0/UB资源限制。
2. 调整block scheduler和swizzle，让基本块数量与AIC数量更匹配，实现负载均衡。
3. 调整CV流水深度或workspace stages，让AIC产出和AIV消费尽量重叠。
4. 检查中间结果路径，优先使用片上缓存或对齐的workspace，避免低效GM中转。
5. 根据Profiling定位MTE2、MTE3、FixPipe问题，针对输入布局、stride、workspace对齐和写回路径优化。

## 案例集

### 案例一：quant_matmul反量化融合

**样例结构**

`examples/12_quant_matmul/quant_matmul.cpp`实现int8输入、int32累加、half输出的量化矩阵乘：

```text
C_int32 = A_int8 * B_int8
D_half = C_int32 * scale[n] * perTokenScale[m]
```

样例中的关键模板配置如下：

```cpp
using L1TileShape = GemmShape<128, 256, 512>;
using L0TileShape = GemmShape<128, 256, 128>;
constexpr uint32_t workspaceStages = 2;
constexpr uint32_t ubStages = 2;
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
using EpilogueTileShape = MatrixShape<32, 256>;
using MatmulKernel =
    Gemm::Kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>;
```

该kernel的融合边界在`include/catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp`中：

| 阶段     | 执行单元 | 工作                                                                            |
| -------- | -------- | ------------------------------------------------------------------------------- |
| Matmul   | AIC      | 读取A/B，执行int8 BlockMmad，FixPipe把int32中间结果写到workspace                |
| Epilogue | AIV      | 从workspace读取int32，加载`scale`和`perTokenScale`，完成广播乘、类型转换和D写回 |

AIC和AIV在workspace槽位形成生产者-消费者流水。AIC写完当前stage后设置`flagAicFinishStoreList[stageId]`，AIV消费完成后设置`flagAivFinishComputeList[stageId]`，AIC在复用同一stage前等待AIV完成。

**性能瓶颈判断**

如果Profiling显示AIC占比高，优先看matmul侧：

| 现象                   | 可能原因                          | 调优方向                                      |
| ---------------------- | --------------------------------- | --------------------------------------------- |
| AIC长时间运行，AIV稀疏 | `L1TileShape`过小或基本块数量过多 | 增大M/N/K tile，减少循环次数                  |
| MTE2占比高             | A/B输入布局或stride不友好         | 检查A RowMajor、B ColumnMajor是否匹配业务输入 |
| FixPipe占比高          | int32中间结果写workspace压力大    | 检查workspace地址对齐和stage布局              |

如果AIV占比高，优先看epilogue侧：

| 现象                            | 可能原因                        | 调优方向                                           |
| ------------------------------- | ------------------------------- | -------------------------------------------------- |
| AIC频繁等待AIV释放stage         | 反量化UB tile过小或Vector计算重 | 调整`EpilogueTileShape`、`ubStages`或减少scale搬运 |
| scale/per-token scale搬运占比高 | scale访问重复，M/N tile不匹配   | 调整M/N tile，让scale复用和输出写回更连续          |
| 尾块AIV耗时异常                 | M/N不对齐导致UB和GM搬运尾块多   | 优先选择能降低尾块比例的TileShape                  |

**调优要点**

**1. 配平AIC和AIV阶段**

`L1TileShape<128, 256, 512>`意味着每个AIC基本块输出`128 x 256`个int32中间结果。AIV侧`EpilogueTileShape<32, 256>`会把一个AIC tile在M方向拆成4个epilogue tile处理。若AIV成为瓶颈，不能只增大matmul tile；更大的AIC tile会增加单个stage中AIV需要消费的数据，可能让AIC更频繁等待stage释放。

可按下面方式做候选搜索：

```text
候选1：固定L1TileShape::M/N，调整EpilogueTileShape::ROW
候选2：固定EpilogueTileShape，调整L1TileShape::M/N以改善核间均衡
候选3：同时调整L1TileShape::K和L0TileShape::K以改善Cube效率
```

每组候选都需要确认workspace大小：

```text
workspace_size = L1TileShape::M * L1TileShape::N * aicCoreNum * workspaceStages * sizeof(uint32_t)
```

workspace过大时，GM访问和cache压力可能抵消融合收益。

**2. 调整workspaceStages**

`workspaceStages = 2`是典型双缓冲。若流水图显示AIC写完stage后经常等待AIV，可以尝试增加stage数来扩大AIC提前量。但stage数不是越大越好，它会线性增加workspace容量和同步flag数量，也可能放大GM中间结果读写压力。

建议只有在下面条件同时满足时再增加stage：

| 条件                       | 说明                                         |
| -------------------------- | -------------------------------------------- |
| AIC单块耗时短于AIV单块耗时 | AIC有机会通过提前计算覆盖AIV                 |
| workspace带宽不是瓶颈      | 否则增加stage只会增加GM压力                  |
| 基本块数量足够多           | 太少时首尾flush开销占比高，增加stage收益有限 |

**3. 调整BlockScheduler和Swizzle**

样例根据`m > n`选择`GemmIdentityBlockSwizzle<3, 0>`，否则选择`<3, 1>`。调优时先用当前TileShape估算基本块数量：

```text
block_num = CeilDiv(M, L1TileShape::M) * CeilDiv(N, L1TileShape::N)
tail_blocks = block_num % aicCoreNum
```

若`tail_blocks`很小但非0，会出现尾轮长尾；若`block_num < aicCoreNum`，AIC利用率不足。此时优先调整`L1TileShape::M/N`，让基本块数接近AIC数量的整数倍，再微调swizzle方向和offset改善A/B访问局部性。

### 案例二：streamk_matmul尾轮提前计算

**样例结构**

`examples/37_streamk_matmul`实现Stream-K matmul。普通块仍由AIC直接计算并写回C，尾轮中不足以填满所有AIC的基本块沿K方向切分，产生更多工作单元，AIC把部分和写到workspace，AIV再归约得到结果C。

样例关键配置如下：

```cpp
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Streamk<enableUnitFlag, enableShuffleK>;
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;
using BlockScheduler = typename Gemm::Block::StreamkGemmIdentityBlockSwizzle<3, 0>;
constexpr uint32_t computeLength = 192 * 1024 / sizeof(ElementAccumulator);
using ReduceAdd =
    Catlass::Gemm::Kernel::StreamkReduceAdd<ArchTag, BlockScheduler, ElementAccumulator, ElementC, computeLength>;
```

Stream-K的详细原理可参考[StreamkMatmul说明](../../../../examples/102_dynamic_optimized_matmul/docs/zh/StreamkMatmul.md)。在CV融合视角下，它有两个关键点：

| 优化         | CV融合含义                                             |
| ------------ | ------------------------------------------------------ |
| 尾轮切K      | AIC把尾轮拆成更多K slice，减少尾轮AIC空闲              |
| 提前计算尾轮 | AIC提前产生尾轮部分和，使AIV归约可以与后续Cube计算重叠 |

**尾轮提前计算**

普通matmul的尾轮形态通常是：

```text
round 0..r-1: AIC计算完整基本块
round r:      少量AIC计算尾轮基本块，其余AIC空闲
```

Stream-K尾轮切分后，如果等所有普通块完成后再让AIV归约，Vector归约会落在kernel末尾，成为额外尾巴。样例中的调度把尾轮Stream-K块提前到倒数第二轮附近执行，使AIV能够更早开始读取workspace并归约部分和。抽象流水如下：

```text
时间片 t:     AIC计算提前的Stream-K尾轮部分和 -> workspace
时间片 t+1:   AIV归约尾轮部分和
时间片 t+1:   AIC继续计算普通完整块
```

这种方式本质上是用后续普通块的Cube计算时间掩盖Vector归约时间。

**性能瓶颈判断**

Stream-K适用于尾轮负载不均衡的shape，经验判断如下：

```text
B = CeilDiv(M, m1) * CeilDiv(N, n1) # m1/n1为切分任务块的L1TileShape::M/N取值
当 B / C > 1 且 B % C <= C * 0.8 时，可能收益较好
```

其中`C`为AIC数量。可以按下面方式判断是否应该使用Stream-K：

| 现象         | 判断                                           |
| ------------ | ---------------------------------------------- |
| `B % C == 0` | 无尾轮，Stream-K归约开销通常没有收益           |
| `B < C`      | M/N基本块太少，应优先考虑Split-K或更小M/N tile |
| `B / C == 1` | 只有一轮普通块，提前尾轮可掩盖空间有限         |
| 尾轮块数很少 | Stream-K更可能受益                             |
| K很短        | 切K后slice过少，AIV归约开销可能超过收益        |

**调优要点**

**1. 先调M/N基本块，再决定是否Stream-K**

Stream-K解决的是尾轮不均衡，不是所有负载不均衡问题。如果`L1TileShape::M/N`选择不合理导致基本块数量远小于AIC数量，应先调整TileShape。只有在完整轮次足够、尾轮块数不足时，Stream-K才更容易体现收益。

建议先计算：

```text
m_blocks = CeilDiv(M, L1TileShape::M)
n_blocks = CeilDiv(N, L1TileShape::N)
mn_blocks = m_blocks * n_blocks
normal_rounds = mn_blocks / aicCoreNum
tail_blocks = mn_blocks % aicCoreNum
k_slices_per_block = CeilDiv(K, L1TileShape::K)
streamk_work = tail_blocks * k_slices_per_block
```

`streamk_work`越接近或超过AIC数量，尾轮越容易被均摊；如果`k_slices_per_block`太小，尾轮可切分粒度不足。

**2. 控制AIV归约的UB压力**

`StreamkReduceAdd`用UB读取多个partial sum并做Add/Cast/写回。样例中`computeLength = 192 * 1024 / sizeof(ElementAccumulator)`，需要满足UB空间约束。归约时每个AIV处理的行数由`COMPUTE_LENGTH`、参与归约的AIV数量和tile列宽共同决定。

如果AIV归约成为瓶颈，可检查：

| 参数             | 影响                                          |
| ---------------- | --------------------------------------------- |
| `L1TileShape::N` | 决定每行partial sum长度，影响UB一次可处理行数 |
| `tail_blocks`    | 决定需要归约的Stream-K块数量                  |
| `splitkSliceNum` | 决定每个输出块要累加多少份partial sum         |
| `computeLength`  | 决定AIV每次搬运和归约的数据上限               |

调优方向通常是避免过大的`L1TileShape::N`导致AIV每次只能处理很少行，同时避免过小tile导致AIC普通块效率下降。

**3. 检查workspace大小和对齐**

Stream-K workspace大小为：

```text
max(2MB, L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator) * aicCoreNum * 2)
```

每个AIC有两片workspace，用于处理跨块时的前后两段partial sum。若Profiling显示MTE3或MTE2偏高，优先检查workspace起始地址、每片workspace步长和输出C矩阵stride是否满足高效搬运要求。对齐问题会直接影响AIC写partial sum和AIV读partial sum。

**4. 验证尾轮提前是否真正覆盖了归约**

观察流水图时重点看三件事：

| 目标        | 期望现象                                   |
| ----------- | ------------------------------------------ |
| AIC尾轮提前 | Stream-K块不全部集中在kernel末尾           |
| AIV归约重叠 | AIV Add/Cast/MTE3与后续AIC BlockMmad有重叠 |
| 末尾flush短 | kernel尾部只剩少量AIV归约和写回            |

如果AIV归约仍然集中在末尾，通常说明可提前计算的普通块不足、尾轮切分粒度不足或同步点过晚。

### 案例三：FAI多阶段CV流水

`examples/49_ascend950_flash_attention_infer`样例的详细性能优化分析在[FA算子性能优化](./FA_kernel_optimization.md)，本文只进行关键点概括。

FAI计算链路为：

```text
QK(AIC) -> Online Softmax(AIV) -> PV(AIC) -> RescaleO(AIV)
```

这是典型的多阶段CV融合。与quant_matmul和streamk_matmul相比，FAI有更强的数据依赖：Softmax依赖QK结果，PV依赖Softmax结果，RescaleO依赖PV结果和online softmax状态。因此调优重点不是简单增加并发，而是让相邻`kvSeq`块之间形成稳定流水。

FAI样例中的关键优化方向包括：

| 方向                   | 说明                                                               |
| ---------------------- | ------------------------------------------------------------------ |
| TileShape调整          | 增大`qSeqBase/kvSeqBase`可减少循环和同步次数，但会增加UB/L1/L0占用 |
| CV流水                 | 通过QK、Softmax、PV、RescaleO的跨块重叠减少AIC/AIV互等             |
| 片上中转               | AIV将Softmax结果P从UB搬到AIC侧L1，避免`UB -> GM -> L1`中转         |
| 多核负载均衡           | causal或稀疏mask下按有效block数切分，避免按q行均分造成长尾         |
| Paged Attention        | 检查block table、KV cache page连续性和对齐，避免MTE2跳读           |
| FixPipe和workspace对齐 | 保证输出和workspace分段起始地址按512B等要求对齐                    |

更详细的参数解释、资源预算和调优路径可参考[FA算子性能优化](./FA_kernel_optimization.md)。

## 融合算子调优检查表

### Tile和资源

| 检查项           | 说明                                       |
| ---------------- | ------------------------------------------ |
| L1/L0/UB是否超限 | Tile变大前先计算静态buffer占用             |
| M/N基本块数量    | 尽量避免`block_num < aicCoreNum`或尾轮过小 |
| K方向切分粒度    | Stream-K和Split-K需要足够K slice才能均衡   |
| epilogue tile    | quant_matmul中AIV tile要和AIC输出tile配平  |

### CV流水

| 检查项          | 说明                                           |
| --------------- | ---------------------------------------------- |
| AIC是否等待AIV  | 可能是AIV计算、搬运或stage不足                 |
| AIV是否等待AIC  | 可能是Cube、MTE2或FixPipe瓶颈                  |
| stage数是否合适 | stage过少覆盖不足，过多增加workspace和同步压力 |
| 首尾flush占比   | 小shape或少循环时融合流水收益有限              |

### 搬运和对齐

| 检查项       | 说明                                          |
| ------------ | --------------------------------------------- |
| 输入layout   | A/B/Q/K/V物理布局要匹配kernel的TLA/Layout定义 |
| GM workspace | 起始地址和分段offset尽量保证512B对齐          |
| stride       | 连续维stride应满足高效DataCopy/FixPipe要求    |
| 中间结果路径 | 优先片上中转；必须落GM时减少重复读写          |

### 负载均衡

| 检查项          | 说明                                |
| --------------- | ----------------------------------- |
| 每核基本块数    | Matmul类先看M/N基本块分布           |
| 每核有效block数 | FAI causal/mask场景按有效块估算权重 |
| 尾轮大小        | Stream-K重点看`B % C`和K slice数量  |
| swizzle         | Tile确定后再调swizzle方向和offset   |

## 总结

CV融合算子的性能优化核心是“配平”和“覆盖”：让AIC和AIV的单阶段耗时尽量接近，用stage、workspace或片上缓存把生产和消费重叠起来，同时避免切块、同步和GM中转带来的膨胀。

三个样例的侧重点不同：

| 样例           | 最重要的调优问题                                              |
| -------------- | ------------------------------------------------------------- |
| quant_matmul   | AIC int8 matmul和AIV反量化是否配平，workspace stage是否合适   |
| streamk_matmul | 尾轮是否足够不均衡，提前计算是否覆盖AIV归约                   |
| FAI            | QK/Softmax/PV/RescaleO是否形成稳定流水，多核有效block是否均衡 |
