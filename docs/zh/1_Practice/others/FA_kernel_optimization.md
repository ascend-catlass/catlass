# Flash Attention Infer算子性能优化

## 写在前面

该文档主要说明，如何从算子结构和源码实现出发，对`examples/49_ascend950_flash_attention_infer`中的FlashAttention Infer(FAI)样例进行性能分析和调优。本文主要从以下几个角度进行FAI样例调优方案分析：

- FAI样例的主要性能瓶颈可能来自哪里。
- 哪些参数是编译期模板参数，哪些参数来自tiling或runtime输入。
- 调整TileShape、CV流水、分核策略、输入布局或Paged Attention参数时，会影响哪些片上资源和执行路径。

## 样例介绍

该样例面向Flash Attention推理前向算子`FlashAttentionInfer`，计算流程为：

```text
S = Q * K^T * scale + mask
P = softmax(S)
O = P * V
```

在源码实现中，该计算被拆成四个主要阶段：

| 阶段           | 执行单元 | 主要工作                        | 数据流      |
| -------------- | -------- | ------------------------------- | ----------- |
| QK             | AIC      | 计算`Q * K^T`                   | Q/K -> S    |
| Online Softmax | AIV      | 处理mask、max、sum、exp并生成P  | S -> P      |
| PV             | AIC      | 计算`P * V`                     | P/V -> Otmp |
| RescaleO       | AIV      | 按online softmax状态归一化更新O | Otmp -> O   |

FAI计算涉及多个CV操作，优化FAI算子的目标集中于让`QK -> Online Softmax -> PV -> RescaleO`这条链路在AIC/AIV之间尽量形成稳定流水，并在多核之间尽量避免长尾。

当前样例的几个关键实现点如下：

```cpp
// examples/49_ascend950_flash_attention_infer/fai_tiling.h
constexpr int32_t BLOCK_BASE_SIZE = 128;

// examples/49_ascend950_flash_attention_infer/fai_kernel.h
using L1TileShape = tla::Shape<_128, _128, _128>;
using L0TileShape = L1TileShape;

// examples/49_ascend950_flash_attention_infer/fai_kernel_utils.h
constexpr uint32_t KERNEL_TASK_NUM = 3;
```

`L1TileShape`三个维度分别对应`qSeqBase`、`kvSeqBase`和`embedBase`。当前样例中`L1TileShape::K`要求与head dimension匹配，默认适用于`headDim = 128`的场景。

需要特别注意，`BLOCK_BASE_SIZE`和`L1TileShape`不是两个完全独立的参数。`BLOCK_BASE_SIZE`用于tiling阶段统计q/kv基本块和多核权重，`L1TileShape`用于kernel实际执行的tile形状。只修改其中一侧，可能导致tiling权重模型和kernel实际循环粒度不一致。

## 优化方案

### 调整TileShape参数组合

TileShape决定每次QK、Softmax、PV和RescaleO处理的基本粒度。较大的`qSeqBase`或`kvSeqBase`通常可以减少外层循环次数、同步次数和小块搬运次数，但会增加UB、L1和L0资源占用，也可能在mask或尾块场景中引入更多无效计算。

调整前至少需要检查以下静态资源项：

```cpp
MM1_RESULT_SIZE = qSeqBase / CV_RATIO * kvSeqBase * sizeof(ElementS);
MM2_RESULT_SIZE = qSeqBase / CV_RATIO * Align64(embedBase) * sizeof(ElementOTmp);
SHARE_UB_SIZE = CeilDiv(qSeqBase, NUM2) * sizeof(ElementS);
MM2_LEFT_SIZE = qSeqBase * kvSeqBase * sizeof(ElementP);
```

这些资源项主要影响：

| 资源        | 主要占用                                                            |
| ----------- | ------------------------------------------------------------------- |
| UB          | `bmm1TensorList[2]`、`bmm2TensorList[2]`、`sumUb`、`expUb`、`maxUb` |
| L1          | `mm2AL1TensorList[KERNEL_TASK_NUM]`                                 |
| L0A/L0B/L0C | QK/PV的BlockMmad tile                                               |
| 同步事件    | C1/V1、V1/C2、C2/V2之间的流水同步                                   |

常见调整方向如下：

| 现象                          | 调整方向                                 | 风险                                |
| ----------------------------- | ---------------------------------------- | ----------------------------------- |
| q/kv外层循环次数多，同步频繁  | 增大`qSeqBase`或`kvSeqBase`              | UB/L1/L0占用上升                    |
| AIV Softmax或RescaleO频繁启动 | 优先增大`qSeqBase`                       | q尾块浪费可能增加                   |
| QK/PV小块matmul过多           | 优先增大`kvSeqBase`                      | causal或稀疏mask下无效kv块可能增加  |
| causal mask尾块浪费明显       | 谨慎增大`kvSeqBase`，必要时减小kv tile   | 循环和同步次数会上升                |
| `headDim != 128`              | 同步修改`L1TileShape::K`并检查静态buffer | BlockMmad和中间buffer约束可能不满足 |

如果只是把TileShape调大而不计算资源预算，优化很容易变成随机试错。更合理的做法是先根据目标shape列出候选`qSeqBase/kvSeqBase/embedBase`，确认UB/L1/L0资源可容纳，再做性能验证。

### 优化AIC/AIV协同流水

FAI中，AIC负责QK/PV矩阵乘，AIV负责Online Softmax和RescaleO。虽然同一个`kvSeq`块内部存在阶段依赖，但相邻`kvSeq`块之间可以形成流水重叠。理想steady-state形态为：

```text
task t:     AIC QK
task t+1:   AIV OnlineSoftmax
task t+2:   AIC PV
task t+3:   AIV RescaleO
```

源码中与流水相关的关键对象包括：

| 对象                                 | 作用                       |
| ------------------------------------ | -------------------------- |
| `taskId`                             | 流水任务编号               |
| `runInfo[4]`                         | 保存不同流水阶段的执行状态 |
| `bmm1TensorList[2]`                  | QK输出UB双缓冲             |
| `bmm2TensorList[2]`                  | PV输出UB双缓冲             |
| `mm2AL1TensorList[KERNEL_TASK_NUM]`  | Softmax结果P的L1缓存       |
| `sumUb/maxUb/expUb[KERNEL_TASK_NUM]` | Online Softmax状态缓存     |

同步路径如下：

| 同步flag          | 方向       | 含义                         |
| ----------------- | ---------- | ---------------------------- |
| `SYNC_C1_V1_FLAG` | AIC -> AIV | QK结果就绪，AIV可执行Softmax |
| `SYNC_V1_C2_FLAG` | AIV -> AIC | P已搬到L1，AIC可执行PV       |
| `SYNC_C2_V2_FLAG` | AIC -> AIV | PV结果就绪，AIV可更新O       |

样例中的关键数据通路是AIV将Softmax结果P从UB直接搬到AIC侧L1：

```cpp
using CopyUbToL1P = Tile::CopyUb2L1Tla<ArchTag, decltype(vf1OutUb), TensorDst>;
```

该路径避免了`UB -> GM -> L1`中转，降低GM带宽压力，也减少PV启动前的等待。

- 如果AIC经常等待AIV，通常说明Vector侧成为瓶颈：应优先检查Softmax、mask读取、RescaleO、小`qSeqBase`导致的频繁AIV调度，以及`KERNEL_TASK_NUM`是否不足以覆盖Vector延迟。
- 如果AIV经常等待AIC，通常说明Cube或搬运成为瓶颈：应优先检查QK/PV tile粒度、K/V布局连续性、MTE2搬运和Paged Attention跳读。

增加`KERNEL_TASK_NUM`可能提升流水覆盖能力，但它也会增加L1中的`mm2AL1TensorList`占用、UB中的Softmax状态占用、同步事件压力以及末尾flush阶段复杂度。因此应先确认TileShape和多核切分合理，再考虑调整流水深度。

### 优化多核负载均衡

Causal或稀疏mask场景下，不同`qSeq`块可见的`kvSeq`块数量不同。如果简单按qSeq行数平均分核，早期和后期q块的有效计算量可能差异很大，长尾核会决定整体耗时。

FAI在tiling阶段按有效block数做贪心切分，核心逻辑位于`examples/49_ascend950_flash_attention_infer/fai_tiling.h`：

| 函数或字段                | 作用                                         |
| ------------------------- | -------------------------------------------- |
| `GetCalcBlockNumsOneHead` | 统计单head下mask后的有效`qSeq x kvSeq`块数量 |
| `ComputeSplitNBSeq`       | 沿Batch/Head/qSeq三轴做贪心切分              |
| `bnAxisStartIdx`          | 记录每个AI Core的起始batch/head              |
| `sparseStartIdx`          | 记录每个AI Core的起始qSeq block              |

可以用以下抽象权重理解分核策略：

```text
core_weight[i] = sum(valid_kv_blocks for qSeq blocks assigned to core i)
target_weight = total_valid_blocks / blockDim
imbalance = max(core_weight) / average(core_weight)
```

如果核间有效block数差距很大，应优先检查mask模式、`actualSeqLengths`、`actualSeqLengthsKV`和`blockDim`。如果有效任务数本身很少，使用过多AI Core只会增加同步和调度开销。Paged Attention场景下还要注意，block table跳读成本没有完全体现在有效block数量中；即使各核有效block数接近，也可能因为KV cache物理布局不同而出现耗时差异。

贪心分核也有局限：当单个qSeq block的权重大于目标core weight时，它无法继续拆分该block；当batch很小、qSeq很短或mask裁剪后有效块很少时，也无法填满所有core。这类场景下应考虑减少`blockDim`、调整`qSeqBase`，或重新设计更细粒度的切分策略。

### 优化输入布局、MTE2和FixPipe

当性能受搬运影响时，首先要确认业务侧输入物理布局与kernel TLA布局一致。当前样例中，Q/O按`[batch * qSeqlen, qHeads * embed]`方向连续，V按`[batch * kvSeqlen, kvHeads * embed]`方向连续，K使用ColumnMajor以适配`Q * K^T`访问。

MTE2相关问题通常来自：

- 上层实际layout与kernel TLA layout不一致。
- K/V在QK或PV访问路径上跨大stride。
- GQA场景中`qHeads / kvHeads`分组不连续。
- mask张量不是按`[batch * qSeqlen, kvSeqlen]`连续存放。
- Paged Attention下block table导致KV cache频繁跳读。

如果业务侧无法改变输入物理布局，可以考虑新增专用`LayoutTag`或TileCopy适配。但这类修改会同时影响QK、PV和mask访问路径，必须重新验证正确性和性能。

FixPipe相关问题通常出现在O写回路径，尤其是小`embed`、尾块多、O地址不对齐或workspace分段不对齐时。建议保证Q/K/V/O首地址尽量按512Byte对齐，连续维stride尽量满足搬运对齐要求。若后续引入GM workspace缓存中间结果，分配多个workspace分段时应使用类似`RoundUp(offset, 512)`的方式保证每段起始地址对齐。

### 优化Paged Attention路径

启用`PAGED_CACHE_FLAG`后，kernel会按`blockSize`对`kvSeqlen`向上取整，并通过`blockTables`间接访问KV cache：

```cpp
kvSeqlen = RoundUp(kvSeqlen, blockSize);
```

`blockSize`过小会减少尾块浪费，但会增加block table访问次数和KV跳读概率；`blockSize`过大有利于page内连续访问，但可能增加尾块无效计算。优化Paged Attention时，应优先保证KV cache page内部连续、page起始地址对齐、block table顺序符合kernel访问顺序，并确保`actualSeqLengthsKV`准确参与分核。

如果只有Paged Attention场景出现MTE2偏高或核间耗时差异，优先检查block table和KV cache物理布局，而不是直接缩小TileShape。TileShape只能改变单次计算粒度，不能消除由page组织方式造成的随机跳读。

## 总结

融合算子场景可以参考此案例优化
