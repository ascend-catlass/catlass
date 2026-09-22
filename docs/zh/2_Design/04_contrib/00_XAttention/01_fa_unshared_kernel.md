# Unshared FA 推理系列模板设计

本系列模板面向 **Unshared（非共享 KV）FlashAttention 推理场景**：每个 attention head 的 KV 有效序列长度各不相同（如 Multi-LoRA / 独立上下文场景），因此 softmax 阶段需要按 head 构造独立的 mask，且 KV 序列较短、可一次性处理完，无需在线 rescale 累积输出 O。

系列包含以下模板：

| 模板 | 层级 | Policy | 源文件 |
| --- | --- | --- | --- |
| BlockMmad（QK） | GEMM | `Gemm::MmadAtlasA2UnsharedFAQK` | `include/catlass/gemm/block/block_mmad_unshared_fa_qk.hpp` |
| BlockMmad（PV） | GEMM | `Gemm::MmadAtlasA2UnsharedFAPV` | `include/catlass/gemm/block/block_mmad_unshared_fa_pv.hpp` |
| BlockEpilogue（Softmax） | Epilogue | `Epilogue::EpilogueAtlasA2FAUnsharedSoftmax` / `EpilogueAscend950FAUnsharedSoftmax` | `include/catlass/epilogue/block/block_epilogue_fa_unshared_softmax.hpp` |

三者的协作关系：

```
Q @ K^T ──► (S) ──► FAUnsharedSoftmax ──► P(f16/bf16), gm(rowMax), gl(rowSum)
   UnsharedFAQK                      │
                                     └─(跨核通知 softmaxReady)─► P @ V ──► OTmp(f32) ──► 外部 combine: O = OTmp / gl
                                                                   UnsharedFAPV
```

## DispatchPolicy 定义

`include/catlass/gemm/dispatch_policy.hpp`：

```cpp
struct MmadAtlasA2UnsharedFAQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2UnsharedFAPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};
```

`include/catlass/epilogue/dispatch_policy.hpp`：

```cpp
struct EpilogueAtlasA2FAUnsharedSoftmax {
    using ArchTag = Arch::AtlasA2;
};

struct EpilogueAscend950FAUnsharedSoftmax {
    using ArchTag = Arch::Ascend950;
};
```

## BlockMmad MmadAtlasA2UnsharedFAQK 设计方案

### 模板参数

```cpp
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
          class BiasType_, class TileCopy_, class TileMmad_>
struct BlockMmad<MmadAtlasA2UnsharedFAQK, L1TileShape_, L0TileShape_, AType_, BType_, CType_,
                 BiasType_, TileCopy_, TileMmad_>;
```

- `L1TileShape_`：L1 级 tile 形状，QK 场景典型值 `GemmShape<128, 256, 128>`（N 为单次搬运的 KV tile 长度）。
- `L0TileShape_`：L0 级 tile 形状，一般与 L1 相同。
- `AType_`/`BType_`：Q（RowMajor）与 K（ColumnMajor）的 `GemmType<Element, Layout>`。
- `CType_`：输出 S 矩阵，**仅支持 RowMajor**（`static_assert` 约束）。

### 内存布局

- L1：`l1A` 与 `l1B` 连续分配（`l1B` 起始偏移为 `L1A_SIZE = M*K*sizeof(ElementA)`）。
- L0A/L0B/L0C：按 `STAGES = 2` 计算 pingpong buffer 尺寸（本模板单次调用完成一个 block，实际按单缓冲使用）。
- L0C 中 S 的布局为 `layout::zN`（MMAD 原生输出布局）。

### 事件同步与执行流程

构造函数完成缓冲初始化，并预置三组硬件事件（`EVENT_ID0`）：`MTE1_MTE2`、`M_MTE1`、`FIX_M`；析构函数对称 Wait，保证 Kernel 退出时流水排空。

`operator()` 单次调用内完成一个 block 的完整搬运与计算：

1. `WaitFlag(MTE1_MTE2)` → GM→L1 搬运 A（Q）→ `SetFlag(MTE2_MTE1)` → Wait；
2. `WaitFlag(M_MTE1)` → L1→L0A（`copyL1ToL0A`）；
3. GM→L1 搬运 B（K^T）→ L1→L0B；
4. `tileMmad(l0C, l0A, l0B, mRound, nRound, actualShape.k())` 执行 MMAD；
5. FIX 通路：`copyL0CToGm` 将 S 写回 GM（供 softmax epilogue 读取）。

m/n/k 维度均按 `L1AlignHelper` 对齐规则 `RoundUp` 后参与 L0 布局，`tileMmad` 的 K 用真实值 `actualShape.k()`。

## BlockMmad MmadAtlasA2UnsharedFAPV 设计方案

### 与 FAQK 的差异

- 数据通路相同（GM→L1→L0→MMAD→L0C→GM），差异在于 **A（即 P 矩阵）的搬运时机由跨核同步控制**。

### 跨核同步设计

```cpp
void operator()(..., GemmCoord actualShape, Arch::CrossCoreFlag softmaxReady)
{
    // 1. 先搬运 B（V）：无需依赖 softmax 结果
    copyGmToL1B(l1BTensor, gB, layoutBInL1, layoutTileB);
    ...
    copyL1ToL0B(...);

    // 2. 跨核等待 softmax 完成通知（P 已写出 GM）
    Arch::CrossCoreWaitFlag(softmaxReady);
    copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutTileA);  // gA 即 P
    ...
}
```

PV GEMM 与 softmax epilogue 通常运行在不同核（或不同 subBlock）上：V 的搬运与 softmax 计算**重叠执行**，等 softmax 发出 `softmaxReady` 通知后才开始搬 P，从而隐藏 softmax 时延。构造/析构仅维护 `MTE1_MTE2 (EVENT_ID2)` 一组事件，QK 阶段使用的 `EVENT_ID0` 组保留给 MMAD/FIX 流水。

PV 场景典型 L1TileShape 为 `GemmShape<128, 128, 256>`（K 维为 KV 序列长度方向，单 tile 内完成整个短序列）。

## BlockEpilogue EpilogueAtlasA2FAUnsharedSoftmax 设计方案

### 模板参数与构造

```cpp
BlockEpilogue(Arch::Resource<ArchTag>& resource, float tor_,
              uint32_t unsharedKvSeqLen, uint32_t maxDecodeStep,
              uint32_t headNum, uint32_t groupSize);
```

- `OutputType_`：P 矩阵类型（fp16/bfloat16）。
- `InputType_`：S 矩阵类型（float）。
- `MaskType_`：mask 类型。
- 构造参数：softmax scale `tor`、每个 head 的 KV 有效长度 `unsharedKvSeqLen`、最大 decode 步数、head 数与 GQA group 大小。

### UB 内存布局

| Tensor | 元素类型 | 起始偏移（字节） | 用途 |
| --- | --- | --- | --- |
| lsUbTensor | float | 0 | S 分数矩阵（加 scale 后） |
| lpUbTensor32 / tvUbTensor16 | float / Output | `2 * 32768` | P（cast 后）复用区 |
| lmUbTensor | float | `3 * 32768` | 行最大值 rowMax |
| llUbTensor | float | `3 * 32768 + 4 * 512` | 行和 rowSum |
| tvUbTensor | float | `3 * 32768 + 8 * 512` | 临时向量（Brcb 展开等） |
| unsharedMaskUbTensor | float | `3 * 32768 + 12 * 512` | 按 head 构造的加法 mask |

### Unshared mask 的设备侧构造（InitUnsharedMaskV2）

mask 尺寸为 `[headNum * groupSize, kSeqTileRound]`（`kSeqTileRound = ceil(maxDecodeStep*headNum / 8) * 8`）。构造逻辑：

1. 先整体填充 `lowest()`（负无穷，经 Add 加到 S 上等效屏蔽）；
2. 对每个 head（`colOffset = (headOffset + round) * maxDecodeStep`），将 `[colOffset, colOffset + unsharedKvSeqLen)` 区间填充 0（等效不屏蔽）；
3. 考虑 8 元素 block 对齐（`FLOAT_BLOCK_SIZE`），未对齐尾部用 0/lowest 二次修正；
4. 按 subBlock（双核）将 head 数一分为二，各自构造本核负责的行段。

由于每个 head 的有效区间起点随 `round * maxDecodeStep` 平移，**不同 head 的 mask 不同**——这正是 "Unshared" 的含义。

### 算法流程（SubCoreCompute）

```
S = DataCopy(GM)                          // MTE2, EVENT_ID3
S = S * tor                               // Muls scale
S = S + unsharedMask                      // 加法 mask（越界位置变 -inf）
lm = ReduceMax(S, row)                    // WholeReduceMax + Max 折叠（>128 列时）
S = S - broadcast(lm)                     // Brcb 展开行最大值后逐行相减
ls = Exp(S)
lp = Cast(ls, f32 -> f16/bf16)            // P 矩阵
ll = ReduceSum(ls, row)                   // WholeReduceSum + Add 折叠
写出: P -> GM（DataCopy, T_BLOCK_SIZE=16 对齐）
      gm(lm), gl(ll) -> GM（按 head 偏移，非 8 对齐时用 DataCopyPad）
```

由于 KV 序列一次处理完，本模板**不做 O 的 rescale 与归一化**，`gm/gl` 交由外部（上层 kernel 或 combine kernel）完成 `O = OTmp / gl`。双 subBlock 场景下按 head 切分行区间并行计算，事件全部使用 `EVENT_ID3` 避免与 GEMM 侧冲突。

## Ascend950 特化

`EpilogueAscend950FAUnsharedSoftmax` 直接继承 AtlasA2 实现（Ascend C 向量 API 兼容）：

```cpp
template <class OutputType_, class InputType_, class MaskType_>
class BlockEpilogue<EpilogueAscend950FAUnsharedSoftmax, OutputType_, InputType_, MaskType_>
    : public BlockEpilogue<EpilogueAtlasA2FAUnsharedSoftmax, OutputType_, InputType_, MaskType_> {
    using Base::Base;
};
```

## 使用示例

摘自 xllm-ops（https://gitcode.com/xLLM-AI/xllm_ops）`x_attention/op_kernel/x_attention_catlass_helper.h` 的 `CallUnsharedInferKernel`（真实工程用法）：

```cpp
using QKL1TileShape = GemmShape<128, 256, 128>;
using QKL0TileShape = QKL1TileShape;
using MmadDispatchPolicyQK = Gemm::MmadAtlasA2UnsharedFAQK;
using BlockMmadQK = Gemm::Block::BlockMmad<MmadDispatchPolicyQK, QKL1TileShape, QKL0TileShape,
                                           QType, KType, SType>;

using DispatchPolicyFAUnsharedSoftmax = Epilogue::EpilogueAtlasA2FAUnsharedSoftmax;
using EpilogueFAUnsharedSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyFAUnsharedSoftmax,
                                                                 PType, SType, maskType>;

using PVL1TileShape = GemmShape<128, 128, 256>;
using MmadDispatchPolicyPV = Gemm::MmadAtlasA2UnsharedFAPV;
using BlockMmadPV = Gemm::Block::BlockMmad<MmadDispatchPolicyPV, PVL1TileShape, PVL0TileShape,
                                           PType, VType, OTmpType>;

using UnsharedFAInferKernel = UnsharedFAInferKernel<BlockMmadQK, BlockMmadPV,
                                                    EpilogueFAUnsharedSoftmax, isPAEnabled>;
UnsharedFAInferKernel unsharedInferKernel(tilingData);
unsharedInferKernel(params);
```

类型约定：Q/K/V/P/O/Mask 为 `INPUT_T`（fp16/bf16），S/OTmp 为 float；上层 `UnsharedFAInferKernel` 负责 QK→softmax→PV 的调度与 `softmaxReady` 跨核通知。

## 与其他系列的差异

- 相比 Shared FA 系列（`FAIQKSplitRow` + `OnlineSoftmaxCopySumMax` + `RescaleOWithoutDivSum`）：Unshared 场景 KV 短且每 head 独立，softmax **单趟完成**，不维护跨 KV tile 的在线状态（rowMax/rowSum 的历史累积与 O rescale），因此无需 SplitRow 与 RescaleO 模板。
- 相比 FAI（PagedAttention 共享前缀）系列：不依赖 blockTables 分页 KV 索引，mask 在设备侧按 head 静态构造。
