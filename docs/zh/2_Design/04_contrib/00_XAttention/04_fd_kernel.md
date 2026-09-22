# FD 系列 Flash Attention 推理核设计文档

## 系列概述

FD 系列是 XFAI 系列（见 [03_xfai_kernel.md](./03_xfai_kernel.md)）在 **causal 三角 mask** 场景下的增强实现。其核心思路是：

- **GEMM 层完全复用 XFAI**：QK 与 PV 两个 BlockMmad 直接使用 `MmadAtlasA2XFAIQK` / `MmadAtlasA2XFAIPV`，不新增矩阵乘模板；
- **Epilogue 层新增 FD 专属模板**：`EpilogueAtlasA2OnlineSoftmax_FD`（在线 softmax，带参数化三角 mask）与 `EpilogueAtlasA2RescaleO_FD`（O 重缩放，支持 SplitKV），替换 XFAI 的对应 epilogue；
- **通用 CombineScale**：`EpilogueAtlasA2CombineScale` 是一个不绑定 FD 的通用合并模板，负责 shared（causal 三角部分）与 unshared（非共享上下文部分）两路 softmax 中间结果的合并，在 FD 推理场景与 `EpilogueAtlasA2RescaleO_FD` 配合完成最终输出。

FD 系列共 3 个新增模板：

| 模板 | 类型 | 文件 |
| --- | --- | --- |
| `EpilogueAtlasA2OnlineSoftmax_FD<LSE_MODE_>` | BlockEpilogue | `include/catlass/epilogue/block/block_epilogue_online_softmax_FD.hpp` |
| `EpilogueAtlasA2RescaleO_FD<LSE_MODE_>` | BlockEpilogue | `include/catlass/epilogue/block/block_epilogue_rescale_o_FD.hpp` |
| `EpilogueAtlasA2CombineScale` | BlockEpilogue | `include/catlass/epilogue/block/block_epilogue_combine_scale.hpp` |

其中 OnlineSoftmax_FD 与 CombineScale 在文件末尾均提供了 `EpilogueAscend950*` 同名特化（直接继承 AtlasA2 版本），用于 950 平台的 Policy 名注册。

## DispatchPolicy 定义

```cpp
// 1. FD 在线 softmax epilogue（QK 后：行归约 + P 下搬运）
template <LseMode LSE_MODE_ = LseMode::OUT_AND_LSE>
struct EpilogueAtlasA2OnlineSoftmax_FD {
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr EpilogueType TYPE = EpilogueType::ATLAS_A2;
};

// 2. FD 输出重缩放 epilogue（PV 后：O 缩放累加 + LSE 写出）
template <LseMode LSE_MODE_ = LseMode::OUT_AND_LSE>
struct EpilogueAtlasA2RescaleO_FD {
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr EpilogueType TYPE = EpilogueType::ATLAS_A2;
};

// 3. 通用两路合并 epilogue（shared/unshared softmax 结果合并）
struct EpilogueAtlasA2CombineScale {
    static constexpr EpilogueType TYPE = EpilogueType::ATLAS_A2;
};

// 950 平台特化（继承 AtlasA2 实现）
template <LseMode LSE_MODE_>
struct EpilogueAscend950OnlineSoftmax_FD : EpilogueAtlasA2OnlineSoftmax_FD<LSE_MODE_> {};
struct EpilogueAscend950CombineScale : EpilogueAtlasA2CombineScale {};
```

## EpilogueAtlasA2OnlineSoftmax_FD 设计

### 类形态与构造

```cpp
template <typename Policy_, typename OutputType_, typename InputType_, typename MaskType_>
class BlockEpilogue<EpilogueAtlasA2OnlineSoftmax_FD<LSE_MODE_>, OutputType_, InputType_, MaskType_> {
public:
    BlockEpilogue(Resource *resource, float scaleValue_);  // scaleValue = softmax 缩放因子 1/sqrt(d)
};
```

与 XFAI OnlineSoftmax 一致，构造时传入 softmax scale；区别在于 operator() 提供了带三角 mask 参数的版本。

### UB 空间布局

| 区域 | 偏移（单位：fp32 元素） | 大小 | 用途 |
| --- | --- | --- | --- |
| ls | 0 | 8192（乒乓） | S 子块行 softmax 中间量（分核暂存） |
| lp / mask / mask32 | 4*16384 | 共 4*16384 | P 中间量 / mask 比特 / mask 展开 |
| tv | 10*16384 | - | 向量计算临时区 |
| lm / hm / gm / gl / dm | 10*16384 + {0..5}*1024 | 各 1KB | 行内 max / 全局 max / 全局 sum / 重缩放系数 |
| mask16 | 11*16384 | - | 16bit mask 暂存 |

关键点：**dm（重缩放系数区）按 stackTile 周期分区**，`dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE(256) + rowOffset`，即每个 stackTile 周期拥有独立的 256 行 dm 槽位，避免跨周期覆盖。

### 行归约三分支

按行块长度选择归约策略（与 XFAI 相同）：

- **SPECTILE512**：三次级联 BlockReduceMax（512→256→…→1），适用于满行宽；
- **SPECTILE256**：`SetVecMask(32)` + `SetBlockReduceMask(4)` 的短行归约；
- **TAILTILE**：整段归约 + 尾段 `SetMask` 处理非对齐行。

### SubCoreCompute 六步流程（doTriUMask 模板参数）

```cpp
template <bool doTriUMask>
void SubCoreCompute(...) {
    // ① CalcLocalRowMax:  行内局部 max（lm）
    // ② UpdateGlobalRowMax: hm = max(lm, gm); dm = exp(gm - hm)  ← 产生本周期重缩放系数
    // ③ CalcExp:          p = exp(s * scale - hm)
    // ④ WaitFlag(V_MTE2)（非三角 mask 时）；DownCastP: p 降精度到 P 类型 + SetFlag(V_MTE3)
    // ⑤ CalcLocalRowSum:  gl += Σ exp(...)
    // ⑥ WaitFlag(V_MTE3) + CopyPUbToGm: P 写回 GM 供 PV GEMM 消费
}
```

### 双 operator() 与三角 mask 机制

提供两个调用入口：

1. **无 mask 版**：`SubCoreCompute<false>`，用于非 causal 或整块免 mask 的 stackTile；
2. **带 mask 版**：`SubCoreCompute<true>`，额外参数 `triUp / triDown / kvSStartIdx / kvSEndIdx / qkReady`。

三角 mask 偏移计算（causal 语义）：

```
if (triUp >= kvSStartIdx) {
    maskStart = RoundDown(triUp - kvSStartIdx, BLOCK);  // 三角起点折算到本 stackTile 内偏移
} else {
    全列有效（无需 mask）
}
if (triDown < kvSEndIdx) { maskEnd = ...; } else { 全列有效; }
```

即：triUp 为左上三角起点（query 相对位置），triDown 为右下止点，二者把当前 KV stackTile 划分为「全 mask / 部分三角 / 全有效」三段，仅部分三角段执行逐列 mask 计算。

## EpilogueAtlasA2RescaleO_FD 设计

### 类形态与构造

```cpp
template <typename Policy_, typename OutputType_, typename InputType_, typename UpdateType_, typename LseType_>
class BlockEpilogue<EpilogueAtlasA2RescaleO_FD<LSE_MODE_>, OutputType_, InputType_, UpdateType_, LseType_> {
public:
    BlockEpilogue(Resource *resource);  // 注意：无 scaleValue 参数
};
```

比 OnlineSoftmax 多两个类型参数：`UpdateType_`（中间结果 gOUpdate 的精度，通常 fp32）与 `LseType_`（LSE 输出精度）。

### UB 空间布局

| 区域 | 偏移（fp32 元素） | 用途 |
| --- | --- | --- |
| lo | 6*16384 | 上一周期 O 暂存（旧 O） |
| go | 8*16384 | 累计 O（16/32bit 双视图） |
| tv | 10*16384 | 向量临时区 |
| hm / gm | 10*16384 + {9,10}*1024 | 行 max |
| gl 与 lse32 共享 | 10*16384 + 12*1024 | 行 sum / LSE 中间量 |
| dm | 10*16384 + 13*1024 | 重缩放系数（与 OnlineSoftmax 的 dm 分区对齐） |

### 核心 algorithm：首 / 中 / 末 tile 三分支

对每个 stackTile 周期：

```
非首 tile:
    WaitFlag(V_MTE3, EVENT_ID3) → DataCopy lo(旧O) → SetFlag(MTE2_V, EVENT_ID0)
    Brcb dm 广播到 tv
    go = go * dm_block                      // 旧累计缩放
    WaitFlag(EVENT_ID0) 后 go = lo + go     // 累加当前 PV 结果
    SetFlag(V_MTE3, EVENT_ID3)

首 tile:
    go = lo（直接 DataCopy，无缩放）

末 tile (isLastStackTile):
    Brcb gl 广播 → go = go / gl             // 归一化
    Cast 到输出精度（bf16 用 CAST_RINT，否则 CAST_NONE；仅 !isSplitkv）
    !isSplitkv: CopyOToGm        → gOutput（bf16）
    isSplitkv:   CopyOToGmFp32   → gCombineo（fp32，不 cast，供后续 CombineScale 合并）
```

### CopyOToGm 三段式搬运

`CopyOToGm` / `CopyOToGmFp32` 均按 **prologue（前缀 token）/ integral（整 head）/ epilogue（尾 token）** 三段执行 `DataCopyPad`：每行搬运 `embed` 列有效数据 + `oHiddenSize - embed` 列 pad。行切分由 `rowNumTile = RoundDown(8192/embed, 8)` 决定，行循环内按 token-head 折算 `proTokenIdx / proTokenNum / integralHeadNum / epiTokenNum`。

### LSE 处理与中间结果回写

- **LSE_OUT 模式**：`isLastRowLoop` 时 `lse = ln(gl) + gm`，Brcb 广播后写 `gLse`（DataCopyPad，带 `(qHeads-1)*4` 列 pad）；
- **isSplitkv**：仅写 `gCombineLse`（fp32 中间量，供 CombineScale 合并）；
- **needRowLoop 且非末 tile**：`goUbTensor32` DataCopy 回 `gOUpdate`（fp32 中间结果回写 GM，供下个周期读入 lo）。

事件对：EVENT_ID0/1/3/4/5/6，其中 `MTE3_MTE2(EVENT_ID6)` 作为跨周期栅栏。

## EpilogueAtlasA2CombineScale 设计

### 类形态

```cpp
template <typename Policy_, typename OutputType_, typename InputType_>
class BlockEpilogue<EpilogueAtlasA2CombineScale, OutputType_, InputType_> {
public:
    BlockEpilogue(Resource *resource);
};
```

仅两个类型参数（无 mask / lse 模板参数），是纯通用合并模板。

### UB 空间布局（按字节）

| 区域 | 偏移 | 说明 |
| --- | --- | --- |
| sharedOut | 0 - 64k | shared 路 O 中间量（32k 乒乓） |
| unsharedOut | 64k - 128k | unshared 路 O 中间量 |
| sharedGl / unsharedGl | 128k / 132k | 两路行 sum |
| sharedGm / unsharedGm | 136k / 140k | 两路行 max |
| realGm / realGl | 144k / 148k | 合并后全局 max / sum |
| out | 152k - 184k | 输出暂存 |

### 合并算法（五步）

```
① BlockReduceMax(8→1) 压缩 sharedGm/Gl（shared 路每行按 SOFTMAX_BROAD_SIZE=8 重复存放）
② realGm = max(sharedGm, unsharedGm)
③ α = exp(sharedGm - realGm);  β = exp(unsharedGm - realGm)
   gl = sharedGl * α + unsharedGl * β
④ Brcb 广播 α/β → Out = sharedOut * α + unsharedOut * β
⑤ Brcb realGl 广播 → Out = Out / realGl（归一化）
→ Cast 到 ElementOutput（bf16 用 CAST_RINT）→ DataCopy gFinalOutput
```

### 流水组织

主循环 `rowLoopNum + preLoad=1` 双缓冲（pingpongFlag 切换）：加载段 DataCopy 两路 gm/gl（shared 路每行 8 个，`sumMaxOffsetIoShared = row * 8`；unshared 路整行，非 8 对齐用 DataCopyPad）与两路 Out；计算段执行上述五步。事件对 `MTE3_MTE2` 乒乓 + `EVENT_ID4`（MTE2_V / V_MTE3）。

## 使用示例

以下摘自 xllm-ops（https://gitcode.com/xLLM-AI/xllm_ops）真实工程 `x_flash_attention_infer/op_kernel/x_flash_attention_infer_fd.h`。

### 类型组装（FDInfer 入口）

```cpp
// GEMM 层：复用 XFAI Policy
using DispatchPolicyQK = MmadAtlasA2XFAIQK<PagedCacheFlag, false>;
using DispatchPolicyPV = MmadAtlasA2XFAIPV;

// Epilogue 层：FD 专属 + 通用 CombineScale
using EpilogueOnlineSoftmax = BlockEpilogue<
    EpilogueAtlasA2OnlineSoftmax_FD<lseMode>, PType, SType, maskType>;
using EpilogueRescaleO = BlockEpilogue<
    EpilogueAtlasA2RescaleO_FD<lseMode>, OType, OTmpType, OUpdateType, LseType>;
using CombineScale = Epilogue::Block::CombineScale<OType, LseType>;  // 预组装别名

// 组装 FD 推理核
template <...> using FAInferKernelFD = ...;
```

### 构造与 causal 三分支调用（vec 核）

```cpp
// 构造：OnlineSoftmax 带 scale，其余仅传 resource
EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
EpilogueRescaleO epilogueRescaleO(resource);
CombineScale combineScale(resource);

// causal 分支参数
int32_t triUp   = noSkipKvS - qSBlockSize;   // 左上三角起点
int32_t triDown = noSkipKvS;                 // 右下止点
int32_t kvSStartIdx = kvSIdx * pagedBlockSize;
bool doTriUMask = triUp < kvSEndIdx - 1;     // mask 长度仅 1 时相当于不加

if (doTriUMask) {
    // 带 mask 版：13 参数，含 qkReady/triUp/triDown/kvSStartIdx/kvSEndIdx
    epilogueOnlineSoftmax(gS, gPUb, ..., qkReady, triUp, triDown, kvSStartIdx, kvSEndIdx, ...);
} else {
    // 无 mask 版
    epilogueOnlineSoftmax(gS, gPUb, ..., isLastStackTile, ...);
}
// softmaxReady 通过 CrossCoreSetFlag<0x2, PIPE_MTE3> 通知 PV GEMM
```

### RescaleO 调用（PV 后，vec 核）

```cpp
Arch::CrossCoreWaitFlag(pvReady);   // 等待 PV GEMM 完成
epilogueRescaleO(
    gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
    gLse[gmOffsetLse], gmlse[gmlse0ffset], gmlo[gmlooffset],
    layoutO, layoutOTmp, layoutUpdate, layoutLse,
    actualBlockShapePV, qSBlockSize, qNBlockSize,
    (stackSeqCount - PRE_LAUNCH == 0),          // isFirstStackTile
    nowkvSIdx + blockStackNum >= kvEnd,          // isLastStackTile
    curStackTileMod, isSplitKV,
    layoutgmLse, layoutgmLo);                    // splitkv 中间量布局
```

### CombineScale 调用（全核 SyncAll 后）

```cpp
AscendC::SyncAll();
combineScale(
    qHeads,                          // 头数
    extraInfo->totalSplitNodeNum,    // split 槽数
    embedV,                          // V 头维度
    extraInfo,                       // SplitInfo（TilingData 中定义）
    gmlse,                           // LSE 中间量
    gmlo,                            // O 中间量（workspace）
    gO,                              // 最终输出
    gActualQseqlen,                  // 实际 seq len
    true);
```

另见 `x_attention/op_kernel/x_attention_catlass_helper.h` 中 `CallCombineScale`：以 `EpilogueAtlasA2CombineScale` + `BlockEpilogue<Policy, OutputType(INPUT_T), InputType(float)>` 组装独立 CombineScaleKernel，在 vec 核 SyncAll 后调用。

## 与 XFAI 系列差异对比

| 维度 | XFAI（EpilogueAtlasA2OnlineSoftmax/RescaleO） | FD（本系列） |
| --- | --- | --- |
| mask 能力 | token mask 布尔式（整块生效） | triUp/triDown 参数化三角 mask（逐列生效） |
| operator() 入口 | 单一版本 | 无 mask 版 + 带 mask 版（doTriUMask 模板分支） |
| dm 存储 | 单周期覆盖 | 按 stackTile 周期分区（256 行/周期） |
| RescaleO 类型参数 | 4 个 | 5 个（新增 UpdateType_/LseType_） |
| SplitKV 输出路由 | gCombineo/gCombineLse | isSplitkv 分支：fp32→gCombineo / bf16→gOutput |
| CombineScale | SplitKV N-way 合并（RescaleO 内） | 独立通用模板，shared/unshared 两路合并 |
| GEMM Policy | XFAI 专属 | 完全复用 XFAI（无新增 GEMM） |
