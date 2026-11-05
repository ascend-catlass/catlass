# FD Series Flash Attention Inference Kernel Design Document

## Series Overview

The FD series is an enhanced implementation of the XFAI series (see [03_xfai_kernel.md](./03_xfai_kernel.md)) for the **causal triangular mask** scenario. Its core idea is:

- **The GEMM layer fully reuses XFAI**: the QK and PV BlockMmad templates directly use `MmadAtlasA2XFAIQK` / `MmadAtlasA2XFAIPV`, with no new matrix-multiply template;
- **The Epilogue layer adds FD-specific templates**: `EpilogueAtlasA2OnlineSoftmax_FD` (online softmax with a parameterized triangular mask) and `EpilogueAtlasA2RescaleO_FD` (O rescaling with SplitKV support), replacing the corresponding XFAI epilogues;
- **A general CombineScale**: `EpilogueAtlasA2CombineScale` is a general merge template not bound to FD. It merges the softmax intermediate results of the shared (causal triangular part) and unshared (non-shared context part) paths, and works together with `EpilogueAtlasA2RescaleO_FD` in the FD inference scenario to produce the final output.

The FD series adds 3 templates in total:

| Template | Type | File |
| --- | --- | --- |
| `EpilogueAtlasA2OnlineSoftmax_FD<LSE_MODE_>` | BlockEpilogue | `include/catlass/epilogue/block/block_epilogue_online_softmax_FD.hpp` |
| `EpilogueAtlasA2RescaleO_FD<LSE_MODE_>` | BlockEpilogue | `include/catlass/epilogue/block/block_epilogue_rescale_o_FD.hpp` |
| `EpilogueAtlasA2CombineScale` | BlockEpilogue | `include/catlass/epilogue/block/block_epilogue_combine_scale.hpp` |

Both OnlineSoftmax_FD and CombineScale provide an `EpilogueAscend950*` specialization with the same name at the end of the file (directly inheriting the Atlas A2 version), used to register the Policy name for the 950 platform.

## DispatchPolicy Definition

```cpp
// 1. FD online softmax epilogue (after QK: row reduction + P downcast transfer)
template <LseMode LSE_MODE_ = LseMode::OUT_AND_LSE>
struct EpilogueAtlasA2OnlineSoftmax_FD {
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr EpilogueType TYPE = EpilogueType::ATLAS_A2;
};

// 2. FD output rescale epilogue (after PV: O scale accumulation + LSE write-out)
template <LseMode LSE_MODE_ = LseMode::OUT_AND_LSE>
struct EpilogueAtlasA2RescaleO_FD {
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr EpilogueType TYPE = EpilogueType::ATLAS_A2;
};

// 3. general two-path merge epilogue (merge of shared/unshared softmax results)
struct EpilogueAtlasA2CombineScale {
    static constexpr EpilogueType TYPE = EpilogueType::ATLAS_A2;
};

// 950 platform specializations (inheriting the AtlasA2 implementation)
template <LseMode LSE_MODE_>
struct EpilogueAscend950OnlineSoftmax_FD : EpilogueAtlasA2OnlineSoftmax_FD<LSE_MODE_> {};
struct EpilogueAscend950CombineScale : EpilogueAtlasA2CombineScale {};
```

## EpilogueAtlasA2OnlineSoftmax_FD Design

### Class Form and Construction

```cpp
template <typename Policy_, typename OutputType_, typename InputType_, typename MaskType_>
class BlockEpilogue<EpilogueAtlasA2OnlineSoftmax_FD<LSE_MODE_>, OutputType_, InputType_, MaskType_> {
public:
    BlockEpilogue(Resource *resource, float scaleValue_);  // scaleValue = softmax scaling factor 1/sqrt(d)
};
```

Consistent with XFAI OnlineSoftmax, the softmax scale is passed at construction; the difference is that operator() provides a version with triangular mask parameters.

### UB Space Layout

| Area | Offset (unit: fp32 element) | Size | Purpose |
| --- | --- | --- | --- |
| ls | 0 | 8192 (pingpong) | row softmax intermediate of the S sub-block (staged per core) |
| lp / mask / mask32 | 4*16384 | 4*16384 in total | P intermediate / mask bits / expanded mask |
| tv | 10*16384 | - | vector computation temporary area |
| lm / hm / gm / gl / dm | 10*16384 + {0..5}*1024 | 1KB each | in-row max / global max / global sum / rescale coefficient |
| mask16 | 11*16384 | - | 16-bit mask staging |

Key point: the **dm (rescale coefficient area) is partitioned by stackTile cycle**, `dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE(256) + rowOffset`, that is, each stackTile cycle owns independent dm slots for 256 rows, avoiding cross-cycle overwrite.

### Three Branches of Row Reduction

The reduction strategy is selected by the row block length (the same as XFAI):

- **SPECTILE512**: three cascaded BlockReduceMax (512→256→…→1), for the full row width;
- **SPECTILE256**: short-row reduction with `SetVecMask(32)` + `SetBlockReduceMask(4)`;
- **TAILTILE**: whole-segment reduction plus a tail `SetMask` to handle unaligned rows.

### Six-Step SubCoreCompute Flow (doTriUMask template parameter)

```cpp
template <bool doTriUMask>
void SubCoreCompute(...) {
    // 1. CalcLocalRowMax:  in-row local max (lm)
    // 2. UpdateGlobalRowMax: hm = max(lm, gm); dm = exp(gm - hm)  <- produces the rescale coefficient of this cycle
    // 3. CalcExp:          p = exp(s * scale - hm)
    // 4. WaitFlag(V_MTE2) (when not a triangular mask); DownCastP: p downcast to the P type + SetFlag(V_MTE3)
    // 5. CalcLocalRowSum:  gl += Sum exp(...)
    // 6. WaitFlag(V_MTE3) + CopyPUbToGm: P is written back to GM for the PV GEMM to consume
}
```

### Dual operator() and Triangular Mask Mechanism

Two call entries are provided:

1. **Version without mask**: `SubCoreCompute<false>`, for non-causal stackTiles or stackTiles that are entirely mask-free;
2. **Version with mask**: `SubCoreCompute<true>`, with the extra parameters `triUp / triDown / kvSStartIdx / kvSEndIdx / qkReady`.

Triangular mask offset computation (causal semantics):

```
if (triUp >= kvSStartIdx) {
    maskStart = RoundDown(triUp - kvSStartIdx, BLOCK);  // convert the triangle start into an offset within this stackTile
} else {
    full column valid (no mask needed)
}
if (triDown < kvSEndIdx) { maskEnd = ...; } else { full column valid; }
```

That is: triUp is the start of the upper-left triangle (relative query position) and triDown is the lower-right end point; together they divide the current KV stackTile into three segments: "fully masked / partially triangular / fully valid", and only the partially triangular segment performs per-column mask computation.

## EpilogueAtlasA2RescaleO_FD Design

### Class Form and Construction

```cpp
template <typename Policy_, typename OutputType_, typename InputType_, typename UpdateType_, typename LseType_>
class BlockEpilogue<EpilogueAtlasA2RescaleO_FD<LSE_MODE_>, OutputType_, InputType_, UpdateType_, LseType_> {
public:
    BlockEpilogue(Resource *resource);  // note: no scaleValue parameter
};
```

It has two more type parameters than OnlineSoftmax: `UpdateType_` (precision of the intermediate result gOUpdate, usually fp32) and `LseType_` (LSE output precision).

### UB Space Layout

| Area | Offset (fp32 element) | Purpose |
| --- | --- | --- |
| lo | 6*16384 | staging of the previous cycle O (old O) |
| go | 8*16384 | accumulated O (16/32-bit dual view) |
| tv | 10*16384 | vector temporary area |
| hm / gm | 10*16384 + {9,10}*1024 | row max |
| gl shared with lse32 | 10*16384 + 12*1024 | row sum / LSE intermediate |
| dm | 10*16384 + 13*1024 | rescale coefficient (aligned with the dm partitioning of OnlineSoftmax) |

### Core Algorithm: First / Middle / Last Tile Branching

For each stackTile cycle:

```
non-first tile:
    WaitFlag(V_MTE3, EVENT_ID3) → DataCopy lo(old O) → SetFlag(MTE2_V, EVENT_ID0)
    Brcb dm broadcast to tv
    go = go * dm_block                      // scale the old accumulation
    after WaitFlag(EVENT_ID0) go = lo + go     // accumulate the current PV result
    SetFlag(V_MTE3, EVENT_ID3)

first tile:
    go = lo (direct DataCopy, no scaling)

last tile (isLastStackTile):
    Brcb gl broadcast → go = go / gl             // normalization
    Cast to the output precision (bf16 uses CAST_RINT, otherwise CAST_NONE; only when !isSplitkv)
    !isSplitkv: CopyOToGm        -> gOutput (bf16)
    isSplitkv:   CopyOToGmFp32   -> gCombineo (fp32, no cast, for the subsequent CombineScale merge)
```

### Three-Segment CopyOToGm Transfer

Both `CopyOToGm` / `CopyOToGmFp32` perform `DataCopyPad` in three segments — **prologue (prefix tokens) / integral (whole heads) / epilogue (tail tokens)**: each row transfers `embed` valid columns + `oHiddenSize - embed` padded columns. The row split is determined by `rowNumTile = RoundDown(8192/embed, 8)`, and within the row loop `proTokenIdx / proTokenNum / integralHeadNum / epiTokenNum` are derived from the token-head conversion.

### LSE Handling and Intermediate Result Write-Back

- **LSE_OUT mode**: on `isLastRowLoop`, `lse = ln(gl) + gm`, then after Brcb broadcast it is written to `gLse` (DataCopyPad, with `(qHeads-1)*4` column padding);
- **isSplitkv**: only write `gCombineLse` (fp32 intermediate, for the CombineScale merge);
- **needRowLoop and not the last tile**: `goUbTensor32` is DataCopy'd back to `gOUpdate` (fp32 intermediate result written back to GM, for the next cycle to read as lo).

Event pairs: EVENT_ID0/1/3/4/5/6, where `MTE3_MTE2(EVENT_ID6)` acts as the cross-cycle barrier.

## EpilogueAtlasA2CombineScale Design

### Class Form

```cpp
template <typename Policy_, typename OutputType_, typename InputType_>
class BlockEpilogue<EpilogueAtlasA2CombineScale, OutputType_, InputType_> {
public:
    BlockEpilogue(Resource *resource);
};
```

Only two type parameters (no mask / lse template parameters), it is a purely general merge template.

### UB Space Layout (in bytes)

| Area | Offset | Description |
| --- | --- | --- |
| sharedOut | 0 - 64k | O intermediate of the shared path (32k pingpong) |
| unsharedOut | 64k - 128k | O intermediate of the unshared path |
| sharedGl / unsharedGl | 128k / 132k | row sum of the two paths |
| sharedGm / unsharedGm | 136k / 140k | row max of the two paths |
| realGm / realGl | 144k / 148k | global max / sum after merging |
| out | 152k - 184k | output staging |

### Merge Algorithm (Five Steps)

```
1. BlockReduceMax(8→1) compresses sharedGm/Gl (each row of the shared path is stored repeated by SOFTMAX_BROAD_SIZE=8)
2. realGm = max(sharedGm, unsharedGm)
3. α = exp(sharedGm - realGm);  β = exp(unsharedGm - realGm)
   gl = sharedGl * α + unsharedGl * β
4. Brcb broadcasts α/β → Out = sharedOut * α + unsharedOut * β
5. Brcb broadcasts realGl → Out = Out / realGl (normalization)
→ Cast to ElementOutput (bf16 uses CAST_RINT) → DataCopy gFinalOutput
```

### Pipeline Organization

The main loop uses `rowLoopNum + preLoad=1` double buffering (pingpongFlag toggling): the load segment DataCopies the two paths of gm/gl (8 per row for the shared path, `sumMaxOffsetIoShared = row * 8`; the whole row for the unshared path, using DataCopyPad when not 8-aligned) and the two paths of Out; the compute segment executes the five steps above. Event pairs `MTE3_MTE2` pingpong + `EVENT_ID4` (MTE2_V / V_MTE3).

## Usage Example

The following is excerpted from the real xllm-ops (https://gitcode.com/xLLM-AI/xllm_ops) engineering file `x_flash_attention_infer/op_kernel/x_flash_attention_infer_fd.h`.

### Type Assembly (FDInfer Entry)

```cpp
// GEMM layer: reuse the XFAI Policy
using DispatchPolicyQK = MmadAtlasA2XFAIQK<PagedCacheFlag, false>;
using DispatchPolicyPV = MmadAtlasA2XFAIPV;

// Epilogue layer: FD-specific + general CombineScale
using EpilogueOnlineSoftmax = BlockEpilogue<
    EpilogueAtlasA2OnlineSoftmax_FD<lseMode>, PType, SType, maskType>;
using EpilogueRescaleO = BlockEpilogue<
    EpilogueAtlasA2RescaleO_FD<lseMode>, OType, OTmpType, OUpdateType, LseType>;
using CombineScale = Epilogue::Block::CombineScale<OType, LseType>;  // pre-assembled alias

// assemble the FD inference kernel
template <...> using FAInferKernelFD = ...;
```

### Construction and Causal Three-Branch Invocation (vec core)

```cpp
// construction: OnlineSoftmax takes a scale, the rest take only resource
EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
EpilogueRescaleO epilogueRescaleO(resource);
CombineScale combineScale(resource);

// causal branch parameters
int32_t triUp   = noSkipKvS - qSBlockSize;   // start of the upper-left triangle
int32_t triDown = noSkipKvS;                 // lower-right end point
int32_t kvSStartIdx = kvSIdx * pagedBlockSize;
bool doTriUMask = triUp < kvSEndIdx - 1;     // when the mask length is only 1 it is equivalent to no mask

if (doTriUMask) {
    // version with mask: 13 parameters, including qkReady/triUp/triDown/kvSStartIdx/kvSEndIdx
    epilogueOnlineSoftmax(gS, gPUb, ..., qkReady, triUp, triDown, kvSStartIdx, kvSEndIdx, ...);
} else {
    // version without mask
    epilogueOnlineSoftmax(gS, gPUb, ..., isLastStackTile, ...);
}
// softmaxReady is notified to the PV GEMM through CrossCoreSetFlag<0x2, PIPE_MTE3>
```

### RescaleO Invocation (after PV, vec core)

```cpp
Arch::CrossCoreWaitFlag(pvReady);   // wait for the PV GEMM to finish
epilogueRescaleO(
    gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
    gLse[gmOffsetLse], gmlse[gmlse0ffset], gmlo[gmlooffset],
    layoutO, layoutOTmp, layoutUpdate, layoutLse,
    actualBlockShapePV, qSBlockSize, qNBlockSize,
    (stackSeqCount - PRE_LAUNCH == 0),          // isFirstStackTile
    nowkvSIdx + blockStackNum >= kvEnd,          // isLastStackTile
    curStackTileMod, isSplitKV,
    layoutgmLse, layoutgmLo);                    // splitkv intermediate layout
```

### CombineScale Invocation (after all-core SyncAll)

```cpp
AscendC::SyncAll();
combineScale(
    qHeads,                          // number of heads
    extraInfo->totalSplitNodeNum,    // number of split slots
    embedV,                          // V head dimension
    extraInfo,                       // SplitInfo (defined in TilingData)
    gmlse,                           // LSE intermediate
    gmlo,                            // O intermediate (workspace)
    gO,                              // final output
    gActualQseqlen,                  // actual seq len
    true);
```

See also `CallCombineScale` in `x_attention/op_kernel/x_attention_catlass_helper.h`: it assembles an independent CombineScaleKernel with `EpilogueAtlasA2CombineScale` + `BlockEpilogue<Policy, OutputType(INPUT_T), InputType(float)>`, and calls it after SyncAll on the vec core.

## Differences from the XFAI Series

| Dimension | XFAI (EpilogueAtlasA2OnlineSoftmax/RescaleO) | FD (this series) |
| --- | --- | --- |
| mask capability | token mask boolean style (effective for the whole block) | triUp/triDown parameterized triangular mask (effective per column) |
| operator() entry | a single version | version without mask + version with mask (doTriUMask template branch) |
| dm storage | overwritten each cycle | partitioned by stackTile cycle (256 rows/cycle) |
| RescaleO type parameters | 4 | 5 (adds UpdateType_/LseType_) |
| SplitKV output routing | gCombineo/gCombineLse | isSplitkv branch: fp32→gCombineo / bf16→gOutput |
| CombineScale | SplitKV N-way merge (inside RescaleO) | independent general template, merge of the shared/unshared two paths |
| GEMM Policy | XFAI-specific | fully reuses XFAI (no new GEMM) |
