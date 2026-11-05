# Design of the Unshared FA Inference Template Series

This template series targets the **Unshared (non-shared KV) Flash Attention inference scenario**: the effective KV sequence length of each attention head differs (for example, in Multi-LoRA / independent-context scenarios). Therefore, the softmax stage must build an independent mask per head, the KV sequence is short and can be processed in one pass, and there is no need to accumulate the output O with online rescaling.

The series contains the following templates:

| Template | Level | Policy | Source file |
| --- | --- | --- | --- |
| BlockMmad (QK) | GEMM | `Gemm::MmadAtlasA2UnsharedFAQK` | `include/catlass/gemm/block/block_mmad_unshared_fa_qk.hpp` |
| BlockMmad (PV) | GEMM | `Gemm::MmadAtlasA2UnsharedFAPV` | `include/catlass/gemm/block/block_mmad_unshared_fa_pv.hpp` |
| BlockEpilogue (Softmax) | Epilogue | `Epilogue::EpilogueAtlasA2FAUnsharedSoftmax` / `EpilogueAscend950FAUnsharedSoftmax` | `include/catlass/epilogue/block/block_epilogue_fa_unshared_softmax.hpp` |

The cooperation among the three:

```
Q @ K^T ──► (S) ──► FAUnsharedSoftmax ──► P(f16/bf16), gm(rowMax), gl(rowSum)
   UnsharedFAQK                      │
                                     └─(cross-core notify softmaxReady)─► P @ V ──► OTmp(f32) ──► external combine: O = OTmp / gl
                                                                   UnsharedFAPV
```

## DispatchPolicy Definition

`include/catlass/gemm/dispatch_policy.hpp`:

```cpp
struct MmadAtlasA2UnsharedFAQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2UnsharedFAPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};
```

`include/catlass/epilogue/dispatch_policy.hpp`:

```cpp
struct EpilogueAtlasA2FAUnsharedSoftmax {
    using ArchTag = Arch::AtlasA2;
};

struct EpilogueAscend950FAUnsharedSoftmax {
    using ArchTag = Arch::Ascend950;
};
```

## BlockMmad MmadAtlasA2UnsharedFAQK Design

### Template Parameters

```cpp
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
          class BiasType_, class TileCopy_, class TileMmad_>
struct BlockMmad<MmadAtlasA2UnsharedFAQK, L1TileShape_, L0TileShape_, AType_, BType_, CType_,
                 BiasType_, TileCopy_, TileMmad_>;
```

- `L1TileShape_`: the L1-level tile shape. A typical value in the QK scenario is `GemmShape<128, 256, 128>` (N is the KV tile length moved in a single transfer).
- `L0TileShape_`: the L0-level tile shape, generally the same as L1.
- `AType_`/`BType_`: the `GemmType<Element, Layout>` of Q (RowMajor) and K (ColumnMajor).
- `CType_`: the output S matrix, which **only supports RowMajor** (enforced by `static_assert`).

### Memory Layout

- L1: `l1A` and `l1B` are allocated contiguously (the start offset of `l1B` is `L1A_SIZE = M*K*sizeof(ElementA)`).
- L0A/L0B/L0C: the pingpong buffer sizes are computed with `STAGES = 2` (this template completes one block in a single call and is effectively used as a single buffer).
- The layout of S in L0C is `layout::zN` (the native MMAD output layout).

### Event Synchronization and Execution Flow

The constructor initializes the buffers and presets three hardware events (`EVENT_ID0`): `MTE1_MTE2`, `M_MTE1`, and `FIX_M`; the destructor performs symmetric Wait calls to ensure the pipeline is drained when the Kernel exits.

A single `operator()` call completes the full transfer and computation of one block:

1. `WaitFlag(MTE1_MTE2)` → GM→L1 transfer of A (Q) → `SetFlag(MTE2_MTE1)` → Wait;
2. `WaitFlag(M_MTE1)` → L1→L0A (`copyL1ToL0A`);
3. GM→L1 transfer of B (K^T) → L1→L0B;
4. `tileMmad(l0C, l0A, l0B, mRound, nRound, actualShape.k())` performs MMAD;
5. FIX path: `copyL0CToGm` writes S back to GM (for the softmax epilogue to read).

The m/n/k dimensions all participate in the L0 layout after `RoundUp` according to the `L1AlignHelper` alignment rules, and `tileMmad` uses the real value `actualShape.k()` for K.

## BlockMmad MmadAtlasA2UnsharedFAPV Design

### Differences from FAQK

- The data path is the same (GM→L1→L0→MMAD→L0C→GM). The difference is that **the timing of the A (that is, the P matrix) transfer is controlled by cross-core synchronization**.

### Cross-Core Synchronization Design

```cpp
void operator()(..., GemmCoord actualShape, Arch::CrossCoreFlag softmaxReady)
{
    // 1. Transfer B (V) first: no dependency on the softmax result
    copyGmToL1B(l1BTensor, gB, layoutBInL1, layoutTileB);
    ...
    copyL1ToL0B(...);

    // 2. Cross-core wait for the softmax completion notification (P has been written to GM)
    Arch::CrossCoreWaitFlag(softmaxReady);
    copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutTileA);  // gA is P
    ...
}
```

The PV GEMM and the softmax epilogue usually run on different cores (or different subBlocks): the transfer of V and the softmax computation **overlap**, and the transfer of P starts only after softmax sends the `softmaxReady` notification, thereby hiding the softmax latency. The constructor/destructor only maintain the `MTE1_MTE2 (EVENT_ID2)` event group, while the `EVENT_ID0` group used by the QK stage is reserved for the MMAD/FIX pipeline.

A typical L1TileShape in the PV scenario is `GemmShape<128, 128, 256>` (the K dimension runs along the KV sequence length, so the whole short sequence is completed within a single tile).

## BlockEpilogue EpilogueAtlasA2FAUnsharedSoftmax Design

### Template Parameters and Construction

```cpp
BlockEpilogue(Arch::Resource<ArchTag>& resource, float tor_,
              uint32_t unsharedKvSeqLen, uint32_t maxDecodeStep,
              uint32_t headNum, uint32_t groupSize);
```

- `OutputType_`: the P matrix type (fp16/bfloat16).
- `InputType_`: the S matrix type (float).
- `MaskType_`: the mask type.
- Construction parameters: the softmax scale `tor`, the effective KV length of each head `unsharedKvSeqLen`, the maximum decode steps, the number of heads, and the GQA group size.

### UB Memory Layout

| Tensor | Element type | Start offset (bytes) | Purpose |
| --- | --- | --- | --- |
| lsUbTensor | float | 0 | S score matrix (after adding scale) |
| lpUbTensor32 / tvUbTensor16 | float / Output | `2 * 32768` | P (after cast) reuse area |
| lmUbTensor | float | `3 * 32768` | Row maximum rowMax |
| llUbTensor | float | `3 * 32768 + 4 * 512` | Row sum rowSum |
| tvUbTensor | float | `3 * 32768 + 8 * 512` | Temporary vector (Brcb expansion, etc.) |
| unsharedMaskUbTensor | float | `3 * 32768 + 12 * 512` | Additive mask built per head |

### Device-Side Construction of the Unshared Mask (InitUnsharedMaskV2)

The mask size is `[headNum * groupSize, kSeqTileRound]` (`kSeqTileRound = ceil(maxDecodeStep*headNum / 8) * 8`). The construction logic:

1. First fill everything with `lowest()` (negative infinity, which masks out the position when added to S);
2. For each head (`colOffset = (headOffset + round) * maxDecodeStep`), fill the interval `[colOffset, colOffset + unsharedKvSeqLen)` with 0 (equivalent to not masking);
3. Account for 8-element block alignment (`FLOAT_BLOCK_SIZE`) and correct the unaligned tail with 0/lowest a second time;
4. Split the number of heads in two by subBlock (dual core), each constructing the row segment it is responsible for.

Because the start of the effective interval of each head shifts with `round * maxDecodeStep`, **the mask differs between heads** — this is exactly the meaning of "Unshared".

### Algorithm Flow (SubCoreCompute)

```
S = DataCopy(GM)                          // MTE2, EVENT_ID3
S = S * tor                               // Muls scale
S = S + unsharedMask                      // additive mask (out-of-range positions become -inf)
lm = ReduceMax(S, row)                    // WholeReduceMax + Max fold (when there are >128 columns)
S = S - broadcast(lm)                     // subtract row by row after Brcb expanding the row maximum
ls = Exp(S)
lp = Cast(ls, f32 -> f16/bf16)            // P matrix
ll = ReduceSum(ls, row)                   // WholeReduceSum + Add fold
write out: P -> GM (DataCopy, T_BLOCK_SIZE=16 aligned)
      gm(lm), gl(ll) -> GM (by head offset, use DataCopyPad when not 8-aligned)
```

Because the KV sequence is processed in one pass, this template **does not perform rescale and normalization of O**; `gm/gl` are handed to the outside (the upper-level kernel or the combine kernel) to compute `O = OTmp / gl`. In the dual subBlock scenario, the row interval is split by head for parallel computation, and all events use `EVENT_ID3` to avoid conflicts with the GEMM side.

## Ascend 950 Specialization

`EpilogueAscend950FAUnsharedSoftmax` directly inherits the Atlas A2 implementation (the AscendC vector APIs are compatible):

```cpp
template <class OutputType_, class InputType_, class MaskType_>
class BlockEpilogue<EpilogueAscend950FAUnsharedSoftmax, OutputType_, InputType_, MaskType_>
    : public BlockEpilogue<EpilogueAtlasA2FAUnsharedSoftmax, OutputType_, InputType_, MaskType_> {
    using Base::Base;
};
```

## Usage Example

Excerpted from `CallUnsharedInferKernel` in `x_attention/op_kernel/x_attention_catlass_helper.h` of xllm-ops (https://gitcode.com/xLLM-AI/xllm_ops) (real engineering usage):

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

Type conventions: Q/K/V/P/O/Mask are `INPUT_T` (fp16/bf16), and S/OTmp are float; the upper-level `UnsharedFAInferKernel` is responsible for scheduling QK→softmax→PV and for the `softmaxReady` cross-core notification.

## Differences from Other Series

- Compared with the Shared FA series (`FAIQKSplitRow` + `OnlineSoftmaxCopySumMax` + `RescaleOWithoutDivSum`): in the Unshared scenario the KV is short and independent per head, so softmax **completes in a single pass** and does not maintain online state across KV tiles (historical accumulation of rowMax/rowSum and O rescale); therefore SplitRow and RescaleO templates are not needed.
- Compared with the FAI series (PagedAttention shared prefix): it does not rely on the blockTables paged KV index, and the mask is statically constructed on the device side per head.
