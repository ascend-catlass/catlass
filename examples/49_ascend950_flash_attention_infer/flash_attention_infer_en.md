# CATLASS FlashAttention Infer Design Document

## Overview

CATLASS FlashAttention Infer is a FlashAttention inference operator optimized for Ascend950 hardware, implemented based on the CATLASS Gemm API. The operator structure consists of the following parts:

- Tiling calculation;
- Kernel implementation;
- Block and Epilogue components in the Kernel that are suitable for FlashAttention inference operations;
- The Block and Epilogue components used rely on the Tile components provided by the template library.

This document describes in detail the Kernel implementation of the Flash Attention Infer operator, including key designs such as main process logic, Cube/Vector pipeline, L1/UB memory allocation, as well as the Tiling segmentation strategy.

### Operator Functionality

The Flash Attention Infer operator implements the following calculation process:

```text
O = FlashAttention(Q, K, V, mask)
  = softmax(Q * K^T / sqrt(d)) * V
```

Where:

- **Q**: Query tensor with shape `[batch, qSeqlen, qHeads, headDim]`
- **K**: Key tensor with shape `[batch, kvSeqlen, kvHeads, headDim]`
- **V**: Value tensor with shape `[batch, kvSeqlen, kvHeads, headDim]`
- **mask**: Attention mask (optional)
- **O**: Output tensor with shape `[batch, qSeqlen, qHeads, headDim]`

<img src="../../docs/assets/images/flash_attention_infer_operator.png" width="40%">

- Supports **GQA** (Grouped Query Attention) functionality.

- Supports **Paged Attention** mode, implementing paged management of KV Cache through blockTable.

- Supports **Attention Mask** functionality, with two mask modes: top-left and bottom-right.

- **CV pipeline preload**, AIC and AIV multi-level pipeline parallelism to improve computational efficiency.

- The current template Kernel does not support the variable-length ActualSeq feature.

<img src="../../docs/assets/images/flash_attention_infer_cv_pipeline.png" width="50%">

---

## Data Structures and Type Definitions

### Core Type Definitions

```cpp
// L1 Tile Shape: <qSeqlen, kvSeqlen, embed>
using L1TileShape = tla::Shape<_128, _128, _128>;
using L0TileShape = L1TileShape;

// Data types
using ElementQ = Dtype;              // Q element type (FP16/BF16)
using ElementK = Dtype;              // K element type
using ElementV = Dtype;              // V element type
using ElementS = float;              // Score type (FP32)
using ElementP = Dtype;              // P type (OnlineSoftmax intermediate result)
using ElementO = Dtype;              // O element type
using ElementOTmp = float;           // O temporary type (FP32)
using ElementMask = uint8_t;         // Mask type

// Layout types
using LayoutTagQ = layout::RowMajor;   // Q: Row-major
using LayoutTagK = layout::ColumnMajor; // K: Column-major
using LayoutTagV = layout::RowMajor;   // V: Row-major
using LayoutTagS = layout::RowMajor;   // S: Row-major
using LayoutTagP = layout::zN;         // P: zN format
using LayoutTagO = layout::RowMajor;   // O: Row-major
```

### Core Component Types

```cpp
// Q * K^T matrix multiplication component
using DispatchPolicyQK = Gemm::MmadFAIQK<ArchTag, enablePaFlag>;
using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<...>;
using TileMmadQK = Gemm::Tile::TileMmadTla<...>;
using BlockMmadQK = Gemm::Block::BlockMmadTla<...>;

// Online Softmax component
using DispatchPolicySoftmax = Epilogue::EpilogueAscend950FASoftmax<enableMaskFlag>;
using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<...>;

// P * V matrix multiplication component
using DispatchPolicyPV = Gemm::MmadFAIPV<ArchTag, enablePaFlag>;
using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<...>;
using TileMmadPV = Gemm::Tile::TileMmadTla<...>;
using BlockMmadPV = Gemm::Block::BlockMmadTla<...>;

// O update and normalization component
using DispatchPolicyRescaleO = Epilogue::EpilogueAscend950FARescaleO;
using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<...>;
```

#### Block Mmad

The operator uses two types of Block Mmad components:

- `BlockMmadQK` is a partial specialization of the BlockMmad template class, used to handle the matrix multiplication of Q and K in FlashAttention Infer. Header file: [block_mmad_fai_qk_tla.hpp](../../include/catlass/gemm/block/block_mmad_fai_qk_tla.hpp).
- `BlockMmadPV` is a partial specialization of the BlockMmad template class, used to handle the matrix multiplication of P and V in FlashAttention Infer. Header file: [block_mmad_fai_pv_tla.hpp](../../include/catlass/gemm/block/block_mmad_fai_pv_tla.hpp).

#### Block Epilogue

The operator uses two types of Block Epilogue components:

- `EpilogueOnlineSoftmax` is a partial specialization of the BlockEpilogue template class, used to handle the online softmax operation in FlashAttention Infer. Header file: [block_epilogue_fa_softmax_ascend950.hpp](../../include/catlass/epilogue/block/block_epilogue_fa_softmax_ascend950.hpp).
- `EpilogueRescaleO` is a partial specialization of the BlockEpilogue template class, used to handle the rescaleO operation in FlashAttention Infer. Header file: [block_epilogue_fa_rescale_o_ascend950.hpp](../../include/catlass/epilogue/block/block_epilogue_fa_rescale_o_ascend950.hpp).

#### Tile Mmad & Tile Copy

In the Block components used by the Kernel, the TileMmadTla component in tile_mmad.hpp and the PackedTileCopyTlaToUB component in tile_copy.hpp are used, and new TileCopySoftmax and TileCopyRescaleO components for FA Epilogue processing are added, as well as the ub->l1 path CopyUb2L1Tla component newly added for Ascend950, for example:

```c++
using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK, ElementS, LayoutTagS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;

using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<
        ArchTag, ElementMask, ElementP, LayoutTagMask, LayoutTagP>;

using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutTagO, LayoutTagOTmp>;

using CopyUbToL1P = Tile::CopyUb2L1Tla<ArchTag, decltype(vf1OutUb), TensorDst>;
```

These Tile components are responsible for data movement between GM, L1, L0 and UB, as well as the underlying implementation of matrix multiplication and Softmax. PackedTileCopyTlaToUB supports TLA (Tensor Layout Abstraction) layout, which can efficiently handle data movement requirements of different layouts. Tile::CopyUb2L1Tla supports direct movement of calculation results on AIV Ub to AIC L1, achieving efficiency improvement compared with the previous Ub->GM->L1 movement implementation.

---

## Main Kernel Class: FAInferKernel

### Class Definition

**Location**:`fai_kernel.h:47-545`

```cpp
template <
    class BlockMmadQK,           // Q*K^T matrix multiplication component
    class BlockMmadPV,           // P*V matrix multiplication component
    class EpilogueOnlineSoftmax,  // Online Softmax component
    class EpilogueRescaleO,      // O update component
    bool PAGED_CACHE_FLAG>       // Whether to enable Paged Attention
class FAInferKernel {
    // ...
};
```

### Member Variables

| Variable Name                       | Type                     | Description                                  |
| ----------------------------------- | ------------------------ | -------------------------------------------- |
| `bmm1TensorList[NUM2]`              | `LocalTensor<ElementS>`    | Double buffer for BMM1 results (S matrix)    |
| `bmm2TensorList[NUM2]`              | `LocalTensor<ElementOTmp>` | Double buffer for BMM2 results (O temporary) |
| `mm2AL1TensorList[KERNEL_TASK_NUM]` | `LocalTensor<ElementP>`    | L1 cache for MM2 input (P matrix)            |
| `expUb[KERNEL_TASK_NUM]`            | `LocalTensor<ElementS>`    | UB cache for exp values                      |
| `sumUb[KERNEL_TASK_NUM]`            | `LocalTensor<ElementS>`    | UB cache for sum values                      |
| `maxUb[KERNEL_TASK_NUM]`            | `LocalTensor<ElementS>`    | UB cache for max values                      |
| `constInfo`                         | ConstInfo                | Constant information                         |
| `runInfo[4]`                        | RunInfo                  | Runtime information (circular buffer)        |
| `runParam`                          | RunParamStr              | Runtime parameters                           |
| `blockIdx`                          | uint32_t                 | Current AI Core index                        |
| `subBlockIdx`                       | uint32_t                 | AIV core sub-index                           |

### Memory Layout

#### Memory Layout

```text
+------------------------+  ubBufAddrStart = 0
| bmm1TensorList[0]      |  MM1_RESULT_SIZE = 64 * 128 * sizeof(float) = 32KB
+------------------------+
| bmm2TensorList[0]      |  MM2_RESULT_SIZE = 64 * 128 * sizeof(float) = 32KB
+------------------------+
| bmm1TensorList[1]      |  MM1_RESULT_SIZE = 32KB
+------------------------+
| bmm2TensorList[1]      |  MM2_RESULT_SIZE = 32KB
+------------------------+
| expUb[0]              |  SHARE_UB_SIZE = 64 * sizeof(float) = 256B (AIV)
+------------------------+
| maxUb[0]              |  SHARE_UB_SIZE = 256B (AIV)
+------------------------+
| sumUb[0]              |  SHARE_UB_SIZE = 256B (AIV)
+------------------------+
| expUb[1]              |  SHARE_UB_SIZE = 256B (AIV)
+------------------------+
| maxUb[1]              |  SHARE_UB_SIZE = 256B (AIV)
+------------------------+
| sumUb[1]              |  SHARE_UB_SIZE = 256B (AIV)
+------------------------+
| expUb[2]              |  SHARE_UB_SIZE = 256B (AIV)
+------------------------+
| maxUb[2]              |  SHARE_UB_SIZE = 256B (AIV)
+------------------------+
| sumUb[2]              |  SHARE_UB_SIZE = 256B (AIV)
+------------------------+
| Epilogue Internal Buf        |  Reallocated inside Epilogue
+------------------------+
```

#### L1 Memory Layout

```text
+------------------------+  l1BufAddrStart = 0
| mm2AL1TensorList[0]    |  MM2_LEFT_SIZE = 128 * 128 * sizeof(Dtype) = 32KB
+------------------------+
| mm2AL1TensorList[1]    |  MM2_LEFT_SIZE = 32KB
+------------------------+
| mm2AL1TensorList[2]    |  MM2_LEFT_SIZE = 32KB
+------------------------+
| BlockMmadQK internal L1     |  Reallocated inside Block
+------------------------+
| BlockMmadPV internal L1     |  Reallocated inside Block
+------------------------+
```

---

## Multi-Core Tiling Algorithm

The Tiling segmentation algorithm is mainly used to evenly distribute computing tasks to multiple AI Cores to achieve efficient parallel computing.

### Core Objectives

- **Load balancing**: Make the computation load of each AI core as balanced as possible
- **Memory efficiency**: Rational utilization of on-chip memory (UB/L1)
- **Parallel optimization**: Maximum multi-core parallelism

### Data Structures

#### FAInfo - Input Parameter Structure

```cpp
struct FAInfo {
    int64_t batchSize = 0;           // Batch size
    int64_t numOfHeads = 0;          // Number of Query heads
    int64_t numOfKVHeads = 0;        // Number of Key/Value heads (for GQA)
    int64_t seqSize = 0;             // Query sequence length
    int64_t seqInnerSize = 0;        // Key/Value sequence length
    int64_t headSize = 0;            // Dimension of each head

    uint32_t numBlocks = 0;          // Total number of blocks (PFA)
    uint32_t blockSize = 0;          // Number of tokens per block
    uint32_t maxBlockNumPerBatch = 0; // Maximum number of blocks per batch

    uint32_t maskType = SPARSE_MODE_NO_MASK; // Mask type
    float scaleValue = 1.0;          // Scaling factor (usually 1/sqrt(dk))
    int64_t *actualSeqLengths{nullptr};      // Actual Q sequence length array
    int64_t *actualSeqLengthsKV{nullptr};   // Actual KV sequence length array
};
```

#### FATilingData - Output Tiling Data Structure

```cpp
class FATilingData {
public:
    InputParamsRegbase inputParamsRegbase;      // Input parameters
    MultiCoreParamsRegbase multiCoreParamsRegbase; // Multi-core segmentation parameters
};
```

InputParamsRegbase - Input Parameters

| Field                    | Type     | Description                               |
| ------------------------ | -------- | ----------------------------------------- |
| batch                    | int64_t  | Batch size                                |
| qHeads                   | int64_t  | Number of Query heads                     |
| kvHeads                  | int64_t  | Number of Key/Value heads                 |
| groupSize                | int64_t  | Group size (qHeads / kvHeads)             |
| qSeqlen                  | int64_t  | Query sequence length                     |
| kvSeqlen                 | int64_t  | Key/Value sequence length                 |
| embed                    | int64_t  | Embedding dimension                       |
| scaleValue               | float    | Scaling factor                            |
| attenMaskCompressMode    | uint8_t  | Attention mask compression mode           |
| isActualSeqLengthsNull   | uint8_t  | Whether Q actual sequence length is null  |
| isActualSeqLengthsKVNull | uint8_t  | Whether KV actual sequence length is null |
| actualSeqLengthsSize     | uint32_t | Size of Q actual sequence length array    |
| actualSeqLengthsKVSize   | uint32_t | Size of KV actual sequence length array   |
| headNumRatio             | uint32_t | Head number ratio (qHeads / kvHeads)      |
| blockSize                | uint32_t | Block size                                |
| blockTableDim2           | uint32_t | Block table dimension                     |
| paBlockNumSum            | uint32_t | Total number of blocks                    |
| attenMaskQSeqlen         | uint32_t | Attention mask Q sequence length          |
| attenMaskKvSeqlen        | uint32_t | Attention mask KV sequence length         |

MultiCoreParamsRegbase - Multi-core Parameters

| Field                        | Type     | Description                                 |
| ---------------------------- | -------- | ------------------------------------------- |
| coreNum                      | int32_t  | Actual number of cores used                 |
| totalSize                    | int64_t  | Total computation amount (number of blocks) |
| qSeqlenOuterSize             | int64_t  | Number of outer blocks in Q sequence        |
| splitFactorSize              | int64_t  | totalSize / coreNum                         |
| splitFactorTailSize          | int64_t  | totalSize % coreNum                         |
| bnAxisStartIdx[MAX_CORE_NUM] | uint32_t | Batch-Head axis start index array           |
| sparseStartIdx[MAX_CORE_NUM] | int64_t  | qSeq start index array                      |

### Core Algorithm Flow

#### Main Function: GetFATilingParam

**Location**:`fai_tiling.h:283-333`

**Function Signature**:

```cpp
int32_t GetFATilingParam(const FAInfo &faInfo, uint32_t blockDim, FATilingData& faTilingData)
```

**Parameter Description:**:

- `faInfo`: Input parameters
- `blockDim`: Number of available AI Cores
- `faTilingData`: Output tiling data

**Algorithm Flow**:

```text
1. Fill input parameters (FillInputParams)
   ├─ Copy basic information (batch, qHeads, kvHeads, seqSize, etc.)
   ├─ Calculate groupSize = qHeads / kvHeads
   └─ Set mask type and scaling factor

2. Fill actual sequence lengths (FillActualSeqLengths)
   ├─ If actualSeqLengths is null, fill with qSeqlen
   ├─ If actualSeqLengthsKV is null, fill with kvSeqlen
   └─ Save to actualSeqLengths and actualSeqLengthsKV arrays

3. Calculate total number of Blocks
   ├─ Iterate over each batch
   ├─ Calculate preTokensLeftUp and nextTokensLeftUp for each batch
   ├─ Fix row invalid cases (FixParamWithRowInvalid)
   └─ Accumulate valid block count for each batch

4. Calculate target weight
   coreWeightTarget = (totalBlockNumsOneHead * qHeads) / blockDim

5. Perform multi-core segmentation (ComputeSplitNBSeq)
   ├─ Use greedy algorithm to split along Batch/HeadNum/QSeqLen axes
   ├─ Fill bnAxisStartIdx and sparseStartIdx arrays
   └─ Determine actual number of cores used

6. Fill output parameters
   ├─ qSeqlenOuterSize = ceil(qSeqlen / BLOCK_BASE_SIZE)
   ├─ coreNum = actual number of cores used
   ├─ totalSize = total computation amount
   ├─ splitFactorSize = ceil(totalSize / coreNum)
   └─ splitFactorTailSize = totalSize % splitFactorSize
```

#### ComputeSplitNBSeq - Greedy Multi-Core Segmentation

**Location**: `fai_tiling.h:183-235`

**Functionality**: Use greedy algorithm to split tasks along Batch/HeadNum/QSeqLen axes

**Parameters**:

- `batchSize`: Batch size
- `tilingElementArrayLen`: Tiling array length (MAX_CORE_NUM)
- `actualSeqLengths`: Q actual sequence length array
- `actualSeqLengthsKV`: KV actual sequence length array
- `sOuterSize`: Outer block size (128)
- `sInnerSize`: Inner block size (128)
- `coreWeightTarget`: Target computation amount per core
- `curCore`: Current core index (input/output)

**Algorithm Flow**:

```text
Initialization:
  bnAxisStartIdx[MAX_CORE_NUM] = 0
  qSeqAxisStartIdx[MAX_CORE_NUM] = 0
  curWeight = 0
  curCore = 0

Iterate over three axes:
  for batchIdx in [0, batchSize):
    for headNum in [0, qHeads):
      1. Calculate preTokensLeftUp and nextTokensLeftUp
      2. Fix row invalid cases
      3. Calculate outerBlockNums and innerBlockNums

      for sOuterIndex in [0, outerBlockNums):
        1. Calculate number of inner blocks for current row:
           sInnerBlockNums = GetSInnerBlockNums(...)

        2. Greedily determine whether to switch to new core:
           diff = coreWeightTarget * (curCore + 1) - curWeight
           if sInnerBlockNums - diff > diff:
              curCore += 1
              bnAxisStartIdx[curCore] = batchIdx * qHeads + headNum
              qSeqAxisStartIdx[curCore] = sOuterIndex

        3. Accumulate computation amount for current row:
           curWeight += sInnerBlockNums
```

**Greedy Strategy:**

- Check if target computation amount is exceeded when adding an outer block (sOuterBlock)
- If `sInnerBlockNums - diff > diff`, it means the computation amount of current core is close to target, switch to new core
- This strategy ensures that the computation amount of each core is as balanced as possible

---

## Main Process Logic

### Init Function

**Location**: `fai_kernel.h:82-161`

**Functionality**: Initialize Kernel parameters and memory allocation

```cpp
CATLASS_DEVICE void Init(FAIKernelParams const& params)
```

**Flow**:

```text
1. Get current Core index
   ├─ AIC: blockIdx = GetBlockIdx()
   └─ AIV: blockIdx = GetBlockIdx() >> 1

2. Get AIV core index
   subBlockIdx = GetSubBlockIdx()  // 0: AIV0, 1: AIV1

3. Read parameters from Tiling data
   ├─ Basic parameters (batch, qHeads, kvHeads, seqSize, etc.)
   ├─ Scale value
   ├─ Mask type
   ├─ Multi-core segmentation parameters
   └─ Paged Attention parameters

4. Initialize synchronization events
   ├─ MM2_RES_INTRA_EVENT[2] = {7, 8}
   ├─ MM1_RES_INTRA_EVENT[2] = {9, 10}
   └─ Set initial Flag

5. Allocate UB memory
   ├─ bmm1TensorList[2] (shared by AIC/AIV)
   ├─ bmm2TensorList[2] (shared by AIC/AIV)
   ├─ expUb[3] (AIV)
   ├─ maxUb[3] (AIV)
   └─ sumUb[3] (AIV)

6. Allocate L1 memory
   └─ mm2AL1TensorList[3] (shared by AIC/AIV)
```

### operator() Function

**Location**:`fai_kernel.h:163-457`

**Functionality**: Kernel main function

```cpp
CATLASS_DEVICE void operator()(FAIKernelParams const &params)
```

**Flow**:

```text
1. Initialization
   ├─ Init(params)
   ├─ Create BlockMmadQK instance
   ├─ Create BlockMmadPV instance
   ├─ Create EpilogueOnlineSoftmax instance
   └─ Create EpilogueRescaleO instance

2. Check if Core is valid
   if (blockIdx >= coreNum) return;

3. Create input/output Tensors
   ├─ Q: [batch * qSeqlen, kvHeads * groupSize * embed]
   ├─ K: [kvHeads * embed, batch * kvSeqlen]
   ├─ V: [batch * kvSeqlen, kvHeads * embed]
   ├─ Mask: [batch * qSeqlen, kvSeqlen]
   └─ O: [batch * qSeqlen, kvHeads * groupSize * embed]

4. Determine intra-core segmentation range
   ├─ bnAxisStartIdx = multiCoreParamsRegbase.bnAxisStartIdx[blockIdx]
   ├─ bnAxisEndIdx = multiCoreParamsRegbase.bnAxisStartIdx[blockIdx + 1]
   └─ qSeqAxisStartIdx = multiCoreParamsRegbase.sparseStartIdx[blockIdx]

5. Main loop: Triple loop
   for bnIdx in [bnAxisStartIdx, bnAxisEndIdx):
       ├─ Calculate batch and head indices
       ├─ for qSeqAxisIndex in [qSeqAxisStartIdx, qSeqAxisEnd):
       │   ├─ Calculate Q sequence coordinates
       │   └─ for kvSeqLoopCount in [kvSeqLoopStartIdx, kvSeqLoopLimit]:
       │       ├─ Step 1: AIC executes Q*K^T
       │       ├─ Step 2: AIV executes Softmax
       │       ├─ Step 3: AIC executes P*V
       │       └─ Step 4: AIV executes O update
       └─ qSeqAxisStartIdx = 0
```

### Detailed Explanation of Triple Loop

#### First Layer Loop: Batch-Head Axis

**Location**:`fai_kernel.h:249-454`

```cpp
for (uint32_t bnIdx = bnAxisStartIdx; bnIdx < bnAxisEndIdx; ++bnIdx) {
    bool lastBN = (bnIdx == bnAxisEndIdx - 1);

    // Calculate batch and head indices
    runParam.batchOuterIdx = bnIdx / (kvHeads * headNumRatio);
    runParam.kvHeadsOuterIdx = (bnIdx / headNumRatio) % kvHeads;
    runParam.groupIdx = bnIdx % headNumRatio;

    // Calculate actual sequence lengths
    ComputeParamBatch(runParam, constInfo, attenMaskInfo);

    // Calculate Q sequence loop information
    ComputeQseqLoopInfo<qSeqlenTemplateType>(runParam, constInfo, lastBN, nextQSeqAxisIdx);

    // ...
}
```

**Explanation**:

- Iterate over Batch-Head axis according to Tiling segmentation results
- Support GQA (Grouped Query Attention): Multiple Query heads share KV heads

#### Second Layer Loop: Q Sequence Axis

**Location**:`fai_kernel.h:256-453`

```cpp
int64_t tempQSeqAxisEnd = lastBN ? (runParam.qSeqLoopTimes + 3) : runParam.qSeqLoopTimes;
for (int64_t qSeqAxisIndex = qSeqAxisStartIdx; qSeqAxisIndex < tempQSeqAxisEnd; ++qSeqAxisIndex) {
    // Handle last 3 special loops (for pipelined execution of tail blocks)
    bool notLastThreeLoop = true;
    bool notLastTwoLoop = true;
    if (lastBN) {
        int32_t extraQSeqAxis = qSeqAxisIndex - runParam.qSeqLoopTimes;
        switch (extraQSeqAxis) {
            case -1: isLastBmm1 = true; break;
            case 0: notLastThreeLoop = false; break;
            case 1: notLastThreeLoop = false; notLastTwoLoop = false; break;
            case 2: notLast = false; notLastThreeLoop = false; notLastTwoLoop = false; break;
        }
    }

    if (notLastThreeLoop) {
        // Calculate Q sequence parameters
        runParam.qSeqOuterAxisIdx = qSeqAxisIndex % qSeqlenOuterSize;
        ComputeParamQSeq<qSeqlenTemplateType>(runParam, constInfo, qSeqAxisIndex);

        // Calculate KV sequence loop information
        ComputeKvSeqLoopInfo<kvSeqlenTemplateType>(runParam, constInfo);
    }

    // ...
}
```

**Explanation**:

- Q sequence is segmented by 128 (BLOCK_BASE_SIZE)
- Last 3 loops are used for pipelined execution of tail blocks (ensure all tasks are completed)

#### Third Layer Loop: KV Sequence Axis

**Location**:`fai_kernel.h:291-452`

```cpp
for (int64_t kvSeqLoopCount = runParam.kvSeqLoopStartIdx; kvSeqLoopCount <= kvSeqLoopLimit; ++kvSeqLoopCount) {
    // Step 1: AIC executes Q*K^T
    if (notLastThreeLoop) {
        SetRunInfo(runInfo[taskId & 3], runParam, taskId, kvSeqLoopCount, kvSeqLoopLimit, multiCoreInnerIdx);
        if ASCEND_IS_AIC {
            // Prepare input Tensors
            auto tensorInQ = GetTile(tensorQWithLayout, ...);
            auto tensorInK = GetTile(tensorKWithLayout, ...);
            auto tensorMM1OWithLayout = MakeTensor(bmm1TensorList[taskIdMod2], ...);

            // Execute Q*K^T
            blockMmadMmadQK(tensorInQ, tensorInK, tensorMM1OWithLayout, ...);

            // Set synchronization Flag
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_V1_FLAG[taskIdMod2]);
        }
    }

    // Step 2: AIV executes Softmax
    if (taskId > 0 && notLastTwoLoop) {
        if ASCEND_IS_AIV {
            // Wait for BMM1 completion
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_V1_FLAG[taskIdMod2]);

            // Prepare input Tensors
            auto bmm1Tensor = MakeTensor(bmm1TensorList[taskIdMod2], ...);
            auto l1Vf1OutTile = GetTile(mm2AL1TensorList[taskIdMod3], ...);
            auto gmMaskTile = GetTile(tensorMaskWithLayout, ...);

            // Execute online Softmax
            epilogueOnlineSoftmax(l1Vf1OutTile, sumUb[multiCoreIdxMod3], maxUb[multiCoreIdxMod3],
                                expUb[taskIdMod3], bmm1Tensor, gmMaskTile, ...);
        }
    }

    // Step 3: AIC executes P*V
    if (taskId > 1 && notLast) {
        if ASCEND_IS_AIC {
            // Wait for Softmax completion
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V1_C2_FLAG[taskIdMod3]);

            // Prepare input Tensors
            auto mm2AL1Tensor = MakeTensor(mm2AL1TensorList[taskIdMod3], ...);
            auto tensorInV = GetTile(tensorVWithLayout, ...);
            auto mm2OutTensor = MakeTensor(bmm2TensorList[taskIdMod2], ...);

            // Execute P*V
            blockMmadMmadPV(mm2AL1Tensor, tensorInV, mm2OutTensor, ...);

            // Set synchronization Flag
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_V2_FLAG[taskIdMod2]);
        }
    }

    // Step 4: AIV executes O update
    if (taskId > 2) {
        if ASCEND_IS_AIV {
            // Wait for BMM2 completion
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_V2_FLAG[taskIdMod2]);

            // Prepare input Tensors
            auto bmm2Tensor = MakeTensor(bmm2TensorList[taskIdMod2], ...);
            auto attenOutGmTile = GetTile(attentionOutGmWithLayout, ...);

            // Execute O update and normalization
            epilogueRescaleO(attenOutGmTile, expUb[taskIdMod3], sumUb[multiCoreIdxMod3],
                            bmm2Tensor, isFirstLoop, isLastUpdate, ...);
        }
    }

    ++taskId;
}
```

**Explanation**:

- Four-stage pipeline: QK^T → Softmax → PV → O update
- Use taskId to implement CV pipeline

---

## Pipeline Detailed Explanation

### Pipeline Timing Diagram

<img src="../../docs/assets/images/flash_attention_infer_cv_pipeline.png" width="50%">

### Synchronization Mechanism

#### Event Definitions

```cpp
// AIC → AIV synchronization events
constexpr uint64_t SYNC_C1_V1_FLAG[2] = {0, 1};   // BMM1 completion
constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4}; // Softmax completion
constexpr uint64_t SYNC_C2_V2_FLAG[2] = {5, 6};   // BMM2 completion

// AIC internal events
constexpr uint64_t MM1_RES_INTRA_EVENT[2] = {9, 10}; // BMM1 internal synchronization
constexpr uint64_t MM2_RES_INTRA_EVENT[2] = {7, 8};  // BMM2 internal synchronization
```

#### Synchronization Flow

```text
Step 1: AIC executes Q*K^T
  ├─ Wait for L0C availability
  ├─ Execute Cube computation
  ├─ Wait for AIV processing completion (reverse synchronization)
  ├─ Copy results to UB
  └─ Set SYNC_C1_V1_FLAG to notify AIV

Step 2: AIV executes Softmax
  ├─ Wait for SYNC_C1_V1_FLAG (AIC notification)
  ├─ Execute Softmax computation
  ├─ Copy results to L1
  └─ Set SYNC_V1_C2_FLAG to notify AIC

Step 3: AIC executes P*V
  ├─ Wait for SYNC_V1_C2_FLAG (AIV notification)
  ├─ Read OnlineSoftmax intermediate results (L1)
  ├─ Execute Cube computation
  ├─ Wait for AIV processing completion (reverse synchronization)
  ├─ Copy results to UB
  └─ Set SYNC_C2_V2_FLAG to notify AIV

Step 4: AIV executes O update
  ├─ Wait for SYNC_C2_V2_FLAG (AIC notification)
  ├─ Read BMM2 results
  ├─ Execute O update and normalization
  └─ Write to GM
```

---

## BlockMmadQK: Q*K^T Matrix Multiplication

### Class Definition

**Location**:`block_mmad_fai_qk_tla.hpp:43-362`

```cpp
template <
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,    // Q
    class ElementB_,    // K
    class ElementC_,    // S
    class ElementBias_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadTla<MmadFAIQK<Arch::Ascend950, ...>, ...> {
    // ...
};
```

### Memory Layout

#### L1 Memory Layout

```text
+------------------------+  l1BufAddrStart
| L1A[Stage 0]         |  L1A_TILE_SIZE = 128 * 128 * sizeof(Dtype) = 32KB
+------------------------+
| L1B[Stage 0]         |  L1B_TILE_SIZE = 128 * 128 * sizeof(Dtype) = 32KB
+------------------------+
| L1A[Stage 1]         |  32KB
+------------------------+
| L1B[Stage 1]         |  32KB
+------------------------+
| ...                  |  ...
+------------------------+
```

#### L0 Memory Layout

```text
L0A Buffer:
+------------------------+
| L0A[Stage 0]         |  L0A_TILE_SIZE = 128 * 128 * sizeof(Dtype) = 32KB
+------------------------+
| L0A[Stage 1]         |  32KB
+------------------------+

L0B Buffer:
+------------------------+
| L0B[Stage 0]         |  L0B_TILE_SIZE = 128 * 128 * sizeof(Dtype) = 32KB
+------------------------+
| L0B[Stage 1]         |  32KB
+------------------------+

L0C Buffer:
+------------------------+
| L0C[Stage 0]         |  L0C_TILE_SIZE = 128 * 128 * sizeof(float) = 32KB
+------------------------+
| L0C[Stage 1]         |  32KB
+------------------------+
```

### Multi-Level Pipeline

#### Pipeline Stages

```cpp
Stage:  GM → L1  → L0  → Cube → L0C → UB
```

#### Stream Operation Flow

```cpp
void operator()(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, ...) {
    // 1. Wait for L0C availability
    WaitFlag<FIX_M>(l0CEventList_[l0CListId_]);

    // 2. Calculate number of loops in K direction
    int32_t kLoops = (blockK + L0_TILE_K - 1) / L0_TILE_K;

    for (int32_t kIdx = 0; kIdx < kLoops; ++kIdx) {
        // 3. Load Q from GM to L1A (only first time)
        if (isFirstLoop) {
            WaitFlag<MTE1_MTE2>(l1AEventList_[l1AListId_]);
            CopyInL1A(tensorL1A, tensorA, blockM, tileK, kIdx * L0_TILE_K);
            SetFlag<MTE2_MTE1>(l1AEventList_[l1AListId_]);
            WaitFlag<MTE2_MTE1>(l1AEventList_[l1AListId_]);
        }

        // 4. Load K from GM to L1B
        WaitFlag<MTE1_MTE2>(l1BEventList_[l1BListId_]);
        CopyInL1B(tensorL1B, tensorB, blockTable, tileK, blockN, kIdx * L0_TILE_K, blockSize);
        SetFlag<MTE2_MTE1>(l1BEventList_[l1BListId_]);
        WaitFlag<MTE2_MTE1>(l1BEventList_[l1BListId_]);

        // 5. Load Q from L1A to L0A
        WaitFlag<M_MTE1>(l0AEventList_[l0ListId_]);
        copyL1ToL0A(tensorL0A, tensorL1TileA);

        // 6. Load K from L1B to L0B
        WaitFlag<M_MTE1>(l0BEventList_[l0ListId_]);
        copyL1ToL0B(tensorL0B, tensorL1TileB);

        // 7. Execute Cube matrix multiplication
        SetFlag<MTE1_M>(l0CEventList_[l0CListId_]);
        WaitFlag<MTE1_M>(l0CEventList_[l0CListId_]);
        tileMmad(tensorL0C, tensorL0A, tensorL0B, tileM, tileN, tileK, initC);

        // 8. Update Stage index
        SetFlag<M_MTE1>(l0AEventList_[l0ListId_]);
        SetFlag<M_MTE1>(l0BEventList_[l0ListId_]);
        l0ListId_ = (l0ListId_ + 1 < STAGES) ? (l0ListId_ + 1) : 0;
    }

    // 9. Copy results to UB
    SetFlag<M_FIX>(l0CEventList_[l0CListId_]);
    WaitFlag<M_FIX>(l0CEventList_[l0CListId_]);

    // 10. Wait for AIV processing completion
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(MM1_RES_INTRA_EVENT[taskId]);

    // 11. Copy L0C to UB
    CopyL0CToDst copyL0CToDst;
    copyL0CToDst(tensorC, tensorL0C);

    // 12. Release L0C
    SetFlag<FIX_M>(l0CEventList_[l0CListId_]);
    l0CListId_ = (l0CListId_ + 1 < STAGES) ? (l0CListId_ + 1) : 0;
}
```

### Paged Attention Support

```cpp
void CopyInL1B(TensorL1B& tensorL1B, TensorB& tensorB, ...) {
    if constexpr(PAGED_CACHE_FLAG_) {
        // Paged loading: find physical block according to blockTable
        int32_t blockLoops = (tileN + blockSize - 1) / blockSize;
        for (int32_t blockIdx = 0; blockIdx < blockLoops; ++blockIdx) {
            int32_t curCopyCol = blockIdx == blockLoops - 1 ? tileN - copyColCnt : blockSize;
            int32_t idInBlockTable = blockTable.GetValue(blockIdx);  // Get physical block ID

            // Load data from physical block
            auto tensorTileB = GetTile(tensorB, MakeCoord(kOffset, idInBlockTable * blockSize), ...);
            copyGmToL1B(tensorL1TileB, tensorTileB);
            copyColCnt += curCopyCol;
        }
    } else {
        // Normal loading
        auto tensorTileB = GetTile(tensorB, MakeCoord(kOffset, 0), ...);
        copyGmToL1B(tensorL1TileB, tensorTileB);
    }
}
```

---

## EpilogueOnlineSoftmax: Online Softmax

### Class Definition

**Location**:`block_epilogue_fa_softmax_ascend950.hpp:32-424`

```cpp
template <
    class L1TileShape_,
    class PType_,      // P type
    class SType_,      // S type
    class MaskType_,   // Mask type
    bool ATTENTION_MASK_FLAG_
>
class BlockEpilogue<EpilogueAscend950FASoftmax<ATTENTION_MASK_FLAG_>, ...> {
    // ...
};
```

### Memory Layout

#### UB Memory Layout

```text
+------------------------+
| expSumUb              |  HALF_VEC_SIZE = 64 * sizeof(float) = 256B
+------------------------+
| nowMaxUb              |  HALF_VEC_SIZE = 256B
+------------------------+
| vf1OutUb[0]          |  HALF_SCM_BLOCK_SIZE = 64 * 128 * sizeof(Dtype) = 16KB
+------------------------+
| attenMaskUb[0]       |  HALF_MASK_BLOCK_SIZE = 64 * 128 * sizeof(uint8_t) = 8KB
+------------------------+
| vf1OutUb[1]          |  16KB
+------------------------+
| attenMaskUb[1]       |  8KB
+------------------------+
```

### Online Softmax Algorithm

#### Mathematical Principle

Standard Softmax:

```text
P[i,j] = exp(S[i,j] - max(S[i,:])) / sum(exp(S[i,:] - max(S[i,:])))
```

Online Softmax (step-by-step calculation):

```text
Initialization:
  max[i] = -inf
  sum[i] = 0

For each KV block's k:
  1. Calculate current block S[i,k] = Q[i] * K[k]^T / scale
  2. Calculate max_k[i] = max(S[i,k]) for current basic block
  3. Update global max for a row: max[i] = max(max[i], max_k[i])
  4. Calculate exp values: exp_k[i] = exp(S[i,k] - max[i])
  5. Update global sum: sum[i] = sum[i] * exp(max_old - max_new) + sum(exp_k[i])
  6. Calculate P[i,k] = exp_k[i] / sum[i]
```

#### Implementation Flow

```cpp
void operator()(TensorDst &vf1OutL1, LocalTensor<ElementS>&sumUb, ...) {
    // 1. Prepare input/output Tensors
    auto vf1OutUb = MakeTensor(vf1OutUbList[taskIdMod2], ...);

    // 2. Load Mask (if enabled)
    if constexpr (ATTENTION_MASK_FLAG) {
        WaitFlag<V_MTE2>(taskIdMod2);
        CopyGm2UbMask copyGm2UbMask;
        copyGm2UbMask(attenMaskUb, attenMaskGm);
        SetFlag<MTE2_V>(taskIdMod2);
        WaitFlag<MTE2_V>(taskIdMod2);
    }

    // 3. Prepare max and sum pointers
    if (isUpdate) {
        lastMaxUbAddr = lastMaxUb.GetPhyAddr();
        nowMaxUbAddr = nowMaxUb.GetPhyAddr();
        expSumUbAddr = expSumUb.GetPhyAddr();
    } else {
        nowMaxUbAddr = lastMaxUb.GetPhyAddr();  // Use lastMax as nowMax for first time
        expSumUbAddr = sumUb.GetPhyAddr();
    }

    // 4. Calculate Mask and Scale
    //    S[i,j] = Q[i] * K[j]^T / scale
    //    If mask is present, set positions with mask 0 to -3e38
    ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRange, ATTENTION_MASK_FLAG>(
        inputAddr, maskUbAddr, maskUbUnrollAddr, nowMaxUbAddr, m, tailN, scaleValue);

    // 5. Update max
    if (isUpdate) {
        UpdateMax<ElementS>(nowMaxUbAddr, lastMaxUbAddr, tailM);
    }

    // 6. Calculate exp and sum
    //    exp_k[i] = exp(S[i,k] - max[i])
    //    sum[i] = sum[i] + sum(exp_k[i])
    WaitFlag<MTE3_V>(taskIdMod3);
    ComputeExpSubSum<ElementP, ElementS, S2_BASE_SIZE, NRange>(
        outputAddr, inputAddr, nowMaxUbAddr, expSumUbAddr, m, blockStride);

    // 7. Copy results to L1
    SetFlag<V_MTE3>(taskIdMod2);
    WaitFlag<V_MTE3>(taskIdMod2);
    CopyUb2L1P copyUb2L1P;
    copyUb2L1P(vf1OutL1, vf1OutUb);

    // 8. Notify AIC
    SetFlag<MTE3_V>(taskIdMod3);
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V1_C2_FLAG);

    // 9. Update exp sum and exp max (for O update)
    if (isUpdate) {
        UpdateExpSumAndExpMax<ElementS>(
            sumUbAddr, expMaxUbAddr, lastMaxUbAddr, expSumUbAddr, nowMaxUbAddr, tailM);
    }
}
```

---

## BlockMmadPV: P*V Matrix Multiplication

### **Class Definition**

**Location**:`block_mmad_fai_pv_tla.hpp:43-317`

```cpp
template <
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,    // P
    class ElementB_,    // V
    class ElementC_,    // O_tmp
    class ElementBias_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadTla<MmadFAIPV<Arch::Ascend950, ...>, ...> {
    // ...
};
```

### Differences from BlockMmadQK

| Feature  | BlockMmadQK                     | BlockMmadPV                     |
| -------- | ------------------------------- | ------------------------------- |
| Input A  | Q (GM)                          | P (L1)                          |
| Input B  | K (GM)                          | V (GM)                          |
| Output C | S (UB)                          | O_tmp (UB)                      |
| L1A      | Loaded from GM                  | Input from L1                   |
| L1B      | Loaded from GM                  | Loaded from GM                  |
| K Loop   | Segmented along embed dimension | None                            |
| N Loop   | None                            | Segmented along embed dimension |

### Stream Operation Flow

```cpp
void operator()(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, ...) {
    // 1. Load V from GM to L1B
    WaitFlag<MTE1_MTE2>(l1BEventList_[l1BListId_]);
    CopyInL1B(tensorL1B, tensorB, blockTable, blockK, blockN, blockSize);
    SetFlag<MTE2_MTE1>(l1BEventList_[l1BListId_]);
    WaitFlag<MTE2_MTE1>(l1BEventList_[l1BListId_]);

    // 2. Wait for L0C availability
    WaitFlag<FIX_M>(l0CEventList_[l0CListId_]);

    // 3. Calculate number of loops in N direction (along embed dimension)
    uint32_t nLoops = (blockN + L0_TILE_N - 1) / L0_TILE_N;
    uint32_t tailSize = blockN % L0_TILE_N;
    uint32_t tailN = tailSize ? tailSize : L0_TILE_N;

    // 4. Load P from L1 to L0A (only first time)
    WaitFlag<M_MTE1>(l0AEventList_[l0AListId_]);
    for (uint32_t nIdx = 0; nIdx < nLoops; nIdx++) {
        uint32_t tileN = (nIdx == (nLoops - 1)) ? tailN : L0_TILE_N;

        // Load P from L1 to L0A (only first time)
        if (nIdx == 0) {
            auto tensorL1TileA = GetTile(tensorA, MakeCoord(0, 0), MakeShape(blockM, blockK));
            copyL1ToL0A(tensorL0A, tensorL1TileA);
        }

        // Load V from L1B to L0B
        WaitFlag<M_MTE1>(l0BEventList_[l0BListId_]);
        auto tensorL1TileB = GetTile(tensorL1B, MakeCoord(0, nIdx * L0_TILE_N), MakeShape(blockK, tileN));
        copyL1ToL0B(tensorL0B, tensorL1TileB);

        // Execute Cube matrix multiplication
        SetFlag<MTE1_M>(l0CEventList_[l0CListId_]);
        WaitFlag<MTE1_M>(l0CEventList_[l0CListId_]);
        auto tensorTileL0C = GetTile(tensorL0C, MakeCoord(0, nIdx * L0_TILE_N), MakeShape(tileM, tileN));
        tileMmad(tensorTileL0C, tensorL0A, tensorL0B, tileM, tileN, tileK, init=true);

        // Update Stage index
        SetFlag<M_MTE1>(l0BEventList_[l0BListId_]);
        l0BListId_ = (l0BListId_ + 1 < STAGES) ? (l0BListId_ + 1) : 0;
    }
    SetFlag<M_MTE1>(l0AEventList_[l0AListId_]);
    l0AListId_ = (l0AListId_ + 1 < STAGES) ? (l0AListId_ + 1) : 0;

    // 5. Copy results to UB
    SetFlag<M_FIX>(l0CEventList_[l0CListId_]);
    WaitFlag<M_FIX>(l0CEventList_[l0CListId_]);

    // 6. Wait for AIV processing completion
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(MM2_RES_INTRA_EVENT[taskId]);

    // 7. Copy L0C to UB
    CopyL0CToDst copyL0CToDst;
    copyL0CToDst(tensorC, tensorL0C);

    // 8. Release L0C
    SetFlag<FIX_M>(l0CEventList_[l0CListId_]);
    l0CListId_ = (l0CListId_ + 1 < STAGES) ? (l0CListId_ + 1) : 0;
}
```

---

## EpilogueRescaleO: O Update and Normalization

### **Class Definition**

**Location**:`block_epilogue_fa_rescale_o_ascend950.hpp:29-216`

```cpp
template <
    class L1TileShape_,
    class OType_,       // O type
    class OTmpType_    // O_tmp type
>
class BlockEpilogue<EpilogueAscend950FARescaleO, ...> {
    // ...
};
```

### Memory Layout

```text
+------------------------+
| vf2OutUb              |  VEC2_UB_SIZE = 64 * 128 * sizeof(float) = 32KB
+------------------------+
```

### O Update Algorithm

#### Mathematical Principle

Standard Flash Attention:

```text
O = softmax(Q * K^T / sqrt(d)) * V
```

Online calculation:

```text
Initialization:
  O[i] = 0
  max[i] = -inf
  sum[i] = 0

For each KV block k:
  1. Calculate S[i,k] = Q[i] * K[k]^T / scale
  2. Calculate max_k[i] = max(S[i,k])
  3. Update max: max[i] = max(max[i], max_k[i])
  4. Calculate exp_k[i] = exp(S[i,k] - max[i])
  5. Update sum: sum[i] = sum[i] * exp(max_old - max_new) + sum(exp_k[i])
  6. Calculate O_tmp[i] = exp_k[i] * V[k] # Calculated by BlockMmadPV
  # Steps 7 and 8 are EpilogueRescaleO flow, where exp(max_old - max_new) is calculated in EpilogueOnlineSoftmax
  7. Update O: O[i] = O[i] * exp(max_old - max_new) + O_tmp[i]
  8. Normalize: O[i] = O[i] / sum[i]
```

#### Implementation Flow

```cpp
void operator()(TensorDst &attenOutGm, const LocalTensor<ElementOTmp> &expMaxUb,
               const LocalTensor<ElementOTmp> &sumUb, TensorSrc &bmm2Res,
               bool isFirstLoop, bool isLastUpdate, uint64_t MM2_RES_INTRA_EVENT) {
    // 1. Wait for MTE3 completion
    WaitFlag<MTE3_V>(eventOMTE3V);

    // 2. Prepare buffer pointers
    __ubuf__ float *vec2ResUbAddr = vf2OutUb.GetPhyAddr();
    __ubuf__ float *bmm2UbAddr = bmm2Res.data().GetPhyAddr();
    __ubuf__ float *expMaxUbAddr = expMaxUb.GetPhyAddr();
    __ubuf__ float *sumUbAddr = sumUb.GetPhyAddr();

   // 3. Update O
    if (isFirstLoop) {
        // First time: direct copy
        DataCopy(vf2OutUb, bmm2Res.data(), m * n);
    } else if (!isLastUpdate) {
        // Intermediate step: O = O * exp(max_old - max_new) + O_tmp
        FlashUpdateNew<ElementOTmp, D_BASE_SIZE>(
            vec2ResUbAddr, bmm2UbAddr, expMaxUbAddr, m, nLoops, tailN);
    } else {
        // Last time: O = (O * exp(max_old - max_new) + O_tmp) / sum
        FlashUpdateLastNew<ElementOTmp, D_BASE_SIZE>(
            vec2ResUbAddr, bmm2UbAddr, expMaxUbAddr, sumUbAddr, m, nLoops, tailN);
    }

    // 4. Notify AIC
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_RES_INTRA_EVENT);

    // 5. For special scenario with only one round, update if for the first and last time
    if (isFirstLoop && isLastUpdate) {
        // Only one KV block: O = O / sum
        LastDivNew<ElementOTmp, D_BASE_SIZE>(
            vec2ResUbAddr, bmm2UbAddr, sumUbAddr, m, nLoops, tailN);
    }

    if (isLastUpdate) {
        // 6. Type conversion (FP32 → FP16/BF16)
        LocalTensor<ElementO> attenOut;
        attenOut.SetAddr(vf2OutUb.address_);
        Cast(attenOut, vf2OutUb, RoundMode::CAST_ROUND, m * D_BASE_SIZE);

        // 7. Copy to GM
        SetFlag<V_MTE3>(eventOVMTE3);
        WaitFlag<V_MTE3>(eventOVMTE3);
        CopyUbToGmO copyUbToGmO;
        copyUbToGmO(attenOutGm, attenOutUb);
    }

    // 8. Release resources
    SetFlag<MTE3_V>(eventOMTE3V);
}
```

---

## Next Step Optimization Suggestions

1. Currently, only BlockMmadQK L0c output-> UB -> EpilogueSoftMax is implemented. Due to UB space limitations, the current template only supports embed <= 128. To support larger embedSize, it is necessary to extend the L0c output-> GM -> UB -> EpilogueSoftMax flow.
2. The current template Kernel does not support the variable-length ActualSeq feature and needs to be adapted.
