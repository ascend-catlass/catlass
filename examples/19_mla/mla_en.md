# CATLASS MLA

CATLASS MLA is a Flash-MLA operator optimized for Ascend Atlas A2 hardware, implemented based on the CATLASS GEMM API. The operator architecture consists of the following hierarchical layers:

- Tiling parameter calculation
- Kernel implementations, featuring two variants: the generic [mla_kernel.cpp](./mla_kernel.cpp) and the specialized [mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp)
- Block-level components within the Kernel that are optimized for Flash-MLA execution
- Tile-level components provided by the template library that underpin the Block-level components.

## Tiling

The tiling calculation logic is located in the [mla.cpp](./mla.cpp) file. Before invoking the operator, all required parameters for tiling must be prepared, assigned to the MLAInfo structure, and passed to the `GetMLATilingParam` function. An example implementation is provided in [mla.cpp](./mla.cpp):

```c++
// Prepare the intermediate structure and host-side memory required for Tiling calculation
MLATiling::MLAInfo mlaInfo;
...
MLATiling::GetMLATilingParam(mlaInfo, blockDim, (uint32_t *)tilingHost);
```

Within the `GetMLATilingParam` function, two internal functions are called: `GetMLATilingCommon` and `GetMLATilingSpec`. These functions handle the core partitioning logic for generic and specialized execution scenarios, respectively.

## Kernel

This operator provides two distinct Kernel implementations:

1. Generic variant ([mla_kernel.cpp](./mla_kernel.cpp)): Delivers optimal performance when `qHeadNum` is set to 16, 32, or 64 (corresponding to TP8, TP4, and TP2 tensor parallel models on the network side).
2. Specialized variant ([mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp)): Delivers optimal performance when `qHeadNum` is set to 128 (corresponding to TP1 tensor parallel models on the network side).

[mla_kernel.cpp](./mla_kernel.cpp) has the following features:

- Implements the classic four-stage Flash Attention execution pipeline, tiling the input Q, QRope, K, and KRope tensors before computation.
- Tiles the input sequence length `kvSeqlen` using `blockSize` as the fundamental unit. Each Attention calculation cycle operates on a single block as its base unit. It enables pre-issuing the QK MMAD and softmax operations for a base block, allowing the CUBE and VECTOR execution stages of adjacent base blocks to overlap and hide latency.
- For the QK and PV matrix multiplications within the same base unit, since K and V share the same data segment, K is kept resident in the L1 buffer. This optimization significantly reduces inbound data transfer bandwidth.

[mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp) has the following features:

- Implements the classic four-stage Flash Attention execution pipeline, tiling the input Q, QRope, K, and KRope tensors before computation.
- Tiles the input sequence length `kvSeqlen` using `blockSize` as the fundamental unit. Each Attention calculation cycle operates on four blocks as its base unit. It enables pre-issuing the QK MMAD and softmax operations for a base block, allowing the CUBE and VECTOR execution stages of adjacent base blocks to overlap and hide latency.
- By scaling up the base unit size, this kernel substantially decreases the outbound data volume during the PV MMAD stage, lowering outbound bandwidth consumption. Conversely, due to hardware buffer capacity limitations, K residency in L1 is disabled in this mode.

The operator leverages Block and Tile-level components to construct the final Kernel through the following pipeline:

1. Instantiates two computational `BlockMmad` modules (QK and PV) along with three `BlockEpilogue` modules (softmax, rescaleO, and flashDecoding).
2. Aggregates these components into a unified `MLAKernel` class, which orchestrates loop scheduling across the individual Blocks.

This composition hierarchy is reflected in the Kernel entry-point definition (using [mla_kernel.cpp](./mla_kernel.cpp) as a reference):

```c++
// GEMM Block module, implementing Q*K^T for Flash MLA
using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
using QType = Gemm::GemmType<ElementQ, LayoutQ>;
using KType = Gemm::GemmType<ElementK, LayoutK>;
using SType = Gemm::GemmType<ElementS, LayoutS>;
using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

// Epilogue Block module, implementing online softmax for the current S base block in Flash MLA
using PType = Gemm::GemmType<ElementP, LayoutP>;
using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
using EpilogueMLASoftmax =
    Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;

// GEMM Block module, implementing P*V for Flash MLA
using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
using VType = Gemm::GemmType<ElementV, LayoutV>;
using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

// Epilogue Block module, implementing output updates for the current O base block in Flash MLA
using OType = Gemm::GemmType<ElementO, LayoutO>;
using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;

// Epilogue Block module, implementing flash decoding logic for Flash MLA
using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
constexpr uint32_t ComputeEleNum = 6144;
using EpilogueMLAFDRescaleO =
    Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

// Kernel level
using MLAKernel = MLAKernel<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax,
                            EpilogueMLARescaleO, EpilogueMLAFDRescaleO>;
```

## Block Mmad

The operator uses two variations of the `BlockMmad` component:

- `BlockMmadQK`: A template specialization designed to handle the $Q*K^T$ matrix multiplication within the Flash-MLA pipeline. The implementation in [block_mmad_mla_qk.hpp](../../include/catlass/gemm/block/block_mmad_mla_qk.hpp) targets the generic [mla_kernel.cpp](./mla_kernel.cpp), while [block_mmad_mla_qk_tp1_spec.hpp](../../include/catlass/gemm/block/block_mmad_mla_qk_tp1_spec.hpp) is tailored for the specialized [mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp).
- `BlockMmadPV`: A template specialization designed to execute the $P*V$ matrix multiplication. The logic in [block_mmad_mla_pv.hpp](../../include/catlass/gemm/block/block_mmad_mla_pv.hpp) maps to the generic [mla_kernel.cpp](./mla_kernel.cpp), whereas [block_mmad_mla_pv_tp1_spec.hpp](../../include/catlass/gemm/block/block_mmad_mla_pv_tp1_spec.hpp) corresponds to the specialized [mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp).

## Block Epilogue

The operator uses three variations of the BlockEpilogue component:

- `EpilogueMLASoftmax`: A template specialization optimized for the online softmax pipeline. Its generic implementation resides in [block_epilogue_mla_softmax.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_softmax.hpp) for [mla_kernel.cpp](./mla_kernel.cpp), and its specialized version is located in [block_epilogue_mla_tp1_softmax.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_tp1_softmax.hpp) for [mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp).
- `EpilogueMLARescaleO`: A template specialization managing the `rescaleO` operation. The implementation file [block_epilogue_mla_rescale_o.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_rescale_o.hpp) pairs with the generic [mla_kernel.cpp](./mla_kernel.cpp), while [block_epilogue_mla_tp1_rescale_o.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_tp1_rescale_o.hpp) links to the specialized [mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp).
- `EpilogueMLAFDRescaleO`: A template specialization covering `flashDecoding` updates when required. The logic defined in [block_epilogue_mla_fd_rescale_o.hpp](../../include/catlass/epilogue/block/block_epilogue_mla_fd_rescale_o.hpp) is shared commonly between both [mla_kernel.cpp](./mla_kernel.cpp) and [mla_kernel_tp1_spec.cpp](./mla_kernel_tp1_spec.cpp).

## Tile Mmad & Tile Copy

The Block-level infrastructure of the generic Kernel depends on tile components located in [tile_mmad.hpp](../../include/catlass/gemm/tile/tile_mmad.hpp) (`tileMmad`) and [tile_copy.hpp](../../include/catlass/gemm/tile/tile_copy.hpp) (`tileCopy`), illustrated as follows:

```c++
using TileMmad = TileMmad_;
using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
using ElementAccumulator =
    typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
```
