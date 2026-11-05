# Copy L1 To BT Overview

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_bt.hpp)

[TOC]

## Overview

The `copy_l1_to_bt` module provides a template class for moving the Bias Table (1D vector) from the L1 (Local Memory, referred to as A1 Buffer) to the BT (Bias Table Buffer, referred to as C2 Buffer). The bias table is used for bias addition and quantization/dequantization operations in matrix multiplication.

Since bias data is a 1D vector, this module exclusively uses `VectorLayout` (rank=1, stride=1) as the fixed data layout. The implementation is split into two architecture-specific sets:

- **Atlas A2** (ARCH 2201): [atlasa2/copy_l1_to_bt.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_bt.hpp)
- **Ascend 950** (ARCH 3510): [ascend950/copy_l1_to_bt.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_bt.hpp)

The module provides two sets of APIs: a **non-TLA style** (directly operating on `LocalTensor`) and a **TLA style** (using `tla::Tensor` wrapper).

## API List

| Component| Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyL1ToBT](./copy_l1_to_bt.md) | Non-TLA| Atlas A2/Ascend 950| Basic L1 → BT 1D vector movement using the DataCopy instruction|
| [TileCopyTla](./tile_copy_tla.md) | TLA | Ascend 950| TLA-style L1 → BT movement, which uses tla::Tensor wrapper|

> **Note**: This module is not intended to be used directly in most cases. Instead, it is used as a member type of `CopyL1ToBT` in [TileCopy](../tile_copy/README.md) and is automatically managed by [blockMmad](../../block/block_mmad.md). Explicit declaration is only required during implementation of custom kernel templates that require manual assembly.

## Applicable Hardware Models

| Hardware Model| Architecture ID| ARCH Macro| Supported Non-TLA Template| Supported TLA Template|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2 | `Arch::AtlasA2` | `CATLASS_ARCH == 2201` | CopyL1ToBT | — |
| Ascend 950 | `Arch::Ascend950` | `CATLASS_ARCH == 3510` | CopyL1ToBT | TileCopyTla |

### Architecture Differences

| Feature| Atlas A2| Ascend 950|
| :------ | :------ | :------ |
| Destination buffer| C2 (Bias Table)| C2 (Bias Table)|
| Movement instruction| `AscendC::DataCopy` | `AscendC::DataCopy` |
| blockLen alignment reference| `BYTE_PER_C2` | `BYTE_PER_C0` |
| B32 alignment processing| No processing| `RoundUp(blockLen, 2)` |
| Source/destination element type| Can be different| Can be different|
| TLA| Not supported| Supported|

## API Calling Examples

### Non-TLA Style (CopyL1ToBT)

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>;

uint32_t vecLen = 256;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;

using CopyOp = CopyL1ToBT<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstBTTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### TLA Style (TileCopyTla, Ascend 950)

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;

const uint32_t vecLen = 256;

auto layoutSrc = tla::MakeLayout<ElementSrc, layout::VectorLayout>(1, vecLen);
auto layoutDst = tla::MakeLayout<ElementDst, layout::VectorLayout>(1, vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstBTTensor, layoutDst, Arch::PositionBias{});

TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

## Template Selection Guide

| Scenario| Recommended Template|
| :------ | :------ |
| General Bias Table L1→BT movement| `CopyL1ToBT` (non-TLA)|
| Atlas A2 Bias Table movement| <idp:inline displayname="code" id="code166567325374">CopyL1ToBT</idp:inline> (non-TLA)|
| Ascend 950 Bias Table movement| `CopyL1ToBT` (non-TLA) or `TileCopyTla` (TLA)|
| TLA programming paradigm used| `TileCopyTla` (unified style, Ascend 950 only)|
