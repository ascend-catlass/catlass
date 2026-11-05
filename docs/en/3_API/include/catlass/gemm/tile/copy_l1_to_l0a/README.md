# Copy L1 To L0A Overview

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_l0a.hpp)

[TOC]

## Overview

The `copy_l1_to_l0a` module provides template classes that move tile blocks of matrix A from L1 (Local Memory, referred to as A1 Buffer) to L0A (A2 Buffer), with support for conversion across multiple layouts. Based on the architectures, the implementation is split into two sets:

- **Atlas A2** (ARCH 2201): [atlasa2/copy_l1_to_l0a.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp)
- **Ascend 950** (ARCH 3510): [ascend950/copy_l1_to_l0a.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0a.hpp)

The module provides two sets of APIs: a **non-TLA style** (directly operating on `LocalTensor`) and a **TLA style** (using `tla::Tensor` wrapper).

## API List

| Component| Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyL1ToL0A](./copy_l1_to_l0a.md) | Non-TLA| Atlas A2/Ascend 950| Basic L1 → L0A movement template, with support for conversion across multiple layouts|
| [TileCopyTla](./tile_copy_tla.md) | TLA | Atlas A2/Ascend 950| TLA-style L1 → L0A movement, which simplifies calling via the tla::Tensor wrapper|
| [TileCopySparseTla](./tile_copy_sparse_tla.md) | TLA | Atlas A2| Sparse GEMM L1 → L0A movement, zN → zZ LoadData3D v2|

> **Note**: This module is not intended to be used directly in most cases. Instead, it is used as a member type of `CopyL1ToL0A` in [TileCopy](../tile_copy/README.md) and is automatically managed by [blockMmad](../../block/block_mmad.md). Explicit declaration is only required during implementation of custom kernel templates that require manual assembly.

## Applicable Hardware Models

| Hardware Model| Architecture ID| ARCH Macro| Supported Non-TLA Template| Supported TLA Template|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2 | `Arch::AtlasA2` | `CATLASS_ARCH == 2201` | CopyL1ToL0A | TileCopyTla |
| Ascend 950 | `Arch::Ascend950` | `CATLASS_ARCH == 3510` | CopyL1ToL0A | TileCopyTla |

### Architecture Differences

| Feature| Atlas A2| Ascend 950|
| :------ | :------ | :------ |
| Destination L0A layout| zZ | zN |
| Basic movement instruction| LoadData2D | LoadData2DParamsV2 |
| l0Batch batch movement| Not supported| Supported (via `operator()` overload)|
| MX Scale floating-point quantization| Not supported| Supported (via `operator()` overload)|
| Vector layout | Not supported| Supported|

## API Calling Examples

### Non-TLA Style (CopyL1ToL0A)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using L1Type = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;

uint32_t row = 256;
uint32_t col = 256;

// Construct the zN layout on L1 and the zZ layout on L0A.
auto layoutSrc = layout::zN::MakeLayout<Element>(row, col);
auto layoutDst = layout::zZ::MakeLayout<Element>(row, col);

AscendC::LocalTensor<Element> srcL1Tensor;
AscendC::LocalTensor<Element> dstL0ATensor;

// Instantiation and call
using CopyOp = CopyL1ToL0A<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0ATensor, srcL1Tensor, layoutDst, layoutSrc);
```

### TLA Style (TileCopyTla)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

// Create a layout using tla::MakeLayout.
auto layoutSrc = tla::MakeLayout<half, layout::zN>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zZ>(M, K);

// Construct a TLA tensor using tla::MakeTensor.
AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0ATensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// Instantiation and call (SFINAE automatically matches the partial specialization based on the source/destination layout trait.)
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Style — Transposed Movement (Atlas A2)

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zZ>(M, K);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// isnZ<LayoutSrc> && iszZ<LayoutDst> → Automatic matching for transposed partial specialization
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Style - Ascend 950 Basic Movement

```cpp
// The destination layout of Ascend 950 is zN.
auto layoutSrc = tla::MakeLayout<half, layout::zN>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zN>(M, K);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// Ascend 950: zN L1 → zN L0A
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Style - Ascend 950 l0Batch Batch Movement

```cpp
uint32_t l0Batch = 4;

// l0Batch overload: continuous movement of multiple batches
copyOp(dstTensor, srcTensor, l0Batch);
```

### TLA Style - Ascend 950 MX Scale Movement

```cpp
using ElementSrc = float8_e4m3_t;
using ElementDst = AscendC::mx_fp8_e4m3_t;
using ElementMxScale = float8_e8m0_t;

// MX Scale K-dimension: one scale value is shared by every MX_SCALE_GROUP_NUM (32) elements along K.
const uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(K);

// Source data layout (L1 zN)
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(M, K);
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});

// Destination data layout (L0A zN, element type: mx_fp8)
auto layoutDst = tla::MakeLayout<ElementDst, layout::zN>(M, K);
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// MX Scale layout (L1 zZ, constructed using MakeMxScaleLayout)
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::zZ, false>(M, mxScaleK);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

// MX Scale overload: L1 zN source data + L1 zZ scale → L0A zN mx data
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```

## Template Selection Guide

| Scenario| Recommended Template|
| :------ | :------ |
| GEMM tile-level L1 → L0A movement| `CopyL1ToL0A` (non-TLA) or `TileCopyTla` (TLA)|
| Transposed movement (nZ → zZ/nZ → zN)| `CopyL1ToL0A` or `TileCopyTla` (automatic matching)|
| Ascend 950 multi-batch movement| `TileCopyTla` (L0Batch overload)|
| Ascend 950 MX floating-point quantization| `TileCopyTla` (MX Scale overload)|
| Convolution-specific NDC1HWC0 movement| `CopyL1ToL0A` (non-TLA, NDC1HWC0 partial specialization)|
| TLA programming paradigm used| `TileCopyTla` (unified style)|
