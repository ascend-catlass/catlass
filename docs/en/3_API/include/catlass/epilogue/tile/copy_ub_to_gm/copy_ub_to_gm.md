# CopyUb2Gm

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm.hpp)

[TOC]

## Function

`CopyUb2Gm` implements data movement from Unified Buffer to global memory in the epilogue stage. It is used to write back the final results, after epilogue processing, from the Unified Buffer to global memory.

- Applicability: Atlas A2 series products and Atlas A3 series products and Ascend 950PR&950DT series products
- Style: non-TLA, directly operating on `AscendC::GlobalTensor`/`AscendC::LocalTensor`
- Data movement with stride using `AscendC::DataCopyPad`

## Template Prototype

```cpp
template <
    class ArchTag,
    class GmType      // Gemm::GemmType<Element, Layout>
>
struct CopyUb2Gm;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag, which can be `Arch::AtlasA2` or `Arch::Ascend950`|
| `GmType` | GM data type, `Gemm::GemmType<Element, Layout>`. The layout determines the global memory output format.|

## Partial Specialization Implementation

| Architecture| Global Memory Layout| Unified Buffer Layout| Description|
| :------ | :------ | :------ | :------ |
| Atlas A2| `RowMajor` | `RowMajor` | Two-dimensional matrix movement, with source stride aligned to C0|
| Atlas A2| `VectorLayout` | `VectorLayout` | One-dimensional vector movement|
| Ascend 950| `RowMajor` | `RowMajor` | Two-dimensional matrix movement|

## APIs

```cpp
void operator()(
    AscendC::GlobalTensor<Element> const &dstTensor,    // Destination global memory GlobalTensor
    AscendC::LocalTensor<Element> const &srcTensor,     // Source Unified Buffer LocalTensor
    LayoutDst const &layoutDst,                         // Destination global memory layout description
    LayoutSrc const &layoutSrc                          // Source Unified Buffer layout description
)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination global memory GlobalTensor|
| `srcTensor` | Source Unified Buffer LocalTensor|
| `layoutDst` | Layout of the destination global memory, including the shape and stride|
| `layoutSrc` | Layout of the source Unified Buffer, including the shape and stride|

## Examples

### RowMajor (Two-Dimensional Matrix)

```cpp
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagDst = layout::RowMajor;

uint32_t rows = 128;
uint32_t cols = 256;

auto layoutDst = LayoutTagDst::MakeLayout<Element>(rows, cols);
auto layoutSrc = LayoutTagDst::MakeLayout<Element>(rows, cols);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagDst>;
using CopyOp = CopyUb2Gm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### VectorLayout (One-Dimensional Vector)

```cpp
using Element = half;
using LayoutTagDst = layout::VectorLayout;

uint32_t length = 256;

auto layoutDst = LayoutTagDst::MakeLayout<Element>(length, 1);
auto layoutSrc = LayoutTagDst::MakeLayout<Element>(length, 1);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagDst>;
using CopyOp = CopyUb2Gm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
