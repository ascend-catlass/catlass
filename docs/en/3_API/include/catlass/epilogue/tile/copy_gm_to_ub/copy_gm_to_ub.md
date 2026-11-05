# CopyGm2Ub

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub.hpp)

[TOC]

## Function

`CopyGm2Ub` implements data movement from global memory to Unified Buffer in the epilogue stage. It is used to move the final output matrices C/X/Y from global memory to Unified Buffer, for subsequent epilogue computations.

- Applicability: Atlas A2 series products and Atlas A3 series products and Ascend 950PR&950DT series products
- Style: non-TLA, directly operating on `AscendC::LocalTensor`/`AscendC::GlobalTensor`
- Data movement with stride using `AscendC::DataCopyPad`

## Template Prototype

```cpp
template <
    class ArchTag,
    class GmType      // Gemm::GemmType<Element, Layout>
>
struct CopyGm2Ub;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag, which can be `Arch::AtlasA2` or `Arch::Ascend950`|
| `GmType` | GM data type, `Gemm::GemmType<Element, Layout>`. The layout triggers different partial specializations.|

## Partial Specialization Implementation

| Architecture| Global Memory Layout| Unified Buffer Layout| Description|
| :------ | :------ | :------ | :------ |
| Atlas A2| `RowMajor` | `RowMajor` | Two-dimensional matrix movement, `DataCopyPad`|
| Atlas A2| `VectorLayout` | `VectorLayout` | One-dimensional vector movement|
| Ascend 950| `RowMajor` | `RowMajor` | Two-dimensional matrix movement, `DataCopyPad`|
| Ascend 950| `VectorLayout` | `VectorLayout` | One-dimensional vector movement|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,     // Destination Unified Buffer LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,    // Source global memory GlobalTensor
    LayoutDst const &layoutDst,                         // Destination Unified Buffer layout description
    LayoutSrc const &layoutSrc                          // Source global memory layout description
)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination Unified Buffer LocalTensor|
| `srcTensor` | Source global memory GlobalTensor|
| `layoutDst` | Layout of the destination Unified Buffer, including the shape and stride|
| `layoutSrc` | Layout of the source global memory, including the shape and stride|

## Examples

### RowMajor (Two-Dimensional Matrix)

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagSrc = layout::RowMajor;
using GmType = Gemm::GemmType<Element, LayoutTagSrc>;

uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(rows, cols);
auto layoutDst = LayoutTagSrc::MakeLayout<Element>(rows, cols);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using CopyOp = CopyGm2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### VectorLayout (One-Dimensional Vector)

```cpp
using Element = half;
using LayoutTagSrc = layout::VectorLayout;

uint32_t length = 256;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(length, 1);
auto layoutDst = LayoutTagSrc::MakeLayout<Element>(length, 1);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;
using CopyOp = CopyGm2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
