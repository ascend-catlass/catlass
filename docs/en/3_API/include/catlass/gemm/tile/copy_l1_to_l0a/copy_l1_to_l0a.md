# CopyL1ToL0A

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_l0a.hpp)

[TOC]

## Function

`CopyL1ToL0A` is a template responsible for moving tile blocks of matrix A from L1 (Local Memory, also referred to as the A1 Buffer) to L0A (the A2 Buffer). It supports conversion between multiple data layouts.

Depending on the source and destination layouts, the template selects an appropriate hardware data movement instruction.

- **zN → zZ**: Nd transposed copy (`ifTranspose = false`)
- **nZ → zZ**: transposed copy (Transpose A) with `ifTranspose = true`. For int8_t data, the `LoadDataWithTranspose` instruction is used.
- **NDC1HWC0 → zZ**: convolution-specific path using `LoadData3Dv2`

This template is not intended to be used directly in most cases. Instead, it is used as a member type of [TileCopy](../tile_copy/README.md) and is automatically managed by `blockMmad`. Explicit declaration is only required during implementation of custom kernels that require manual assembly.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag: Arch::AtlasA2 or Arch::Ascend950
    class L1Type,                     // L1 data description: Gemm::GemmType<Element, Layout, AscendC::TPosition::A1>
    class L0Type = void               // L0A data description: Gemm::GemmType<Element, Layout, AscendC::TPosition::A2> (optional, automatically deduced)
>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};
```

- `ArchTag`: architecture tag, which can be `Arch::AtlasA2` or `Arch::Ascend950`
- `L1Type`: data type of matrix A in L1, which encapsulates Element, Layout, and TPosition.
- `L0Type`: data type of matrix A in L0A. The default value is `void`. Most partial specializations will automatically deduce it.

## Partial Specialization Implementation

### Atlas A2

| Source Layout| Destination Layout| Element Type| Description|
| :------ | :------ | :------ | :------ |
| zN | zZ | Any| Basic Nd copy using LoadData2D|
| zN | zZ | float | float-specific path using LoadData3D|
| nZ | zZ | Any (non-int8_t)| Transposed copy using LoadData2D (ifTranspose=true)|
| nZ | zZ | int8_t | Transposed copy using LoadDataWithTranspose|
| nZ | zZ | float | Transposed copy using LoadData3D (with SetFmatrix alignment)|
| nN | zZ | Any| nN-to-zZ copy using LoadData2D|
| nN | zZ | float | float-specific path using LoadData2dTranspose|
| NDC1HWC0 | zZ | Any| Convolution-specific path using LoadData3Dv2|

### Ascend 950

| Source Layout| Destination Layout| Element Type| Description|
| :------ | :------ | :------ | :------ |
| zN | zN | Any| Basic Nd copy using LoadData2DParamsV2|
| nZ | zN | Non-B8/B4 (int8_t/float8_e4m3_t/float8_e5m2_t/float4, etc.)| Transposed copy using LoadData2DParamsV2|
| nZ | zN | B8/B4 (int8_t/float8_e4m3_t/float8_e5m2_t/float4, etc.)| Transposed copy. Select single-step or step-by-step LoadData based on the L0M alignment.|

> **Note**: The destination layout of `L0Type` on Ascend 950 is zN (not zZ), which is different from that on Atlas A2.

## APIs

### Basic APIs (For All Partial Specializations)

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,   // L0A destination tensor.
    AscendC::LocalTensor<Element> srcTensor,   // L1 source tensor
    LayoutDst layoutDst,                       // L0A data layout
    LayoutSrc layoutSrc                        // L1 data layout
);
```

### Convolution APIs (NDC1HWC0 Partial Specialization)

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,
    AscendC::LocalTensor<Element> srcTensor,
    LayoutDst layoutDst, LayoutSrc layoutSrc,
    uint32_t kStartPt, uint32_t mStartPt,
    uint32_t l1H, uint32_t l1W, uint8_t* padList
);
```

This specialization constructs instances through a static factory method.

```cpp
static CopyL1ToL0A MakeCopyL1ToL0A(
    uint32_t strideW = 0, uint32_t strideH = 0,
    uint32_t filterW = 0, uint32_t filterH = 0,
    uint32_t dilationFilterW = 0, uint32_t dilationFilterH = 0
);
```

## Examples

### Basic zN → zZ Movement (Atlas A2, non-TLA)

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

### nZ → zZ Transposed Movement (Atlas A2, non-TLA)

```cpp
using L1Type = Gemm::GemmType<half, layout::nZ, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<half, layout::zZ, AscendC::TPosition::A2>;

auto layoutSrc = layout::nZ::MakeLayout<half>(row, col);
auto layoutDst = layout::zZ::MakeLayout<half>(row, col);

using CopyOp = CopyL1ToL0A<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0ATensor, srcL1Tensor, layoutDst, layoutSrc);
```

### zN → zN Movement (Ascend 950, non-TLA)

```cpp
using L1Type = Gemm::GemmType<half, layout::zN, AscendC::TPosition::A1>;

auto layoutSrc = layout::zN::MakeLayout<half>(row, col);
auto layoutDst = layout::zN::MakeLayout<half>(row, col);

// On Ascend 950, L0Type is optional, and the corresponding layout is automatically deduced.
using CopyOp = CopyL1ToL0A<Arch::Ascend950, L1Type>;
CopyOp copyOp;
copyOp(dstL0ATensor, srcL1Tensor, layoutDst, layoutSrc);
```
