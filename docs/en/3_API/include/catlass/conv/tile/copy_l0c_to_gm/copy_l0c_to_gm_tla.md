# CopyL0CToGmTla

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_l0c_to_gm.hpp)

[TOC]

## Function

`CopyL0CToGmTla` implements the TLA-style version of writing back accumulated results from L0C to global memory in convolution scenarios. It uses `AscendC::Fixpipe` as a direct path to complete movement, type conversion, and optional ReLU.

- Applicability: Atlas A2 series products and Atlas A3 series products and Ascend 950PR&950DT series products
- Style: TLA

## Template Prototype

```cpp
template <class ArchTag, class TensorSrc, class TensorDst,
          ScaleGranularity DEQUANT_GRANULARITY = NO_QUANT, bool ReluEnable = false>
struct CopyL0CToGmTla;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `TensorSrc` | TLA tensor type (L0C, zN)|
| `TensorDst` | TLA tensor type (GM, NC1HWC0)|
| `DEQUANT_GRANULARITY` | Quantization mode|
| `ReluEnable` | Whether to enable ReLU|

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor,
    uint8_t unitFlag = 0
)
```

## Examples

```cpp
#include "catlass/conv/tile/atlasa2/copy_l0c_to_gm.hpp"

using namespace Catlass::Conv::Tile;

using ElementSrc = float;
using ElementDst = half;
constexpr uint32_t Cout = 64, Ho = 14, Wo = 14, C0 = 16;

auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(Ho * Wo, Cout);
auto layoutDst = tla::MakeLayout<ElementDst, layout::NC1HWC0>(1, Cout / C0, Ho, Wo, C0);

AscendC::LocalTensor<ElementSrc> srcData;
AscendC::GlobalTensor<ElementDst> dstData;

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionGM{});

CopyL0CToGmTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
