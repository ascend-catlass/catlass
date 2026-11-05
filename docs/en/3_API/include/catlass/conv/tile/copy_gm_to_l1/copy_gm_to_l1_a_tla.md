# CopyGmToL1ATla

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_gm_to_l1.hpp)

[TOC]

## Function

`CopyGmToL1ATla` implements Fmap (feature map) data movement from global memory to L1 in TLA style in convolution scenarios. It corresponds to the matrix A, with layout transformation `NC1HWC0` → `NC1HWC0`.

- Applicability: Atlas A2 series products and Atlas A3 series products and Ascend 950PR&950DT series products
- Style: TLA

## Template Prototype

```cpp
template <class Element>
struct CopyGmToL1ATla;
```

| Parameter| Description|
| :------ | :------ |
| `Element` | Element type, for example, `half`|

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // tla::Tensor<LocalTensor<Element>, NC1HWC0, ..., L1>
    TensorSrc const &srcTensor     // tla::Tensor<GlobalTensor<Element>, NC1HWC0, ..., GM>
)
```

## Examples

```cpp
#include "catlass/conv/tile/atlasa2/copy_gm_to_l1.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Hi = 28, Wi = 28, C0 = 16;

auto layoutSrc = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);
auto layoutDst = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);

AscendC::GlobalTensor<Element> srcData;
AscendC::LocalTensor<Element> dstData;

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL1{});

CopyGmToL1ATla<Element> copyOp;
copyOp(dstTensor, srcTensor);
```
