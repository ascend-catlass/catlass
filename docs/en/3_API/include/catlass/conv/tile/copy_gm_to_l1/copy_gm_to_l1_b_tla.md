# CopyGmToL1BTla

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_gm_to_l1.hpp)

[TOC]

## Function

`CopyGmToL1BTla` implements filter (convolution kernel) data movement from global memory to L1 in TLA style in convolution scenarios. It corresponds to matrix B, with layout transformation `CI1KHKWCOCI0` → `CI1KHKWCOCI0`.

- Applicability: Atlas A2 series products and Atlas A3 series products and Ascend 950PR&950DT series products
- Style: TLA

## Template Prototype

```cpp
template <class Element>
struct CopyGmToL1BTla;
```

| Parameter| Description|
| :------ | :------ |
| `Element` | Element type, for example, `half`|

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // tla::Tensor<LocalTensor<Element>, CI1KHKWCOCI0, ..., L1>
    TensorSrc const &srcTensor     // tla::Tensor<GlobalTensor<Element>, CI1KHKWCOCI0, ..., GM>
)
```

## Examples

```cpp
#include "catlass/conv/tile/atlasa2/copy_gm_to_l1.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Kh = 3, Kw = 3, Cout = 64, C0 = 16;

auto layoutSrc = tla::MakeLayout<Element, layout::CI1KHKWCOCI0>(Cin1, Kh, Kw, Cout, C0);
auto layoutDst = tla::MakeLayout<Element, layout::CI1KHKWCOCI0>(Cin1, Kh, Kw, Cout, C0);

AscendC::GlobalTensor<Element> srcData;
AscendC::LocalTensor<Element> dstData;

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL1{});

CopyGmToL1BTla<Element> copyOp;
copyOp(dstTensor, srcTensor);
```
