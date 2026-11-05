# CopyL1ToL0BTla

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0b.hpp)

[TOC]

## Function

`CopyL1ToL0BTla` implements filter data movement from L1 to L0B (TLA style) in convolution scenarios.

- Applicability: Atlas A2 series products and Atlas A3 series products and Ascend 950PR&950DT series products
- Style: TLA

## Template Prototype

```cpp
template <class Element>
struct CopyL1ToL0BTla;
```

| Parameter| Description|
| :------ | :------ |
| `Element` | Element type, for example, `half`|

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // nZ format
    TensorSrc const &srcTensor     // CI1KHKWCOCI0 format
)
```

## Examples

```cpp
#include "catlass/conv/tile/atlasa2/copy_l1_to_l0b.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Kh = 3, Kw = 3, Cout = 64, C0 = 16;

auto layoutSrc = tla::MakeLayout<Element, layout::CI1KHKWCOCI0>(Cin1, Kh, Kw, Cout, C0);
auto layoutDst = tla::MakeLayout<Element, layout::nZ>(Cin1 * Kh * Kw, Cout);

AscendC::LocalTensor<Element> srcData, dstData;
auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL0B{});

CopyL1ToL0BTla<Element> copyOp;
copyOp(dstTensor, srcTensor);
```
