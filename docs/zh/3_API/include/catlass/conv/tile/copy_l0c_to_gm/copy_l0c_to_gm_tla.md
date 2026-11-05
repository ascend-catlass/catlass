# CopyL0CToGmTla

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_l0c_to_gm.hpp)

[TOC]

## 功能说明

`CopyL0CToGmTla` 实现 Conv 场景下累加结果从 L0C 写回 GM 的 TLA 风格版本。通过 `AscendC::Fixpipe` 直接通路完成搬运、类型转换和可选 ReLU。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品
- 风格：TLA

## 模板原型

```cpp
template <class ArchTag, class TensorSrc, class TensorDst,
          ScaleGranularity DEQUANT_GRANULARITY = NO_QUANT, bool ReluEnable = false>
struct CopyL0CToGmTla;
```

| 模板参数              | 说明                           |
| :-------------------- | :----------------------------- |
| `ArchTag`             | 架构标签                       |
| `TensorSrc`           | TLA Tensor 类型（L0C, zN）     |
| `TensorDst`           | TLA Tensor 类型（GM, NC1HWC0） |
| `DEQUANT_GRANULARITY` | 量化模式                       |
| `ReluEnable`          | 是否启用 ReLU                  |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor,
    uint8_t unitFlag = 0
)
```

## 调用示例

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
