# copy_l0c_to_gm

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_l0c_to_gm.hpp)

[TOC]

## 概述

`copy_l0c_to_gm` 模块实现 Conv 场景下累加结果从 L0C（zN 格式）写回 GM（NC1HWC0 格式）。通过 `AscendC::Fixpipe` 直接通路完成数据搬运、类型转换和可选 ReLU。

## API 清单

| API                                       | 风格   | 适用硬件           | 说明                          |
| :---------------------------------------- | :----- | :----------------- | :---------------------------- |
| [CopyL0CToGm](./copy_l0c_to_gm.md)        | 非 TLA | Atlas A2            | Fixpipe + F322F16/BF16 + ReLU |
| [CopyL0CToGmTla](./copy_l0c_to_gm_tla.md) | TLA    | Atlas A2, Ascend 950 | TLA 版本                      |

## 调用示例

### CopyL0CToGmTla（TLA）

```cpp
#include "catlass/conv/tile/atlasa2/copy_l0c_to_gm.hpp"

using namespace Catlass::Conv::Tile;

using ElementSrc = float;
using ElementDst = half;
constexpr uint32_t Cout = 64, Ho = 14, Wo = 14, C0 = 16;

auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(Ho * Wo, Cout);
auto layoutDst = tla::MakeLayout<ElementDst, layout::NC1HWC0>(1, Cout / C0, Ho, Wo, C0);

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionGM{});

CopyL0CToGmTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
