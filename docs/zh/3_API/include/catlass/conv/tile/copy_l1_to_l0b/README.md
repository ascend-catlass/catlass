# copy_l1_to_l0b

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0b.hpp)

[TOC]

## 概述

`copy_l1_to_l0b` 模块实现 Conv 场景下将 Filter（卷积核）数据从 L1 搬运到 L0B（CI1KHKWCOCI0→nZ）。

## API 清单

| API                                       | 风格   | 适用硬件           | 说明        |
| :---------------------------------------- | :----- | :----------------- | :---------- |
| [CopyL1ToL0B](./copy_l1_to_l0b.md)        | 非 TLA | Atlas A2            | LoadData 2D |
| [CopyL1ToL0BTla](./copy_l1_to_l0b_tla.md) | TLA    | Atlas A2, Ascend 950 | TLA 版本    |

## 调用示例

### CopyL1ToL0BTla（TLA）

```cpp
#include "catlass/conv/tile/atlasa2/copy_l1_to_l0b.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Kh = 3, Kw = 3, Cout = 64, C0 = 16;

auto layoutSrc = tla::MakeLayout<Element, layout::CI1KHKWCOCI0>(Cin1, Kh, Kw, Cout, C0);
auto layoutDst = tla::MakeLayout<Element, layout::nZ>(Cin1 * Kh * Kw, Cout);

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL0B{});

CopyL1ToL0BTla<Element> copyOp;
copyOp(dstTensor, srcTensor);
```
