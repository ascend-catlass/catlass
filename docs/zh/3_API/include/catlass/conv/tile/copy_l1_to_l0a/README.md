# copy_l1_to_l0a

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0a.hpp)

[TOC]

## 概述

`copy_l1_to_l0a` 模块实现 Conv 场景下将 Fmap 数据从 L1 搬运到 L0A，同时完成 im2col 操作（NC1HWC0→zZ）。

## API 清单

| API                                       | 风格   | 适用硬件           | 说明                          |
| :---------------------------------------- | :----- | :----------------- | :---------------------------- |
| [CopyL1ToL0A](./copy_l1_to_l0a.md)        | 非 TLA | Atlas A2           | LoadData 3D v2，含 im2col     |
| [CopyL1ToL0ATla](./copy_l1_to_l0a_tla.md) | TLA    | Atlas A2, Ascend 950 | 950 新增 `LoadDataWithStride` |

## 调用示例

### CopyL1ToL0ATla（TLA）

```cpp
#include "catlass/conv/tile/atlasa2/copy_l1_to_l0a.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Hi = 28, Wi = 28, C0 = 16;
constexpr uint32_t Kh = 3, Kw = 3;

auto layoutSrc = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);
auto layoutDst = tla::MakeLayout<Element, layout::zZ>(16, 27);

Conv2dFilterParams params{.strideW_ = 1, .strideH_ = 1, .kw_ = Kw, .kh_ = Kh, .dilationW_ = 1, .dilationH_ = 1};

uint8_t padList[4] = {0, 0, 0, 0};

CopyL1ToL0ATla<Element> copyOp(params);
copyOp(dstTensor, srcTensor, padList);
```
