# copy_gm_to_l1

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_gm_to_l1.hpp)

[TOC]

## 概述

`copy_gm_to_l1` 模块实现 Conv 场景下从 GM（Global Memory）到 L1 的权重/特征图数据搬运。支持 Fmap（特征图，NC1HWC0）和 Filter（卷积核，CI1KHKWCOCI0）两类搬运，每种均有非 TLA 和 TLA 风格。

## API 清单

| API                                        | 风格   | 适用硬件           | 搬运对象        | 说明                      |
| :----------------------------------------- | :----- | :----------------- | :-------------- | :------------------------ |
| [CopyGmToL1](./copy_gm_to_l1.md)           | 非 TLA | Atlas A2            | Fmap / Filter   | 基础搬运，含偏特化分发    |
| [CopyGmToL1ATla](./copy_gm_to_l1_a_tla.md) | TLA    | Atlas A2, Ascend 950 | Fmap (A 矩阵)   | NC1HWC0→NC1HWC0           |
| [CopyGmToL1BTla](./copy_gm_to_l1_b_tla.md) | TLA    | Atlas A2, Ascend 950 | Filter (B 矩阵) | CI1KHKWCOCI0→CI1KHKWCOCI0 |

## 调用示例

### CopyGmToL1（非 TLA，Fmap）

```cpp
#include "catlass/conv/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Conv::Tile;

constexpr uint32_t Cin1 = 4, Hi = 28, Wi = 28, C0 = 16;

using Element = half;
using LayoutTagSrc = layout::NC1HWC0;
using LayoutTagDst = layout::NC1HWC0;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(1, Cin1, Hi, Wi, C0);
auto layoutDst = LayoutTagDst::MakeLayout<Element>(1, Cin1, Hi, Wi, C0);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using CopyOp = CopyGmToL1<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### CopyGmToL1ATla（TLA，Fmap）

```cpp
auto layoutSrc = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);
auto layoutDst = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL1{});

CopyGmToL1ATla<Element> copyOp;
copyOp(dstTensor, srcTensor);
```
