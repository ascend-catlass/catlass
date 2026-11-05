# tile_copy（Conv）

> [代码位置](../../../../../../../../include/catlass/conv/tile/tile_copy.hpp)

[TOC]

## 概述

Conv `tile_copy` 是搬运聚合模板，组合引用 Conv 场景下 GM→L1、L1→L0A/L0B、L0C→GM 的子搬运组件，以类型成员方式暴露供 block 层 Conv 使用。

## API 清单

| API                                            | 风格   | 适用硬件           | 说明                            |
| :--------------------------------------------- | :----- | :----------------- | :------------------------------ |
| [TileCopy](./tile_copy.md)                     | 非 TLA | Atlas A2            | 基础聚合，4 子组件引用          |
| [PackedTileCopyTla](./packed_tile_copy_tla.md) | TLA    | Atlas A2, Ascend 950 | 自动推导中间 Layout，架构自适应 |

## 调用示例

### PackedTileCopyTla

```cpp
#include "catlass/conv/tile/tile_copy.hpp"

using namespace Catlass::Conv::Tile;

using ElementFmap = half;
using ElementFilter = half;
using ElementOutput = half;

using LayoutTagFmap = layout::NC1HWC0;
using LayoutTagFilter = layout::CI1KHKWCOCI0;
using LayoutTagOutput = layout::NC1HWC0;

using Copy = PackedTileCopyTla<Arch::AtlasA2,
    ElementFmap, LayoutTagFmap,
    ElementFilter, LayoutTagFilter,
    ElementOutput, LayoutTagOutput
>;
```
