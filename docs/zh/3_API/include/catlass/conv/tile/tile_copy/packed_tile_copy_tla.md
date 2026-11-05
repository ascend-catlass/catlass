# PackedTileCopyTla

> [代码位置](../../../../../../../../include/catlass/conv/tile/tile_copy.hpp)

[TOC]

## 功能说明

`PackedTileCopyTla` 是 Conv 搬运聚合的 TLA 版本（Atlas A2 + Ascend 950）。从 `ConvType`（Element + LayoutTag）出发，通过 `detail::TagToLayout_t` 自动推导所有中间 L1/L0A/L0B/L0C Layout，并组合 TLA 搬运子组件。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品
- 风格：TLA

## 模板原型

```cpp
template <class ArchTag,
          class ElementFmap_, class LayoutTagFmap_,
          class ElementFilter_, class LayoutTagFilter_,
          class ElementOutput_, class LayoutTagOutput_,
          class ElementBias = void, bool ReluEnable_ = false,
          ScaleGranularity DEQUANT_GRANULARITY_ = NO_QUANT>
struct PackedTileCopyTla;
```

| 模板参数               | 说明                             |
| :--------------------- | :------------------------------- |
| `ArchTag`              | 架构标签                         |
| `ElementFmap_`         | Fmap 元素类型                    |
| `LayoutTagFmap_`       | Fmap LayoutTag（NC1HWC0）        |
| `ElementFilter_`       | Filter 元素类型                  |
| `LayoutTagFilter_`     | Filter LayoutTag（CI1KHKWCOCI0） |
| `ElementOutput_`       | Output 元素类型                  |
| `LayoutTagOutput_`     | Output LayoutTag（NC1HWC0）      |
| `ElementBias`          | Bias 元素类型，默认 `void`       |
| `ReluEnable_`          | ReLU 开关                        |
| `DEQUANT_GRANULARITY_` | 量化模式                         |

## 成员类型定义

| 成员类型                       | 说明                                            |
| :----------------------------- | :---------------------------------------------- |
| `CopyGmToL1A`                  | `Conv::Tile::CopyGmToL1ATla<ElementFmap>`       |
| `CopyGmToL1B`                  | `Conv::Tile::CopyGmToL1BTla<ElementFilter>`     |
| `CopyL1ToL0A`                  | `Conv::Tile::CopyL1ToL0ATla<ElementFmap>`       |
| `CopyL1ToL0B`                  | `Conv::Tile::CopyL1ToL0BTla<ElementFilter>`     |
| `CopyL0CToGm` / `CopyL0CToDst` | `Conv::Tile::CopyL0CToGmTla<...>`（架构自适应） |

内部通过 `detail::TagToLayout_t` 自动推导 L1/L0/L0C 中间 Layout，通过 `L1AlignHelper` 处理对齐。

## 调用示例

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

// 子组件引用:
// typename Copy::CopyGmToL1A
// typename Copy::CopyGmToL1B
// typename Copy::CopyL1ToL0A
// typename Copy::CopyL1ToL0B
// typename Copy::CopyL0CToGm<decltype(dstTensor)>
```
