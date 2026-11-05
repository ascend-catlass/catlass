# PackedTileCopyTla

> [Code Location](../../../../../../../../include/catlass/conv/tile/tile_copy.hpp)

[TOC]

## Function

`PackedTileCopyTla` is the TLA version (Atlas A2 and Ascend 950) for convolution movement and aggregation. Starting from `ConvType` (Element and LayoutTag), it uses `detail::TagToLayout_t` to automatically infer all intermediate layouts for L1/L0A/L0B/L0C and then combines the TLA movement child components.

- Applicability: Atlas A2 series products and Atlas A3 series products and Ascend 950PR&950DT series products
- Style: TLA

## Template Prototype

```cpp
template <class ArchTag,
          class ElementFmap_, class LayoutTagFmap_,
          class ElementFilter_, class LayoutTagFilter_,
          class ElementOutput_, class LayoutTagOutput_,
          class ElementBias = void, bool ReluEnable_ = false,
          ScaleGranularity DEQUANT_GRANULARITY_ = NO_QUANT>
struct PackedTileCopyTla;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `ElementFmap_` | Fmap element type|
| `LayoutTagFmap_` | Fmap LayoutTag (NC1HWC0)|
| `ElementFilter_` | Filter element type|
| `LayoutTagFilter_` | Filter LayoutTag (CI1KHKWCOCI0)|
| `ElementOutput_` | Output element type|
| `LayoutTagOutput_` | Output LayoutTag (NC1HWC0)|
| `ElementBias` | Bias element type. The default value is `void`.|
| `ReluEnable_` | ReLU switch|
| `DEQUANT_GRANULARITY_` | Quantization mode|

## Member Types

| Member Type| Description|
| :------ | :------ |
| `CopyGmToL1A` | `Conv::Tile::CopyGmToL1ATla<ElementFmap>` |
| `CopyGmToL1B` | `Conv::Tile::CopyGmToL1BTla<ElementFilter>` |
| `CopyL1ToL0A` | `Conv::Tile::CopyL1ToL0ATla<ElementFmap>` |
| `CopyL1ToL0B` | `Conv::Tile::CopyL1ToL0BTla<ElementFilter>` |
| `CopyL0CToGm` / `CopyL0CToDst` | `Conv::Tile::CopyL0CToGmTla<...>` (architecture adaptation)|

Internally, the L1/L0/L0C intermediate layout is automatically inferred by `detail::TagToLayout_t` and aligned by `L1AlignHelper`.

## Examples

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

// Child component references
// typename Copy::CopyGmToL1A
// typename Copy::CopyGmToL1B
// typename Copy::CopyL1ToL0A
// typename Copy::CopyL1ToL0B
// typename Copy::CopyL0CToGm<decltype(dstTensor)>
```
