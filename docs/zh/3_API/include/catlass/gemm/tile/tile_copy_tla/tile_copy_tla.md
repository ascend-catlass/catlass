# TileCopyTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## 功能说明

`TileCopyTla` 是 TLA（Tensor Layout Abstraction）风格的核心搬运模板，通过 SFINAE 自动分发到对应的架构特化实现。两个模板参数分别是源 Tensor 和目标 Tensor 的类型，不同 layout 和 position 的组合会自动匹配相应的偏特化。

与 [TileCopyTlaExt](./tile_copy_tla_ext.md) 的区别：`TileCopyTla` 完全由 SFINAE trait（`isRowMajor`、`iszN` 等）自动分发，无需手动指定 LayoutTag。

## 基类声明

```cpp
template <
    class ArchTag,           // 架构标签
    class TensorSrc,         // 源 tensor 类型
    class TensorDst,         // 目标 tensor 类型
    class Enable = void      // SFINAE enable
>
struct TileCopyTla {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileCopyTla, can not find the specialization.");
};
```

## 偏特化实现清单

### Atlas A2（Arch::AtlasA2，CATLASS_ARCH == 2201）

| 方向                    | SFINAE 条件                                             | 实现位置                     | API 文档                                             |
| :---------------------- | :------------------------------------------------------ | :--------------------------- | :--------------------------------------------------- |
| GM→L1 (RowMajor)        | `isRowMajor<LayoutSrc>` && `isRowMajor<LayoutDst>`      | `atlasa2/copy_gm_to_l1.hpp`  | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md)   |
| GM→L1 (ColumnMajor→zZ)  | `isColumnMajor<LayoutSrc>` && `isRowMajor<LayoutDst>`   | `atlasa2/copy_gm_to_l1.hpp`  | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md)   |
| GM→L1 (VectorLayout)    | `isVector<LayoutSrc>` && `isVector<LayoutDst>`          | `atlasa2/copy_gm_to_l1.hpp`  | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md)   |
| L1→L0A (RowMajor→zZ)    | `isRowMajor<LayoutSrc>` && `hasL0ALayout<LayoutDst>`    | `atlasa2/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_tla.md) |
| L1→L0A (zN→zZ)          | `iszN<LayoutSrc>` && `hasL0ALayout<LayoutDst>`          | `atlasa2/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_tla.md) |
| L1→L0B (ColumnMajor→nZ) | `isColumnMajor<LayoutSrc>` && `hasL0BLayout<LayoutDst>` | `atlasa2/copy_l1_to_l0b.hpp` | [copy_l1_to_l0b](../copy_l1_to_l0b/tile_copy_tla.md) |
| L1→L0B (zN→nZ)          | `iszN<LayoutSrc>` && `hasL0BLayout<LayoutDst>`          | `atlasa2/copy_l1_to_l0b.hpp` | [copy_l1_to_l0b](../copy_l1_to_l0b/tile_copy_tla.md) |
| GM→UB                   | `isRowMajor<LayoutSrc>` && `isRowMajor<LayoutDst>`      | `atlasa2/copy_gm_to_ub.hpp`  | [copy_gm_to_ub](../copy_gm_to_ub/tile_copy_tla.md)   |
| UB→GM                   | `isRowMajor<LayoutSrc>` && `isRowMajor<LayoutDst>`      | `atlasa2/copy_ub_to_gm.hpp`  | [copy_ub_to_gm](../copy_ub_to_gm/tile_copy_tla.md)   |

### Ascend 950（Arch::Ascend950，CATLASS_ARCH == 3510）

| 方向                    | SFINAE 条件                                             | 实现位置                       | API 文档                                             |
| :---------------------- | :------------------------------------------------------ | :----------------------------- | :--------------------------------------------------- |
| GM→L1                   | `isRowMajor<LayoutSrc>` && `isRowMajor<LayoutDst>`      | `ascend950/copy_gm_to_l1.hpp`  | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md)   |
| GM→L1 (ColumnMajor)     | `isColumnMajor<LayoutSrc>` && `isRowMajor<LayoutDst>`   | `ascend950/copy_gm_to_l1.hpp`  | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md)   |
| L1→L0A (RowMajor→zZ)    | `isRowMajor<LayoutSrc>` && `hasL0ALayout<LayoutDst>`    | `ascend950/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_tla.md) |
| L1→L0A (zN→zZ)          | `iszN<LayoutSrc>` && `hasL0ALayout<LayoutDst>`          | `ascend950/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_tla.md) |
| L1→L0B (ColumnMajor→nZ) | `isColumnMajor<LayoutSrc>` && `hasL0BLayout<LayoutDst>` | `ascend950/copy_l1_to_l0b.hpp` | [copy_l1_to_l0b](../copy_l1_to_l0b/tile_copy_tla.md) |
| L1→BT                   | `isVector<LayoutSrc>` && `isVector<LayoutDst>`          | `ascend950/copy_l1_to_bt.hpp`  | [copy_l1_to_bt](../copy_l1_to_bt/tile_copy_tla.md)   |

## 调用接口

所有偏特化均提供统一的 `operator()`：

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // 目标 tensor
    TensorSrc const &srcTensor     // 源 tensor
);
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"  // 拉入偏特化
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using Element = half;

auto gmLayout = tla::MakeLayout<Element, layout::RowMajor>(M, K);
auto l1Layout = tla::MakeLayout<Element, layout::RowMajor>(M, K);
auto gmTensor = tla::MakeTensor(gmData, gmLayout, Arch::PositionGM{});
auto l1Tensor = tla::MakeTensor(l1Data, l1Layout, Arch::PositionL1{});

// SFINAE 自动匹配：GM RowMajor → L1 RowMajor
TileCopyTla<Arch::AtlasA2, decltype(gmTensor), decltype(l1Tensor)> copyOp;
copyOp(l1Tensor, gmTensor);
```
