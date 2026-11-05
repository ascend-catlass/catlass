# TileCopyTlaExt

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## 功能说明

`TileCopyTlaExt` 是 `TileCopyTla` 的**扩展变体**，额外接收两个 LayoutTag 参数（`LayoutTagSrc`、`LayoutTagDst`）。用户可基于这两个 LayoutTag 自主分发偏特化，而不依赖 tensor 内部的 layout trait。

适用场景：src tensor 排布为 `PaddingRowMajor` 或 `PaddingColumnMajor`（`tla::MakeLayout` 不原生支持）时，通过 `LayoutTagSrc = PaddingRowMajor` 显式分发到对应偏特化。

与 [TileCopyTla](./tile_copy_tla.md) 的区别：

| 特性           | TileCopyTla                     | TileCopyTlaExt                                  |
| :------------- | :------------------------------ | :---------------------------------------------- |
| 分发方式       | SFINAE trait（`isRowMajor` 等） | 显式 LayoutTag 模板参数                         |
| LayoutTag 参数 | 无（自动推导）                  | `LayoutTagSrc`, `LayoutTagDst`                  |
| Padding layout | 不支持                          | 支持 via `PaddingRowMajor`/`PaddingColumnMajor` |

## 基类声明

```cpp
template <
    class ArchTag,           // 架构标签
    class TensorSrc,         // 源 tensor 类型
    class TensorDst,         // 目标 tensor 类型
    class LayoutTagSrc,      // 源 LayoutTag（偏特化匹配用）
    class LayoutTagDst       // 目标 LayoutTag（偏特化匹配用）
>
struct TileCopyTlaExt {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileCopyTlaExt, can not find the specialization.");
};
```

> **注意**：`LayoutTagSrc` 与 tensor 的物理 Layout 可能不匹配（如 `PaddingRowMajor` vs tensor 实际 `RowMajor`）。它仅用于偏特化分发标签。

## 偏特化实现清单（全 Atlas A2）

| LayoutTagSrc                     | LayoutTagDst    | 源位置     | 目标位置 | 实现位置                    | API 文档                                               |
| :------------------------------- | :-------------- | :--------- | :------- | :-------------------------- | :----------------------------------------------------- |
| RowMajor / PaddingRowMajor       | RowMajor        | GM         | L1 A1    | `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla_ext.md) |
| ColumnMajor / PaddingColumnMajor | RowMajor        | GM         | L1 A1    | `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla_ext.md) |
| RowMajor                         | PaddingRowMajor | UB VECCALC | GM       | `atlasa2/copy_ub_to_gm.hpp` | [copy_ub_to_gm](../copy_ub_to_gm/tile_copy_tla_ext.md) |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor
);
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using Element = half;

// A 矩阵 PaddingRowMajor（alignment 后）
auto gmLayout = tla::MakeLayout<Element, layout::PaddingRowMajor>(M, K_padded);
auto l1Layout = tla::MakeLayout<Element, layout::RowMajor>(M, K_padded);
auto gmTensor = tla::MakeTensor(gmData, gmLayout, Arch::PositionGM{});
auto l1Tensor = tla::MakeTensor(l1Data, l1Layout, Arch::PositionL1{});

// 显式分发：PaddingRowMajor → RowMajor
TileCopyTlaExt<Arch::AtlasA2, decltype(gmTensor), decltype(l1Tensor),
    layout::PaddingRowMajor, layout::RowMajor> copyOp;
copyOp(l1Tensor, gmTensor);
```
