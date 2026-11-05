# PaddingPackedTileCopyTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`PaddingPackedTileCopyTla` 是支持 Padding 的 TLA 搬运模板集合。与 [PackedTileCopyTla](./packed_tile_copy_tla.md) 的关键区别在于 `CopyGmToL1A/B` 使用 `TileCopyTlaExt`（而非 `TileCopyTla`），支持源端 PaddingRowMajor / PaddingColumnMajor layout。

适用于矩阵 Block 对齐后产生 Padding 的场景。`IS_PADDING_A` / `IS_PADDING_B` 控制是否启用 Padding layout tag。

## 引用的 Tile 组件

| 成员别名                   | 引用的底层模板                                                                      | 说明                 |
| :------------------------- | :---------------------------------------------------------------------------------- | :------------------- |
| `CopyGmToL1A`              | `TileCopyTlaExt<ArchTag, TensorA, TensorL1A, PaddingTag/RowMajor, LayoutTagL1A>`    | A 矩阵 GM→L1（Ext）  |
| `CopyGmToL1B`              | `TileCopyTlaExt<ArchTag, TensorB, TensorL1B, PaddingTag/ColumnMajor, LayoutTagL1B>` | B 矩阵 GM→L1（Ext）  |
| `CopyL1ToL0A`              | `TileCopyTla<ArchTag, TensorL1A, TensorL0A>`                                        | A 矩阵 L1→L0A（TLA） |
| `CopyL1ToL0B`              | `TileCopyTla<ArchTag, TensorL1B, TensorL0B>`                                        | B 矩阵 L1→L0B（TLA） |
| `CopyL0CToDst` (Ascend 950) | `CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>`                                       | L0C→Dst              |
| `CopyL0CToGm` (Atlas A2)    | `CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>`                                       | L0C→GM               |

## 模板原型

```cpp
template <
    class ArchTag,                                                   // 架构标签
    class TensorA,                                                   // A tensor
    class LayoutTagA,                                                // A GM layout tag
    class TensorB,                                                   // B tensor
    class LayoutTagB,                                                // B GM layout tag
    class TensorC,                                                   // C tensor
    class LayoutTagC,                                                // C GM layout tag
    class TensorBias = void,                                         // Bias tensor
    class LayoutTagBias = void,                                      // Bias layout tag
    bool IS_PADDING_A = false,                                       // A 矩阵是否需要 Padding
    bool IS_PADDING_B = false                                        // B 矩阵是否需要 Padding
>
struct PaddingPackedTileCopyTla;
```

> **LayoutTagA / LayoutTagB 约束**：仅支持 `layout::RowMajor` 或 `layout::ColumnMajor`。

## Padding 逻辑

当 `IS_PADDING_A = true` 时：

```cpp
using LayoutPaddingTagA = std::conditional_t<
    std::is_same_v<LayoutTagA, layout::RowMajor>,
    layout::PaddingRowMajor,
    layout::PaddingColumnMajor>;

using CopyGmToL1A = TileCopyTlaExt<ArchTag, TensorA, TensorL1A,
    LayoutPaddingTagA, LayoutTagL1A>;   // Ext + Padding
```

## 与 PackedTileCopyTla 的区别

| 模板                       | GM→L1 算子       | 源端 Padding        | 模板参数风格                     |
| :------------------------- | :--------------- | :------------------ | :------------------------------- |
| `PackedTileCopyTla`        | `TileCopyTla`    | 不支持              | Element + LayoutTag（非 Tensor） |
| `PaddingPackedTileCopyTla` | `TileCopyTlaExt` | 支持 IS_PADDING_A/B | Tensor 参数（含 Padding Layout） |

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm;

using ElementA = half;
using ElementB = half;
using ElementC = half;

// A 矩阵 padding 后
auto layoutA = tla::MakeLayout<ElementA, layout::PaddingRowMajor>(M, K_padded);
auto tensorA = tla::MakeTensor(gmTensorA, layoutA, Arch::PositionGM{});

auto layoutB = tla::MakeLayout<ElementB, layout::ColumnMajor>(K_padded, N);
auto tensorB = tla::MakeTensor(gmTensorB, layoutB, Arch::PositionGM{});

auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(M, N);
auto tensorC = tla::MakeTensor(gmTensorC, layoutC, Arch::PositionGM{});

using TileCopy_ = Tile::PaddingPackedTileCopyTla<
    Arch::AtlasA2,
    decltype(tensorA), layout::RowMajor,      // PaddingRowMajor → RowMajor
    decltype(tensorB), layout::ColumnMajor,
    decltype(tensorC), layout::RowMajor,
    void, void,
    true, false>;                              // IS_PADDING_A = true

typename TileCopy_::CopyGmToL1A copyGmToL1A;  // TileCopyTlaExt, PaddingRowMajor
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToGm copyL0CToGm;
```
