# PackedMxTileCopyTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`PackedMxTileCopyTla` 继承自 [PackedTileCopyTla](./packed_tile_copy_tla.md)，**新增** MX Scale（微缩放）搬运通道：`CopyGmToL1MxScaleA` / `CopyGmToL1MxScaleB`。

MX Scale 是一种 FP8 量化中使用的块级（block-wise）scale，以 `float8_e8m0_t` 存储。Pack 阶段将 GM 上的 scale 通过 TLA 搬入 L1（A 侧转换为 zZ 布局，B 侧转换为 nN 布局），再由 `PackedMxTileCopyTla` 管理后续 Tile 操作的调度。

> **限制**：仅 AtlasA2 架构（`CATLASS_ARCH == 2201`）。

## 引用的 Tile 组件

| 成员别名                     | 来源                                                     |
| :--------------------------- | :------------------------------------------------------- |
| `CopyGmToL1A` ~ `CopyL1ToBT` | 继承自 `PackedTileCopyTla<ArchTag, ...>`                 |
| `CopyGmToL1MxScaleA`（新增） | `TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>` |
| `CopyGmToL1MxScaleB`（新增） | `TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>` |

## 模板原型

```cpp
template <
    class ArchTag,                                                   // 架构标签：Arch::AtlasA2
    class ElementA_,                                                 // A 矩阵元素类型（FP8）
    class LayoutTagA,                                                // A GM layout tag
    class ElementB_,                                                 // B 矩阵元素类型（FP8）
    class LayoutTagB,                                                // B GM layout tag
    class ElementMxScaleA_,                                          // A MX Scale 元素类型（float8_e8m0_t）
    class LayoutMxScaleA_,                                           // A MX Scale GM layout
    class ElementMxScaleB_,                                          // B MX Scale 元素类型（float8_e8m0_t）
    class LayoutMxScaleB_,                                           // B MX Scale GM layout
    class ElementC_,                                                 // C 矩阵元素类型
    class LayoutTagC,                                                // C GM layout tag
    class ElementBias = void,                                        // Bias 元素类型
    bool ReluEnable_ = false,                                        // ReLU 开关
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, // 反量化粒度
    class L0CCopyMode = CopyToGM                                     // L0C→Dst 模式
>
struct PackedMxTileCopyTla : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB,
    ElementC_, LayoutTagC, ElementBias, ReluEnable_, DEQUANT_GRANULARITY, L0CCopyMode>;
```

## 新增的成员

```cpp
using ElementMxScaleA = ElementMxScaleA_;
using ElementMxScaleB = ElementMxScaleB_;
using LayoutMxScaleA = LayoutMxScaleA_;
using LayoutMxScaleB = LayoutMxScaleB_;

// L1 布局固定为 zZ(A) 和 nN(B)
using LayoutTagL1MxScaleA = layout::zZ;
using LayoutTagL1MxScaleB = layout::nN;
using LayoutL1MxScaleA = detail::TagToLayout_t<ElementMxScaleA, LayoutTagL1MxScaleA>;
using LayoutL1MxScaleB = detail::TagToLayout_t<ElementMxScaleB, LayoutTagL1MxScaleB>;

template <class TensorMxScaleA>
using CopyGmToL1MxScaleA = TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>;

template <class TensorMxScaleB>
using CopyGmToL1MxScaleB = TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>;
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::PackedMxTileCopyTla<
    Arch::AtlasA2,
    float8_e4m3_t, layout::RowMajor,       // A: FP8 RowMajor
    float8_e4m3_t, layout::ColumnMajor,    // B: FP8 ColumnMajor
    float8_e8m0_t, layout::VectorLayout,   // MX Scale A: RowMajor（GM上）
    float8_e8m0_t, layout::VectorLayout,   // MX Scale B: ColumnMajor（GM上）
    half, layout::RowMajor>;              // C: half RowMajor

typename TileCopy_::CopyGmToL1A         copyGmToL1A;
typename TileCopy_::CopyGmToL1MxScaleA  copyGmToL1MxScaleA;  // 搬入 A scale
typename TileCopy_::CopyGmToL1MxScaleB  copyGmToL1MxScaleB;  // 搬入 B scale
typename TileCopy_::CopyL1ToL0A         copyL1ToL0A;
typename TileCopy_::CopyL0CToGm         copyL0CToGm;
```
