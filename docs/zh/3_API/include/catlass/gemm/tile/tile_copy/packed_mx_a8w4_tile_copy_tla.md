# PackedMxA8W4TileCopyTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`PackedMxA8W4TileCopyTla` 继承自 [PackedTileCopyTla](./packed_tile_copy_tla.md)，同时管理 MX Scale 搬运和 A8W4 量化 B 矩阵搬运。它是 [PackedMxTileCopyTla](./packed_mx_tile_copy_tla.md) 在 A8W4（INT4 weight）场景下的扩展变体。

关键特点：

- A 矩阵为 FP8（`ElementA_`），通过 MX Scale 搬运
- B 矩阵为 INT4（`ElementPrologueB_`，Prologue 前类型）→ INT8（`ElementB_`，Prologue 后类型）
- **重写** `CopyL1ToL0B` 以适配 Prologue 后的 B 数据类型

> **限制**：仅 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 引用的 Tile 组件

| 成员别名                                  | 来源                                                                                           |
| :---------------------------------------- | :--------------------------------------------------------------------------------------------- |
| `CopyGmToL1A` ~ `CopyL1ToBT`（除 L0B 外） | 继承自 `PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagPrologueB, ...>` |
| `CopyL1ToL0B`（重写）                     | `TileCopyTla<ArchTag, TensorL1B, TensorL0B>`（Prologue 后类型）                                |
| `CopyGmToL1MxScaleA`（新增）              | `TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>`                                       |
| `CopyGmToL1MxScaleB`（新增）              | `TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>`                                       |

## 模板原型

```cpp
template <
    class ArchTag,                                                   // 架构标签：Arch::AtlasA2
    class ElementA_,                                                 // A 矩阵元素类型（FP8）
    class LayoutTagA,                                                // A GM layout tag
    class ElementPrologueB_,                                         // B Prologue 前元素类型（INT4）
    class LayoutTagPrologueB,                                        // B Prologue 前 GM layout tag
    class ElementB_,                                                 // B Prologue 后元素类型（INT8）
    class LayoutTagB,                                                // B Prologue 后 GM layout tag
    class ElementMxScaleA_,                                          // A MX Scale 元素类型
    class LayoutMxScaleA_,                                           // A MX Scale GM layout
    class ElementMxScaleB_,                                          // B MX Scale 元素类型
    class LayoutMxScaleB_,                                           // B MX Scale GM layout
    class ElementC_,                                                 // C 矩阵元素类型
    class LayoutTagC,                                                // C GM layout tag
    class ElementBias = void,                                        // Bias 元素类型
    bool ReluEnable_ = false,                                        // ReLU 开关
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, // 反量化粒度
    class L0CCopyMode = CopyToGM                                     // L0C→Dst 模式
>
struct PackedMxA8W4TileCopyTla : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA,
    ElementB_, LayoutTagPrologueB, ElementC_, LayoutTagC, ...>;
```

> **注意**：基类 `PackedTileCopyTla` 使用 `ElementB_` + `LayoutTagPrologueB` 作为 B 矩阵的 Prologue 前类型，`ElementB_` 为 Prologue 后（INT8）。

## 重写与新增的成员

```cpp
// 重写 B 矩阵布局推导：基于 Prologue 后类型
using LayoutB     = detail::TagToLayout_t<ElementPrologueB_, LayoutTagPrologueB>;
using LayoutL1B   = detail::TagToLayout_t<ElementB_, LayoutTagL1B>;
using LayoutL0B   = detail::TagToLayout_t<ElementB_, LayoutTagL0B>;
using TensorL1B   = tla::Tensor<LocalTensor<ElementB_>, LayoutL1B, Coord<0,0>, A1>;
using TensorL0B   = tla::Tensor<LocalTensor<ElementB_>, LayoutL0B, Coord<0,0>, B2>;

// 重写：Prologue 后类型的 L1→L0B
using CopyL1ToL0B = TileCopyTla<ArchTag, TensorL1B, TensorL0B>;

// 新增 MX Scale
template <class TensorMxScaleA> using CopyGmToL1MxScaleA = TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>;
template <class TensorMxScaleB> using CopyGmToL1MxScaleB = TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>;
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::PackedMxA8W4TileCopyTla<
    Arch::AtlasA2,
    float8_e4m3_t, layout::RowMajor,          // A: FP8 RowMajor
    int8_t, layout::RowMajor,                 // B Prologue前: INT4 packed RowMajor
    int8_t, layout::RowMajor,                 // B Prologue后: INT8 RowMajor
    float8_e8m0_t, layout::VectorLayout,      // MX Scale A
    float8_e8m0_t, layout::VectorLayout,      // MX Scale B
    half, layout::RowMajor>;                  // C: half RowMajor

typename TileCopy_::CopyGmToL1A         copyGmToL1A;
typename TileCopy_::CopyGmToL1B         copyGmToL1B;       // INT4 → GM→L1
typename TileCopy_::CopyL1ToL0B         copyL1ToL0B;       // INT8 L1→L0B（Prologue后）
typename TileCopy_::CopyGmToL1MxScaleA  copyGmToL1MxScaleA;
typename TileCopy_::CopyGmToL1MxScaleB  copyGmToL1MxScaleB;
typename TileCopy_::CopyL0CToGm         copyL0CToGm;
```
