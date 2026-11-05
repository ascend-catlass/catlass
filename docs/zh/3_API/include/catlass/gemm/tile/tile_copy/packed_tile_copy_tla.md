# PackedTileCopyTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`PackedTileCopyTla` 是 GEMM Tile 层**最核心**的 TLA 搬运模板集合。所有搬运算子均为 `TileCopyTla` 或 `CopyL0CToGmTla` 类型，通过 LayoutTag 驱动完整的布局推导链：GM LayoutTag → L1 LayoutTag → L0 LayoutTag → tla::Layout → tla::Tensor。

支持 Relu、Dequant（per-tensor / per-channel）、Bias 等功能，通过模板参数开关。

## 引用的 Tile 组件

| 成员别名                   | 引用的底层模板                                                                 | 说明                       |
| :------------------------- | :----------------------------------------------------------------------------- | :------------------------- |
| `CopyGmToL1A`              | `TileCopyTla<ArchTag, TensorA, TensorL1A>`                                     | A 矩阵 GM→L1（TLA）        |
| `CopyGmToL1B`              | `TileCopyTla<ArchTag, TensorB, TensorL1B>`                                     | B 矩阵 GM→L1（TLA）        |
| `CopyGmToL1Bias`           | `TileCopyTla<ArchTag, TensorBias, TensorL1Bias>` 或 `EmptyClass`               | Bias GM→L1（条件性）       |
| `CopyGmToL1Scale`          | `TileCopyTla<ArchTag, TensorQuant, TensorL1Quant>` 或 `EmptyClass`             | Scale GM→L1（per-channel） |
| `CopyL1ToL0A`              | `TileCopyTla<ArchTag, TensorL1A, TensorL0A>`                                   | A 矩阵 L1→L0A（TLA）       |
| `CopyL1ToL0B`              | `TileCopyTla<ArchTag, TensorL1B, TensorL0B>`                                   | B 矩阵 L1→L0B（TLA）       |
| `CopyL1ToBT`               | `TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>` 或 `EmptyClass`             | Bias L1→BT（条件性）       |
| `CopyL0CToDst` (Ascend 950) | `CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>` | L0C→Dst（TLA）             |
| `CopyL0CToGm` (Atlas A2)    | `CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>` | L0C→GM（TLA）              |

## 模板原型

```cpp
template <
    class ArchTag,                                                   // 架构标签
    class ElementA_,                                                 // A 矩阵元素类型
    class LayoutTagA_,                                               // A 矩阵 GM layout tag
    class ElementB_,                                                 // B 矩阵元素类型
    class LayoutTagB_,                                               // B 矩阵 GM layout tag
    class ElementC_,                                                 // C 矩阵元素类型
    class LayoutTagC_,                                               // C 矩阵 GM layout tag
    class ElementBias = void,                                        // Bias 元素类型（可选）
    bool ReluEnable_ = false,                                        // ReLU 开关
    ScaleGranularity DEQUANT_GRANULARITY_ = ScaleGranularity::NO_QUANT, // 反量化粒度
    class L0CCopyMode = CopyToGM                                     // L0C→Dst 模式
>
struct PackedTileCopyTla;
```

## 模板参数说明

| 参数                   | 默认值     | 说明                                           |
| :--------------------- | :--------- | :--------------------------------------------- |
| `ElementBias`          | `void`     | 如非 void，则开启 Bias 搬运通道                |
| `ReluEnable_`          | `false`    | 传递给 `CopyL0CToGmTla` 的 ReLU 开关           |
| `DEQUANT_GRANULARITY_` | `NO_QUANT` | `PER_TENSOR` / `PER_CHANNEL` / `NO_QUANT`      |
| `L0CCopyMode`          | `CopyToGM` | Atlas A2 用 `CopyToGM`，Ascend 950 用 `CopyToUB` |

## 布局推导链（以 RowMajor A 为例）

```cpp
LayoutTagA_ = RowMajor
  → L1ATypeSelector → LayoutTagL1A = v2 (zN)
    → TagToLayout_t → LayoutL1A = tla::Layout<Shape<M,K>, Stride<...>>
      → TensorL1A = tla::Tensor<LocalTensor<half>, LayoutL1A, Coord<0,0>, A1>
  → L0ALayoutSelector → LayoutTagL0A = zZ
    → TagToLayout_t → LayoutL0A = tla::Layout<Shape<...>, Stride<...>>
      → TensorL0A = tla::Tensor<LocalTensor<half>, LayoutL0A, Coord<0,0>, A2>
```

## 调用示例

### 基础调用（无 Bias、无 ReLU、无量化）

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::PackedTileCopyTla<
    Arch::AtlasA2,
    half, layout::RowMajor,
    half, layout::ColumnMajor,
    half, layout::RowMajor>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToGm copyL0CToGm;
```

### 完整调用（Bias + ReLU + per-tensor 量化）

```cpp
using TileCopy_ = Tile::PackedTileCopyTla<
    Arch::AtlasA2,
    int8_t, layout::RowMajor,
    int8_t, layout::ColumnMajor,
    int8_t, layout::RowMajor,
    half,                                    // ElementBias
    false,                                    // ReluEnable
    ScaleGranularity::PER_TENSOR>;

typename TileCopy_::CopyGmToL1A     copyGmToL1A;
typename TileCopy_::CopyGmToL1Bias  copyGmToL1Bias;
typename TileCopy_::CopyL1ToBT      copyL1ToBT;
typename TileCopy_::CopyL0CToGm     copyL0CToGm;     // per-tensor 量化写回
```
