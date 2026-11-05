# SparseTileCopyTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`SparseTileCopyTla` 是稀疏 GEMM 专用的 TLA 搬运模板集合。所有搬运算子均为 TLA 风格，包括稀疏专用的 `CopyL1ToL0BSparseTla` 和 `CopyL0CToGmSparseTla`。

B 矩阵通过 `CopyGmToL1BIdx` 额外搬运 CSR/COO 格式的 index 数据（`int32_t`），L1→L0B 使用 `CopyL1ToL0BSparseTla` 结合 index 进行稀疏解压。

> **限制**：仅 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 引用的 Tile 组件

| 成员别名         | 引用的底层模板                                                                | 说明                      |
| :--------------- | :---------------------------------------------------------------------------- | :------------------------ |
| `CopyGmToL1A`    | `TileCopySparseTla<ArchTag, TensorA, TensorL1A>`                              | A 矩阵 GM→L1（TLA）       |
| `CopyGmToL1B`    | `TileCopySparseTla<ArchTag, TensorB, TensorL1B>`                              | B 矩阵 GM→L1（TLA）       |
| `CopyGmToL1BIdx` | `TileCopySparseTla<ArchTag, TensorIdx, TensorL1BIdx>`                         | B index GM→L1（TLA）      |
| `CopyL1ToL0A`    | `TileCopySparseTla<ArchTag, TensorL1A, TensorL0A>`                            | A 矩阵 L1→L0A（TLA）      |
| `CopyL1ToL0B`    | `CopyL1ToL0BSparseTla<ArchTag, ElementB, TensorL1B, TensorL0B, TensorL1BIdx>` | B 矩阵 L1→L0B（稀疏 TLA） |
| `CopyL0CToGm`    | `CopyL0CToGmSparseTla<ArchTag, TensorL0C, TensorC>`                           | L0C→GM（稀疏 TLA）        |

## 模板原型

```cpp
template <
    class ArchTag,            // 架构标签：Arch::AtlasA2
    class ElementA_,          // A 矩阵元素类型
    class LayoutTagA,         // A 矩阵 GM layout tag
    class ElementB_,          // B 矩阵元素类型
    class LayoutTagB,         // B 矩阵 GM layout tag
    class ElementC_,          // C 矩阵元素类型
    class LayoutTagC          // C 矩阵 GM layout tag
>
struct SparseTileCopyTla;
```

## 布局推导

```cpp
using LayoutTagL1A  = helper::L1ATypeSelector<GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
using LayoutTagL1B  = helper::L1BTypeSelector<GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
using LayoutTagL0A  = layout::zZ;
using LayoutTagL0B  = layout::nZ;
using LayoutTagL1BIdx = helper::L1BTypeSelector<GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
```

- A 侧：GM LayoutTag → L1 LayoutTag → L0A zZ
- B 侧：GM LayoutTag → L1 LayoutTag → L0B nZ
- B index：与 B 相同的 L1 LayoutTag，类型 `int32_t`

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::SparseTileCopyTla<
    Arch::AtlasA2,
    half, layout::RowMajor,          // A: half RowMajor
    half, layout::ColumnMajor,       // B: half ColumnMajor
    half, layout::RowMajor>;         // C: half RowMajor

typename TileCopy_::CopyGmToL1A     copyGmToL1A;
typename TileCopy_::CopyGmToL1B     copyGmToL1B;
typename TileCopy_::CopyGmToL1BIdx  copyGmToL1BIdx;   // 搬运 B 矩阵 index
typename TileCopy_::CopyL1ToL0A     copyL1ToL0A;
typename TileCopy_::CopyL1ToL0B     copyL1ToL0B;       // 稀疏解压
typename TileCopy_::CopyL0CToGm     copyL0CToGm;
```
