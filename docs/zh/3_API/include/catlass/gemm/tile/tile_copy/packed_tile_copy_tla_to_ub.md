# PackedTileCopyTlaToUB

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`PackedTileCopyTlaToUB` 继承自 [PackedTileCopyTla](./packed_tile_copy_tla.md)，**重写** `CopyL0CToDst` 为 `CopyL0CToUBTla`，将累加结果搬运到 UB（统一缓冲区）而非 GM。支持 UB 分拆模式（`NO_SPLIT` / `SPLIT`）。

适用于 Ascend 950 架构下 L0C 结果需经过 Vector 引擎后处理的场景。

> **限制**：仅 Ascend 950 架构（`CATLASS_ARCH == 3510`）。

## 引用的 Tile 组件

| 成员别名                     | 来源                                                                                     |
| :--------------------------- | :--------------------------------------------------------------------------------------- |
| `CopyGmToL1A` ~ `CopyL1ToBT` | 继承自 `PackedTileCopyTla<...>`                                                          |
| `CopyL0CToDst`（重写）       | `CopyL0CToUBTla<ArchTag, TensorL0C, TensorC, CopyMode, DEQUANT_GRANULARITY, ReluEnable>` |

## 模板原型

```cpp
template <
    class ArchTag,                                                   // 架构标签：Arch::Ascend950
    class ElementA_,                                                 // A 矩阵元素类型
    class LayoutTagA,                                                // A 矩阵 GM layout tag
    class ElementB_,                                                 // B 矩阵元素类型
    class LayoutTagB,                                                // B 矩阵 GM layout tag
    class ElementC_,                                                 // C 矩阵元素类型
    class LayoutTagC,                                                // C 矩阵 GM layout tag
    class ElementBias = void,                                        // Bias 元素类型
    CopyL0CToUBMode CopyMode_ = CopyL0CToUBMode::NO_SPLIT,           // UB 分拆模式
    bool ReluEnable = false,                                         // ReLU 开关
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT // 反量化粒度
>
struct PackedTileCopyTlaToUB : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ...>;
```

## 重写的成员

```cpp
template <class TensorC>
using CopyL0CToDst = CopyL0CToUBTla<ArchTag, TensorL0C, TensorC, CopyMode,
    DEQUANT_GRANULARITY, ReluEnable>;
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::PackedTileCopyTlaToUB<
    Arch::Ascend950,
    half, layout::RowMajor,
    half, layout::ColumnMajor,
    half, layout::RowMajor,
    void,
    CopyL0CToUBMode::SPLIT,      // UB 分拆模式
    false,
    ScaleGranularity::PER_TENSOR>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToDst copyL0CToUB;  // → UB，非 GM
```
