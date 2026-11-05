# TileCopyWithPrologueDeqPerTensor

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopyWithPrologueDeqPerTensor` 是带 Prologue 反量化（per-tensor）的搬运模板集合，在 [TileCopy](./tile_copy.md) 基础上增加 `PrologueA` / `PrologueB` 算子类型，并将 `CopyL0CToGm` 固定为 per-tensor 量化。

适用于量化推理场景：GM 上存储量化 weight，Prologue 中完成反量化，L0C→GM 写回时进行 per-tensor 量化。

> **限制**：仅 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 引用的 Tile 组件

| 成员别名         | 引用的底层模板                                                | 说明                 |
| :--------------- | :------------------------------------------------------------ | :------------------- |
| `CopyGmToL1A`    | `CopyGmToL1<ArchTag, AType>`                                  | A 矩阵 GM→L1         |
| `CopyGmToL1B`    | `CopyGmToL1<ArchTag, BType>`                                  | B 矩阵 GM→L1         |
| `CopyL1ToL0A`    | `CopyL1ToL0A<ArchTag, L1AType>`                               | A 矩阵 L1→L0A        |
| `CopyL1ToL0B`    | `CopyL1ToL0B<ArchTag, L1BType>`                               | B 矩阵 L1→L0B        |
| `CopyL0CToGm`    | `CopyL0CToGm<ArchTag, ElementAccumulator, CType, PER_TENSOR>` | L0C→GM（per-tensor） |
| `CopyGmToL1Bias` | `CopyGmToL1<ArchTag, ...>` 或 `void`                          | Bias GM→L1           |
| `CopyL1ToBT`     | `CopyL1ToBT<ArchTag, ...>` 或 `void`                          | Bias L1→BT           |

## 模板原型

```cpp
template <
    class ArchTag,            // 架构标签：Arch::AtlasA2
    class AType,              // A 矩阵 GmType
    class BType,              // B 矩阵 GmType
    class CType,              // C 矩阵 GmType
    class PrologueA_,         // A 矩阵 Prologue 算子类型
    class PrologueB_,         // B 矩阵 Prologue 算子类型
    class BiasType = void     // Bias GmType（可选）
>
struct TileCopyWithPrologueDeqPerTensor;
```

## 成员类型推导

```cpp
using ElementA = typename AType::Element;
using ElementB = typename BType::Element;
using ElementAccumulator =
    typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

using CopyGmToL1A = CopyGmToL1<ArchTag, AType>;
using CopyGmToL1B = CopyGmToL1<ArchTag, BType>;

using PrologueA = PrologueA_;
using PrologueB = PrologueB_;

using CopyL1ToL0A = CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
using CopyL1ToL0B = CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
using CopyL0CToGm = CopyL0CToGm<ArchTag, ElementAccumulator, CType, ScaleGranularity::PER_TENSOR>;
using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType, void>, void, ...>;
using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType, void>, void, ...>;
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
using BType = Gemm::GemmType<int8_t, layout::ColumnMajor>;
using CType = Gemm::GemmType<int8_t, layout::RowMajor>;

using PrologueA = Tile::TileCastInt8ToFp16Dequant<Arch::AtlasA2, ...>;
using PrologueB = Tile::TileCastInt8ToFp16Dequant<Arch::AtlasA2, ...>;

using TileCopy_ = Tile::TileCopyWithPrologueDeqPerTensor<
    Arch::AtlasA2, AType, BType, CType, PrologueA, PrologueB>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToGm copyL0CToGm;  // per-tensor 量化
```
