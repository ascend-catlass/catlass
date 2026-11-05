# TileCopyWithPrologue

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopyWithPrologue` 是带 Prologue 算子的搬运模板集合，在 [TileCopy](./tile_copy.md) 基础上增加 `PrologueA` / `PrologueB` 算子类型。与 [TileCopyWithPrologueDeqPerTensor](./tile_copy_prologue_deq_per_tensor.md) 不同，`CopyL0CToGm` 保持默认 `NO_QUANT`。

适用于需要 Prologue 预处理（如 INT4→INT8 转换）但不需要随路量化的场景。

> **限制**：仅 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 引用的 Tile 组件

| 成员别名         | 引用的底层模板                                              | 说明             |
| :--------------- | :---------------------------------------------------------- | :--------------- |
| `CopyGmToL1A`    | `CopyGmToL1<ArchTag, AType>`                                | A 矩阵 GM→L1     |
| `CopyGmToL1B`    | `CopyGmToL1<ArchTag, BType>`                                | B 矩阵 GM→L1     |
| `CopyL1ToL0A`    | `CopyL1ToL0A<ArchTag, L1AType>`                             | A 矩阵 L1→L0A    |
| `CopyL1ToL0B`    | `CopyL1ToL0B<ArchTag, L1BType>`                             | B 矩阵 L1→L0B    |
| `CopyL0CToGm`    | `CopyL0CToGm<ArchTag, ElementAccumulator, CType, NO_QUANT>` | L0C→GM（非量化） |
| `CopyGmToL1Bias` | `CopyGmToL1<ArchTag, ...>` 或 `void`                        | Bias GM→L1       |
| `CopyL1ToBT`     | `CopyL1ToBT<ArchTag, ...>` 或 `void`                        | Bias L1→BT       |

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
struct TileCopyWithPrologue;
```

## 成员类型推导

与 `TileCopy` 基本一致，额外增加：

```cpp
using PrologueA = PrologueA_;
using PrologueB = PrologueB_;
```

## 与 TileCopyWithPrologueDeqPerTensor 的区别

| 模板                               | CopyL0CToGm 量化粒度 | 适用场景                              |
| :--------------------------------- | :------------------- | :------------------------------------ |
| `TileCopyWithPrologue`             | `NO_QUANT`（默认）   | Prologue 预处理，无需量化写回         |
| `TileCopyWithPrologueDeqPerTensor` | `PER_TENSOR`         | Prologue 反量化 + per-tensor 量化写回 |

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
using BType = Gemm::GemmType<int8_t, layout::ColumnMajor>;
using CType = Gemm::GemmType<half, layout::RowMajor>;

using PrologueB = Tile::TileCastInt4ToInt8<Arch::AtlasA2, ...>;

using TileCopy_ = Tile::TileCopyWithPrologue<
    Arch::AtlasA2, AType, BType, CType, PrologueA, PrologueB>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToGm copyL0CToGm;
```
