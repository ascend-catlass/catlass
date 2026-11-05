# TileCopyPerTokenDequant

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopyPerTokenDequant` 是 epilogue 阶段 per-token 反量化的搬运聚合模板。在 `TileCopy` 的基础上额外引入 `CopyPerTokenScale2Ub` 搬运 per-token scale 到 UB，供反量化计算使用。

- 适用范围：Atlas A2

## 模板原型

```cpp
template <
    class ArchTag,
    class CType,                // int32_t 累加结果（RowMajor）
    class ScaleType,            // per-channel scale（RowMajor）
    class PerTokenScaleType,    // per-token scale（ColumnMajor）
    class DType                 // 反量化目标（RowMajor）
>
struct TileCopyPerTokenDequant;
```

## 成员类型定义

| 成员类型                  | 说明                                            |
| :------------------------ | :---------------------------------------------- |
| `CopyGmToUbC`             | `CopyGm2Ub<Arch, CType>`                        |
| `CopyGmToUbScale`         | `CopyGm2Ub<Arch, ScaleType>`                    |
| `CopyGmToUbPerTokenScale` | `CopyPerTokenScale2Ub<Arch, PerTokenScaleType>` |
| `CopyUbToGmD`             | `CopyUb2Gm<Arch, DType>`                        |

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_copy.hpp"

using namespace Catlass::Epilogue::Tile;

using CType             = Gemm::GemmType<int32_t, layout::RowMajor>;
using ScaleType         = Gemm::GemmType<half, layout::RowMajor>;
using PerTokenScaleType = Gemm::GemmType<half, layout::ColumnMajor>;
using DType             = Gemm::GemmType<half, layout::RowMajor>;

using Copy = TileCopyPerTokenDequant<Arch::AtlasA2, CType, ScaleType, PerTokenScaleType, DType>;

// 成员:
// Copy::CopyGmToUbC            -> CopyGm2Ub<Arch, CType>
// Copy::CopyGmToUbScale        -> CopyGm2Ub<Arch, ScaleType>
// Copy::CopyGmToUbPerTokenScale -> CopyPerTokenScale2Ub<Arch, PerTokenScaleType>
// Copy::CopyUbToGmD            -> CopyUb2Gm<Arch, DType>
```
