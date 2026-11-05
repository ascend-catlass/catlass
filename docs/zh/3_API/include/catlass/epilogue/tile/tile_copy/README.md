# tile_copy（Epilogue）

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## 概述

Epilogue `tile_copy` 是搬运聚合模板，组合引用 `CopyGm2Ub`、`CopyUb2Gm` 等基础搬运模板，以类型成员方式暴露子组件供 block 层 epilogue 使用。不直接执行算子，通过 `GemmType` 参数自动推导所有中间 Layout 和子组件引用。

## API 清单

| API                                                         | 风格   | 适用硬件           | 说明                    |
| :---------------------------------------------------------- | :----- | :----------------- | :---------------------- |
| [TileCopy](./tile_copy.md)                                  | 非 TLA | Atlas A2, Ascend 950 | 基础聚合，2/3/4 Operand |
| [TileCopyBf16](./tile_copy_bf16.md)                         | 非 TLA | Atlas A2, Ascend 950 | BF16 强制类型特化       |
| [TileCopyPerTokenDequant](./tile_copy_per_token_dequant.md) | 非 TLA | Atlas A2            | Per-token 反量化聚合    |
| [TileCopyW4A4Gemm](./tile_copy_w4a4_gemm.md)                | 非 TLA | Atlas A2            | W4A4 GEMM 反量化聚合    |
| [TileCopyDequantTla](./tile_copy_dequant_tla.md)            | TLA    | Atlas A2, Ascend 950 | TLA dequant 聚合        |

## 调用示例

### TileCopy

```cpp
#include "catlass/epilogue/tile/tile_copy.hpp"

using namespace Catlass::Epilogue::Tile;

using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
using DType = Gemm::GemmType<half, layout::RowMajor>;

using Copy = TileCopy<Arch::AtlasA2, CType, DType>;
using CopyC = typename Copy::CopyGmToUbC;
using CopyD = typename Copy::CopyUbToGmD;
```

### TileCopyPerTokenDequant

```cpp
using CType             = Gemm::GemmType<int32_t, layout::RowMajor>;
using ScaleType         = Gemm::GemmType<half, layout::RowMajor>;
using PerTokenScaleType = Gemm::GemmType<half, layout::ColumnMajor>;
using DType             = Gemm::GemmType<half, layout::RowMajor>;

using Copy = TileCopyPerTokenDequant<Arch::AtlasA2, CType, ScaleType, PerTokenScaleType, DType>;
```
