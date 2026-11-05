# tile_copy（Gemv）

> [代码位置](../../../../../../../../include/catlass/gemv/tile/tile_copy.hpp)

[TOC]

## 概述

Gemv `tile_copy` 是搬运聚合模板，根据芯片类型分为 AIV（GM↔UB）和 AIC（GM→L1→L0→GM）两套聚合。

## API 清单

| API                                        | 芯片类型 | 适用硬件  | 数据通路    | 说明                 |
| :----------------------------------------- | :------- | :-------- | :---------- | :------------------- |
| [TileCopyGemvAiv](./tile_copy_gemv_aiv.md) | AIV      | Atlas A2   | GM↔UB       | VecCopy + MatrixCopy |
| [TileCopyGemvAic](./tile_copy_gemv_aic.md) | AIC      | Ascend 950 | GM→L1→L0→GM | 复用 GEMM 搬运组件   |

## 调用示例

### TileCopyGemvAiv

```cpp
#include "catlass/gemv/tile/tile_copy.hpp"

using namespace Catlass::Gemv::Tile;

using ElementA = half;
using AType = Gemm::GemmType<ElementA, layout::RowMajor>;
using XType = Gemm::GemmType<ElementA, layout::VectorLayout>;
using YType = Gemm::GemmType<ElementA, layout::VectorLayout>;

using Copy = TileCopyGemvAiv<Arch::AtlasA2, AType, XType, YType>;
```

### TileCopyGemvAic

```cpp
using Copy = TileCopyGemvAic<Arch::Ascend950, AType, XType, YType>;
```
