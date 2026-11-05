# TileCopyGemvAiv

> [代码位置](../../../../../../../../include/catlass/gemv/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopyGemvAiv` 为 AIV（AI Vector）芯片提供 GEMV 搬运子组件的聚合模板。数据通路为 GM↔UB↔GM，引用 GEMV 专有的搬运组件。

- 适用范围：Atlas A2
- 不直接执行算子，以类型成员方式暴露子组件引用

## 模板原型

```cpp
template <class ArchTag, class AType, class XType, class YType, class BiasType = void>
struct TileCopyGemvAiv;
```

| 模板参数   | 说明                                                  |
| :--------- | :---------------------------------------------------- |
| `ArchTag`  | 架构标签                                              |
| `AType`    | A 矩阵类型 `GemmType<ElementA, RowMajor/ColumnMajor>` |
| `XType`    | X 向量类型 `GemmType<ElementX, VectorLayout>`         |
| `YType`    | Y 向量类型 `GemmType<ElementY, VectorLayout>`         |
| `BiasType` | 偏置类型，默认 `void`                                 |

## 成员类型定义

| 成员类型           | 对应子组件                     | 说明                             |
| :----------------- | :----------------------------- | :------------------------------- |
| `VecCopyGmToUb`    | `Gemv::Tile::VecCopyGmToUB`    | 向量 X: GM→UB                    |
| `VecCopyUbToGm`    | `Gemv::Tile::VecCopyUBToGm`    | 向量 Y: UB→GM（可选 atomic add） |
| `MatrixCopyGmToUb` | `Gemv::Tile::MatrixCopyGmToUB` | 矩阵 A: GM→UB                    |

## 调用示例

```cpp
#include "catlass/gemv/tile/tile_copy.hpp"

using namespace Catlass::Gemv::Tile;

using ElementA = half;
using ElementX = half;
using ElementY = half;

using AType = Gemm::GemmType<ElementA, layout::RowMajor>;
using XType = Gemm::GemmType<ElementX, layout::VectorLayout>;
using YType = Gemm::GemmType<ElementY, layout::VectorLayout>;

using Copy = TileCopyGemvAiv<Arch::AtlasA2, AType, XType, YType>;

// 子组件:
// typename Copy::VecCopyGmToUb
// typename Copy::VecCopyUbToGm
// typename Copy::MatrixCopyGmToUb
```
