# TileCopyGemvAic

> [代码位置](../../../../../../../../include/catlass/gemv/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopyGemvAic` 为 AIC（AI Core）芯片提供 GEMV 搬运子组件的聚合模板。数据通路为 GM→L1→L0A/L0B→L0C→GM，复用 GEMM 的搬运组件。中间 L1/L0A/L0B Type 通过 `Gemv::helper::L1AndL0TypeSelectorGemv` 自动推导。

- 适用范围：Ascend 950
- 不直接执行算子，以类型成员方式暴露子组件引用

## 模板原型

```cpp
template <class ArchTag, class AType, class XType, class YType, class BiasType = void>
struct TileCopyGemvAic;
```

## 成员类型定义

| 成员类型      | 对应子组件                                                 | 说明          |
| :------------ | :--------------------------------------------------------- | :------------ |
| `CopyGmToL1A` | `Gemm::Tile::CopyGmToL1<Arch, XType, L1XType>`             | 向量 X: GM→L1 |
| `CopyGmToL1B` | `Gemm::Tile::CopyGmToL1<Arch, AType, L1AType>`             | 矩阵 A: GM→L1 |
| `CopyL1ToL0A` | `Gemm::Tile::CopyL1ToL0A<Arch, L1XType, L0AType>`          | L1→L0A        |
| `CopyL1ToL0B` | `Gemm::Tile::CopyL1ToL0B<Arch, L1AType, L0BType>`          | L1→L0B        |
| `CopyL0CToGm` | `Gemm::Tile::CopyL0CToGm<Arch, ElementAccumulator, YType>` | L0C→GM        |

## 调用示例

```cpp
#include "catlass/gemv/tile/tile_copy.hpp"

using namespace Catlass::Gemv::Tile;

using ElementA = half;
using AType = Gemm::GemmType<ElementA, layout::RowMajor>;
using XType = Gemm::GemmType<ElementA, layout::VectorLayout>;
using YType = Gemm::GemmType<ElementA, layout::VectorLayout>;

using Copy = TileCopyGemvAic<Arch::Ascend950, AType, XType, YType>;

// 子组件:
// typename Copy::CopyGmToL1A
// typename Copy::CopyGmToL1B
// typename Copy::CopyL1ToL0A
// typename Copy::CopyL1ToL0B
// typename Copy::CopyL0CToGm
```
