# TileCopy

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopy` 是 epilogue 搬运聚合的基础模板，组合引用 `CopyGm2Ub`、`CopyUb2Gm` 等基础搬运模板供 block 层 epilogue 使用。根据 operands 数量（C/D、C/X/D、C/X/Y/D）自动组装所需的 GM→UB 和 UB→GM 子组件。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品
- 不直接执行算子，以类型成员方式暴露子组件引用

## 模板原型

```cpp
// 2 个 Operand：C + D
template <class ArchTag, class CType, class DType>
struct TileCopy;

// 3 个 Operand：C + X + D
template <class ArchTag, class CType, class XType, class DType>
struct TileCopy;

// 4 个 Operand：C + X + Y + D
template <class ArchTag, class CType, class XType, class YType, class DType>
struct TileCopy;
```

## 成员类型定义

| 模板                         | 成员类型                                      | 说明                     |
| :--------------------------- | :-------------------------------------------- | :----------------------- |
| `TileCopy<Arch, C, D>`       | `CopyGmToUbC`                                 | `CopyGm2Ub<Arch, CType>` |
|                              | `CopyUbToGmD`                                 | `CopyUb2Gm<Arch, DType>` |
| `TileCopy<Arch, C, X, D>`    | `CopyGmToUbC`                                 | `CopyGm2Ub<Arch, CType>` |
|                              | `CopyGmToUbX`                                 | `CopyGm2Ub<Arch, XType>` |
|                              | `CopyUbToGmD`                                 | `CopyUb2Gm<Arch, DType>` |
| `TileCopy<Arch, C, X, Y, D>` | `CopyGmToUbC` / `CopyGmToUbX` / `CopyGmToUbY` | `CopyGm2Ub<...>`         |
|                              | `CopyUbToGmD`                                 | `CopyUb2Gm<Arch, DType>` |

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_copy.hpp"

using namespace Catlass::Epilogue::Tile;

using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
using DType = Gemm::GemmType<half, layout::RowMajor>;

using Copy = TileCopy<Arch::AtlasA2, CType, DType>;
using CopyC = typename Copy::CopyGmToUbC;
using CopyD = typename Copy::CopyUbToGmD;

// block 层: CopyC 用于搬运 C 到 UB，CopyD 用于搬运结果回 GM
```
