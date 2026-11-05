# TileCopySparseTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## 功能说明

`TileCopySparseTla` 是稀疏 GEMM 专用的 TLA 搬运模板，与 [TileCopyTla](./tile_copy_tla.md) 的核心区别在于源/目标 Tensor 可能为不同的元素类型（如 B index 数据 `int32_t`）。结构与 `TileCopyTla` 类似，通过 SFINAE 自动分发。

> **限制**：仅 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 基类声明

```cpp
template <
    class ArchTag,           // 架构标签
    class TensorSrc,         // 源 tensor 类型
    class TensorDst,         // 目标 tensor 类型
    class Enable = void      // SFINAE enable
>
struct TileCopySparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileCopySparseTla, can not find the specialization.");
};
```

## 偏特化实现清单（全 Atlas A2）

| 方向   | 说明                                        | 实现位置                     | API 文档                                                    |
| :----- | :------------------------------------------ | :--------------------------- | :---------------------------------------------------------- |
| GM→L1A | 稀疏 A 矩阵 GM RowMajor → L1 RowMajor       | `atlasa2/copy_gm_to_l1.hpp`  | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_sparse_tla.md)   |
| GM→L1B | 稀疏 B 矩阵 GM ColumnMajor → L1 ColumnMajor | `atlasa2/copy_gm_to_l1.hpp`  | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_sparse_tla.md)   |
| L1→L0A | 稀疏 A 矩阵 L1→L0A zZ                       | `atlasa2/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_sparse_tla.md) |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor
);
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

// B 矩阵 index（int32_t） GM→L1
auto idxGmLayout = tla::MakeLayout<int32_t, layout::ColumnMajor>(K, N);
auto idxL1Layout = tla::MakeLayout<int32_t, layout::ColumnMajor>(K, N);
auto idxGmTensor = tla::MakeTensor(idxGm, idxGmLayout, Arch::PositionGM{});
auto idxL1Tensor = tla::MakeTensor(idxL1, idxL1Layout, Arch::PositionL1{});

TileCopySparseTla<Arch::AtlasA2, decltype(idxGmTensor), decltype(idxL1Tensor)> copyOp;
copyOp(idxL1Tensor, idxGmTensor);
```
