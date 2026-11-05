# CopyL1ToL0BSparseTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## 功能说明

`CopyL1ToL0BSparseTla` 是稀疏 GEMM 中 B 矩阵 L1→L0B 搬运的 TLA 模板，需要额外的 `TensorIdx`（稀疏 index tensor，`int32_t`）辅助稀疏解压。与普通 [CopyL1ToL0B](../copy_l1_to_l0b/tile_copy_tla.md) 不同，它需要通过 index 将稠密 B 矩阵在搬入 L0B 时解压为对应的 block。

> **限制**：仅 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 基类声明

```cpp
template <
    class ArchTag,            // 架构标签
    class Element,            // A 矩阵元素类型（影响 L0B column）
    class TensorSrc,          // B 矩阵 L1 tensor
    class TensorDst,          // B 矩阵 L0B tensor
    class TensorIdx,          // 稀疏 index tensor
    class Enable = void       // SFINAE enable
>
struct CopyL1ToL0BSparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported CopyL1ToL0BSparseTla, can not find the specialization.");
};
```

## 偏特化实现（全 Atlas A2）

| 条件             | 说明                            | 实现位置                     | API 文档                                                         |
| :--------------- | :------------------------------ | :--------------------------- | :--------------------------------------------------------------- |
| `isSparseEnabled` | B matrix ColumnMajor→nZ + index | `atlasa2/copy_l1_to_l0b.hpp` | [copy_l1_to_l0b](../copy_l1_to_l0b/copy_l1_to_l0b_sparse_tla.md) |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc, class TensorIdx>
void operator()(
    TensorDst const &l0BTensor,    // L0B 目标
    TensorSrc const &l1BTensor,    // L1 B 源
    TensorIdx const &l1BIdxTensor  // L1 B index
);
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using ElementA = half;
using ElementB = half;

// B 矩阵 ColumnMajor L1 → nZ L0B
auto l1bLayout = tla::MakeLayout<ElementB, layout::ColumnMajor>(K1, N1);
auto l0bLayout = tla::MakeLayout<ElementB, layout::nZ>(K1, N1);
auto l1bTensor = tla::MakeTensor(l1bData, l1bLayout, Arch::PositionL1{});
auto l0bTensor = tla::MakeTensor(l0bData, l0bLayout, Arch::PositionL0B{});

// Index tensor (int32_t, same L1 layout as B)
auto idxLayout = tla::MakeLayout<int32_t, layout::ColumnMajor>(K1, N1);
auto idxTensor = tla::MakeTensor(idxData, idxLayout, Arch::PositionL1{});

CopyL1ToL0BSparseTla<Arch::AtlasA2, ElementA,
    decltype(l1bTensor), decltype(l0bTensor), decltype(idxTensor)> copyOp;
copyOp(l0bTensor, l1bTensor, idxTensor);
```
