# CopyL1ToL0BSparseTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp)

[TOC]

## 功能说明

`CopyL1ToL0BSparseTla` 是 Atlas A2 架构专用的 TLA 风格 sparse L1→L0B 搬运模板。在 Sparse GEMM 场景中，B 矩阵以压缩格式存储（仅存储非零元素），搬运时需要同时传入 index tensor 来指示哪些位置的元素有效。

与普通 [TileCopyTla](./tile_copy_tla.md) 不同，`CopyL1ToL0BSparseTla` 的 `operator()` 接受三个 tensor 参数：src、dst、index，并通过 `AscendC::LoadDataWithSparse` 指令完成带稀疏索引的搬运。

> **限制**：该模板仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`），Ascend 950 不支持。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：Arch::AtlasA2
    class ElementA,                   // B 矩阵元素类型
    class TensorSrc,                  // 源 tensor：tla::Tensor<LocalTensor<Element>, Layout, Coord, A1>
    class TensorDst,                  // 目标 tensor：tla::Tensor<LocalTensor<Element>, Layout, Coord, B2>
    class TensorIdx,                  // Index tensor：tla::Tensor<LocalTensor<uint8_t>, Layout, Coord, A1>
    class Enable = void               // SFINAE 条件
>
struct CopyL1ToL0BSparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0b sparse, can not find the specialization.");
};
```

- `ArchTag`：架构标签，仅支持 `Arch::AtlasA2`
- `ElementA`：B 矩阵元素类型
- `TensorSrc`：L1 上的源 tensor
- `TensorDst`：L0B 上的目标 tensor
- `TensorIdx`：L1 上的 sparse index tensor（元素类型 `uint8_t`）

## 偏特化实现

### Atlas A2

| 源 Tensor | 目标 Tensor | Index Tensor      | SFINAE 条件                                             | 说明                             |
| :-------- | :---------- | :---------------- | :------------------------------------------------------ | :------------------------------- |
| zN L1     | nZ L0B      | zN L1 (`uint8_t`) | `iszN<LayoutSrc> && isnZ<LayoutDst> && iszN<LayoutIdx>` | Sparse 转置拷贝                  |
| nZ L1     | nZ L0B      | nZ L1 (`uint8_t`) | `isnZ<LayoutSrc> && isnZ<LayoutDst> && isnZ<LayoutIdx>` | Sparse 非转置拷贝（Transpose B） |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc, class TensorIdx>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, TensorIdx const &idxTensor);
```

- `dstTensor`：L0B 上的目标 tensor（`tla::Tensor<LocalTensor, Layout, Coord, B2>`）
- `srcTensor`：L1 上的压缩 B 矩阵源 tensor（`tla::Tensor<LocalTensor, Layout, Coord, A1>`）
- `idxTensor`：L1 上的 sparse index tensor，元素类型 `uint8_t`，layout 与 src 相同（`zN` 或 `nZ`）

Index tensor 中每个 `uint8_t` 元素的高 4 位和低 4 位分别表示两个相邻元素的索引信息，通过 `INDEX_SHIFT = 2` 偏移对齐。

## 调用示例

### zN → nZ Sparse 转置搬运（Atlas A2）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t K = 256;
const uint32_t N = 256;

// B 矩阵数据 layout（L1 zN）
auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

// Sparse index layout（L1 zN，元素类型 uint8_t）
auto layoutIdx = tla::MakeLayout<uint8_t, layout::zN>(K, N);

AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0BTensor;
AscendC::LocalTensor<uint8_t> idxL1Tensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});
auto idxTensor = tla::MakeTensor(idxL1Tensor, layoutIdx, Arch::PositionL1{});

// 实例化并调用
CopyL1ToL0BSparseTla<Arch::AtlasA2, half, decltype(srcTensor), decltype(dstTensor), decltype(idxTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor, idxTensor);
```

### nZ → nZ Sparse 非转置搬运（Atlas A2，Transpose B）

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutIdx = tla::MakeLayout<uint8_t, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});
auto idxTensor = tla::MakeTensor(idxL1Tensor, layoutIdx, Arch::PositionL1{});

CopyL1ToL0BSparseTla<Arch::AtlasA2, half, decltype(srcTensor), decltype(dstTensor), decltype(idxTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor, idxTensor);
```
