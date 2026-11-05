# TileCopySparseTla（GM → L1，Sparse）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp)

[TOC]

## 功能说明

`TileCopySparseTla` 是 TLA 风格的 Sparse GEMM GM→L1 数据搬运模板。在 Sparse GEMM 场景中，A 矩阵以压缩格式存储（仅存储非零元素），`TileCopySparseTla` 负责将稀疏化的 A 矩阵从 GM 搬运到 L1，并将指定行优先/列优先的源数据转换为 zN 或 nZ 分形格式。

> **限制**：该模板仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`），Ascend 950 不支持。

## 模板原型

```cpp
template <
    class ArchTag,        // 架构标签
    class TensorSrc,      // 源 Tensor（GM）
    class TensorDst,      // 目标 Tensor（L1）
    class Enable = void   // SFINAE 分发
>
struct TileCopySparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopySparseTla, can not find the specialization.");
};
```

| 模板参数    | 说明                                                                |
| :---------- | :------------------------------------------------------------------ |
| `ArchTag`   | 架构标签，仅支持 `Arch::AtlasA2`                                    |
| `TensorSrc` | 源 Tensor：`tla::Tensor<GlobalTensor<Element>, Layout, Coord, GM>`  |
| `TensorDst` | 目标 Tensor：`tla::Tensor<LocalTensor<Element>, Layout, Coord, A1>` |
| `Enable`    | SFINAE 条件，根据 Layout 自动派发偏特化                             |

## 偏特化实现

### Atlas A2

| 源 Layout              | 目标 Layout | SFINAE 条件                               | 说明                             |
| :--------------------- | :---------- | :---------------------------------------- | :------------------------------- |
| RowMajor / ColumnMajor | zN          | `(isRowMajor \|\| isColumnMajor) && iszN` | ND→zN 转换，通过 `Nd2NzParams`   |
| ColumnMajor            | nZ          | `isColumnMajor && isnZ`                   | ND→nZ 转换，含大 stride 逐行回退 |
| zN（uint32_t 压缩）    | zN          | `iszN<uint32_t> && iszN`                  | zN→zN 直传，处理 16 对齐         |
| nZ（uint32_t 压缩）    | nZ          | `isnZ<uint32_t> && isnZ`                  | nZ→nZ 直传                       |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

- `dstTensor`：L1 上的目标 Tensor（zN 或 nZ 格式）
- `srcTensor`：GM 上的源 Tensor（RowMajor / ColumnMajor / zN / nZ）

## 调用示例

### RowMajor GM → zN L1（Atlas A2）

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto dstLayout = tla::MakeLayout<half, layout::zN>(M, K);

AscendC::GlobalTensor<half> srcGmTensor;
AscendC::LocalTensor<half> dstL1Tensor;

auto srcTensor = tla::MakeTensor(srcGmTensor, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, dstLayout, Arch::PositionL1{});

TileCopySparseTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor);
```

### ColumnMajor GM → nZ L1（Atlas A2）

```cpp
auto srcLayout = tla::MakeLayout<half, layout::ColumnMajor>(M, K);
auto dstLayout = tla::MakeLayout<half, layout::nZ>(M, K);

auto srcTensor = tla::MakeTensor(srcGmTensor, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, dstLayout, Arch::PositionL1{});

TileCopySparseTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor);
```
