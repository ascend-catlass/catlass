# TileCopyFAQTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp)

[TOC]

## 功能说明

`TileCopyFAQTla` 是 TLA 风格的 FlashAttention LoadQ 数据搬运模板。用于将多矩阵（multi-matrix）数据从 GM 搬运到 L1 并转换为 zN 分形格式，服务于 FlashAttention 的 Q 矩阵预加载阶段。源数据为 3D（ndNum*nValue*dValue），通过 `Nd2NzParams` 一次性转换为 zN 格式。

> **限制**：该模板仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`），Ascend 950 不支持。

## 模板原型

```cpp
template <
    class ArchTag,        // 架构标签
    class TensorSrc,      // 源 Tensor（GM，3D 多矩阵）
    class TensorDst       // 目标 Tensor（L1，zN）
>
struct TileCopyFAQTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyFAQTla, can not find the specialization.");
};
```

| 模板参数    | 说明                                                                         |
| :---------- | :--------------------------------------------------------------------------- |
| `ArchTag`   | 架构标签，仅支持 `Arch::AtlasA2`                                             |
| `TensorSrc` | 源 Tensor：`tla::Tensor<GlobalTensor<Element>, Layout, Coord, GM>`，3D shape |
| `TensorDst` | 目标 Tensor：`tla::Tensor<LocalTensor<Element>, Layout, Coord, A1>`，zN 格式 |

## 偏特化实现

### Atlas A2

| 源 Shape          | 目标 Layout | SFINAE 条件       | 说明                                 |
| :---------------- | :---------- | :---------------- | :----------------------------------- |
| 3D（ndNum, n, d） | zN          | `iszN<LayoutDst>` | Nd2Nz 多矩阵转换，大 stride 回退逐行 |

- `ndNum`：矩阵数量（对应 sequence length 维度）
- `nValue`：行维度
- `dValue`：列维度
- 当 `srcNdMatrixStride < STRIDE_LIMIT(65536)` 时一次性 Nd2Nz 搬运；否则逐矩阵逐行回退

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

- `dstTensor`：L1 上的目标 Tensor（zN 格式）
- `srcTensor`：GM 上的源 Tensor（3D 多矩阵，如 RowMajor 3D）

## 调用示例

### FlashAttention Q 矩阵加载（Atlas A2）

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t seqLen = 128;   // ndNum（矩阵数量）
const uint32_t headDim = 64;   // dValue（头维度）
const uint32_t numHeads = 32;  // nValue

// 3D 源 layout：RowMajor(seqLen, numHeads, headDim)
auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(seqLen, numHeads, headDim);
// L1 zN 目标 layout
auto dstLayout = tla::MakeLayout<half, layout::zN>(seqLen * numHeads, headDim);

AscendC::GlobalTensor<half> srcGmTensor;
AscendC::LocalTensor<half> dstL1Tensor;

auto srcTensor = tla::MakeTensor(srcGmTensor, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, dstLayout, Arch::PositionL1{});

TileCopyFAQTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> faqCopyOp;
faqCopyOp(dstTensor, srcTensor);
```
