# TileCopyFAQTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## 功能说明

`TileCopyFAQTla` 是 Flash Attention LoadQ 专用的 TLA 搬运模板，完成 GM（RowMajor）→ L1（zN）的多矩阵 DataCopy。搬运过程中 DataCopy 内部完成 ND→NZ 的 layout 转换。

适用场景：FlashAttention Q 矩阵加载。与通用 `TileCopyTla` 的区别在于固定转换路径 ND→NZ。

> **限制**：仅 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 基类声明

```cpp
template <
    class ArchTag,           // 架构标签
    class TensorSrc,         // 源 tensor（RowMajor GM）
    class TensorDst          // 目标 tensor（zN L1）
>
struct TileCopyFAQTla {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileCopyFAQTla, can not find the specialization.");
};
```

## 偏特化实现（全 Atlas A2）

| 方向                | 实现位置                    | API 文档                                               |
| :------------------ | :-------------------------- | :----------------------------------------------------- |
| GM RowMajor → L1 zN | `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_faq_tla.md) |

与常规 `TileCopyTla` GM→L1 的区别：DataCopy 参数 `col * sizeof(Element)` 替代 `col * sizeof(Element) / ELE_NUM_PER_BLK`，直接逐列拷贝而非 32B 对齐块拷贝。

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // L1 zN tensor
    TensorSrc const &srcTensor     // GM RowMajor tensor
);
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using Element = half;

// FA LoadQ: GM RowMajor → L1 zN
auto qGmLayout = tla::MakeLayout<Element, layout::RowMajor>(seq_len, head_dim);
auto qL1Layout = tla::MakeLayout<Element, layout::zN>(seq_len, head_dim);
auto qGmTensor = tla::MakeTensor(qGm, qGmLayout, Arch::PositionGM{});
auto qL1Tensor = tla::MakeTensor(qL1, qL1Layout, Arch::PositionL1{});

TileCopyFAQTla<Arch::AtlasA2, decltype(qGmTensor), decltype(qL1Tensor)> copyOp;
copyOp(qL1Tensor, qGmTensor);
```
