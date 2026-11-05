# TileCopyTla（GM → UB）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_ub.hpp)

[TOC]

## 功能说明

`TileCopyTla` 的 GM→UB 偏特化负责将 RowMajor 二维矩阵数据从 GM 搬运到 UB（Unified Buffer，`VECCALC`），供 Vector 引擎访问。与 [CopyGm2Ub](./copy_gm_to_ub.md) 的 VectorLayout 不同，TLA 版本支持 RowMajor 二维排布，使用 `AscendC::DataCopyPad` 逐行完成搬运。

> **限制**：仅支持 Atlas A2 架构，源和目标 layout 必须均为 RowMajor。

## 模板原型

```cpp
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value &&
                     tla::detail::isRowMajor<LayoutDst>::value>>;
```

## 偏特化实现

| 架构    | 源位置 | 目标位置 | Layout 要求         | 搬运指令               |
| :------ | :----- | :------- | :------------------ | :--------------------- |
| Atlas A2 | GM     | VECCALC  | RowMajor → RowMajor | `AscendC::DataCopyPad` |

逐行拷贝，每行的长度为 `col * sizeof(ElementSrc)` 字节，行数为 `row`，源 stride 为目标 stride 的字节偏移。

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // 目标 tensor（UB, VECCALC, RowMajor）
    TensorSrc const &srcTensor     // 源 tensor（GM, RowMajor）
);
```

静态约束：

- `TensorSrc::position == GM`，`TensorSrc::Layout` 为 RowMajor
- `TensorDst::position == VECCALC`，`TensorDst::Layout` 为 RowMajor

## 调用示例

```cpp
#include "catlass/gemm/tile/copy_gm_to_ub.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using ElementSrc = half;
using ElementDst = half;

const int M = 128;
const int K = 256;

auto srcLayout = tla::MakeLayout<ElementSrc, layout::RowMajor>(M, K);
auto dstLayout = tla::MakeLayout<ElementDst, layout::RowMajor>(M, K);

auto srcTensor = tla::MakeTensor(srcGmTensor, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstUBTensor, dstLayout, Arch::PositionUB{});

TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
