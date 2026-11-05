# TileCopyTla（UB → GM）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_ub_to_gm.hpp)

[TOC]

## 功能说明

`TileCopyTla` 的 UB→GM 偏特化负责将 RowMajor 二维矩阵数据从 UB（`VECCALC`）搬运到 GM，使用 `AscendC::DataCopyPad` 逐行搬出。

与 [TileCopyTlaExt](./tile_copy_tla_ext.md) 的 PaddingRowMajor 目标排布不同，本模板的目标 Layout 为普通 RowMajor。

> **限制**：仅支持 Atlas A2 架构，源和目标 layout 必须均为 RowMajor。

## 模板原型

```cpp
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::GM>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value &&
                     tla::detail::isRowMajor<LayoutDst>::value>>;
```

## 偏特化实现

| 架构    | 源位置  | 目标位置 | Layout 要求         | 搬运指令               |
| :------ | :------ | :------- | :------------------ | :--------------------- |
| Atlas A2 | VECCALC | GM       | RowMajor → RowMajor | `AscendC::DataCopyPad` |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // 目标 tensor（GM, RowMajor）
    TensorSrc const &srcTensor     // 源 tensor（UB, VECCALC, RowMajor）
);
```

静态约束：

- `TensorSrc::position == VECCALC`，`TensorSrc::Layout` 为 RowMajor
- `TensorDst::position == GM`，`TensorDst::Layout` 为 RowMajor

## 调用示例

```cpp
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using Element = half;

const int M = 128;
const int N = 256;

auto srcLayout = tla::MakeLayout<Element, layout::RowMajor>(M, N);
auto dstLayout = tla::MakeLayout<Element, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcUBTensor, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
