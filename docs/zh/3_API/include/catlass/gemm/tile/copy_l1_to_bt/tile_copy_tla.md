# TileCopyTla（L1 → BT 偏特化）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_bt.hpp)（Ascend 950）

[TOC]

## 功能说明

`TileCopyTla` 是 TLA 风格的通用 tile 搬运模板。其在 `copy_l1_to_bt.hpp` 中定义的偏特化专门负责将 Bias Table（一维向量）从 L1（A1 Buffer）搬运到 BT（Bias Table Buffer，C2 Buffer）。

与 [非 TLA CopyL1ToBT](./copy_l1_to_bt.md) 不同，TLA 版本通过 `tla::Tensor` 封装操作数，由 TLA 运行时自动推导 Layout/Shape/Stride。

> **注意**：该偏特化仅支持 Ascend 950 架构（`CATLASS_ARCH == 3510`）。Atlas A2 没有 TLA 版本的 L1→BT 搬运。

## 模板原型

`TileCopyTla` 定义于 [tile_copy_tla.hpp](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)：

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopyTla;
```

L1 → BT 的偏特化通过 SFINAE 匹配：源 tensor 的 Layout 为 `VectorLayout`（`isVector` trait），目标 tensor 的 Layout 也为 `VectorLayout`，Position 分别为 `A1` 和 `C2`。

## 偏特化实现

### Ascend 950

| 源 Tensor             | 目标 Tensor           | SFINAE 条件                                  | 说明                                                              |
| :-------------------- | :-------------------- | :------------------------------------------- | :---------------------------------------------------------------- |
| VectorLayout L1（A1） | VectorLayout BT（C2） | `isVector<LayoutSrc> && isVector<LayoutDst>` | 一维向量拷贝，使用 `AscendC::DataCopy`，B32 类型自动对齐 blockLen |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor);
```

- `srcTensor`：L1 上的源 tensor（`tla::Tensor<LocalTensor, VectorLayout, Coord, A1>`）
- `dstTensor`：BT Buffer 上的目标 tensor（`tla::Tensor<LocalTensor, VectorLayout, Coord, C2>`）
- `srcTensor` 和 `dstTensor` 元素类型可以不同

## 调用示例

### Ascend 950，TLA

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;

const uint32_t vecLen = 256;

auto layoutSrc = tla::MakeLayout<ElementSrc, layout::VectorLayout>(1, vecLen);
auto layoutDst = tla::MakeLayout<ElementDst, layout::VectorLayout>(1, vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstBTTensor, layoutDst, Arch::PositionBias{});

TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

> **说明**：`tla::MakeLayout<Element, layout::VectorLayout>(1, vecLen)` 中第二个参数为向量长度。TLA 侧 Position 对应关系为：`Arch::PositionL1` → A1，`Arch::PositionBias` → C2。
