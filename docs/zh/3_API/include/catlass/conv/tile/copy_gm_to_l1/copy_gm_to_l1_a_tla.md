# CopyGmToL1ATla

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_gm_to_l1.hpp)

[TOC]

## 功能说明

`CopyGmToL1ATla` 实现 Conv 场景下 Fmap（特征图）从 GM 到 L1 的 TLA 风格搬运。对应 A 矩阵，布局为 `NC1HWC0` → `NC1HWC0`。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品
- 风格：TLA

## 模板原型

```cpp
template <class Element>
struct CopyGmToL1ATla;
```

| 模板参数  | 说明                |
| :-------- | :------------------ |
| `Element` | 元素类型，如 `half` |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // tla::Tensor<LocalTensor<Element>, NC1HWC0, ..., L1>
    TensorSrc const &srcTensor     // tla::Tensor<GlobalTensor<Element>, NC1HWC0, ..., GM>
)
```

## 调用示例

```cpp
#include "catlass/conv/tile/atlasa2/copy_gm_to_l1.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Hi = 28, Wi = 28, C0 = 16;

auto layoutSrc = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);
auto layoutDst = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);

AscendC::GlobalTensor<Element> srcData;
AscendC::LocalTensor<Element> dstData;

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL1{});

CopyGmToL1ATla<Element> copyOp;
copyOp(dstTensor, srcTensor);
```
