# CopyL1ToL0ATla

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0a.hpp)

[TOC]

## 功能说明

`CopyL1ToL0ATla` 实现 Conv 场景下将 Fmap 数据从 L1 搬运到 L0A 的 TLA 风格版本，同时完成 im2col 操作。

- Atlas A2：`LoadData` + `Conv2dFilterParams`
- Ascend 950：`LoadDataWithStride` + `SetLoadDataRepeatWithStride`

## 模板原型

```cpp
template <class Element>
struct CopyL1ToL0ATla;
```

| 模板参数  | 说明                |
| :-------- | :------------------ |
| `Element` | 元素类型，如 `half` |

构造函数接收 `Conv2dFilterParams` 参数。

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor,
    uint8_t *blockPadList
)
```

## 调用示例

```cpp
#include "catlass/conv/tile/atlasa2/copy_l1_to_l0a.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Hi = 28, Wi = 28, C0 = 16;
constexpr uint32_t Kh = 3, Kw = 3;

auto layoutSrc = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);
auto layoutDst = tla::MakeLayout<Element, layout::zZ>(16, 27);

Conv2dFilterParams params{.strideW_ = 1, .strideH_ = 1, .kw_ = Kw, .kh_ = Kh, .dilationW_ = 1, .dilationH_ = 1};

AscendC::LocalTensor<Element> srcData, dstData;
auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL0A{});

uint8_t padList[4] = {0, 0, 0, 0};

CopyL1ToL0ATla<Element> copyOp(params);
copyOp(dstTensor, srcTensor, padList);
```
