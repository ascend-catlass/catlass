# CopyL1ToL0BTla

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0b.hpp)

[TOC]

## 功能说明

`CopyL1ToL0BTla` 实现 Conv 场景下将 Filter 数据从 L1 搬运到 L0B 的 TLA 风格版本。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品
- 风格：TLA

## 模板原型

```cpp
template <class Element>
struct CopyL1ToL0BTla;
```

| 模板参数  | 说明                |
| :-------- | :------------------ |
| `Element` | 元素类型，如 `half` |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // nZ 格式
    TensorSrc const &srcTensor     // CI1KHKWCOCI0 格式
)
```

## 调用示例

```cpp
#include "catlass/conv/tile/atlasa2/copy_l1_to_l0b.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Kh = 3, Kw = 3, Cout = 64, C0 = 16;

auto layoutSrc = tla::MakeLayout<Element, layout::CI1KHKWCOCI0>(Cin1, Kh, Kw, Cout, C0);
auto layoutDst = tla::MakeLayout<Element, layout::nZ>(Cin1 * Kh * Kw, Cout);

AscendC::LocalTensor<Element> srcData, dstData;
auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL0B{});

CopyL1ToL0BTla<Element> copyOp;
copyOp(dstTensor, srcTensor);
```
