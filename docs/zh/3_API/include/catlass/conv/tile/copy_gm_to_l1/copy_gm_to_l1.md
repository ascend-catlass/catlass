# CopyGmToL1

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_gm_to_l1.hpp)

[TOC]

## 功能说明

`CopyGmToL1` 实现 Conv 场景下从 GM 到 L1 的权重/特征图数据搬运（非 TLA 风格），支持 Fmap（特征图）和 Filter（卷积核）两类搬运。

- 适用范围：Atlas A2
- 风格：非 TLA

## 模板原型

```cpp
template <class ArchTag, class GmType, class L1Type = void>
struct CopyGmToL1;
```

| 模板参数  | 说明                                                       |
| :-------- | :--------------------------------------------------------- |
| `ArchTag` | 架构标签                                                   |
| `GmType`  | `Gemm::GemmType<Element, LayoutTag>`，含 Element 和 Layout |
| `L1Type`  | L1 数据类型，默认 `void`（自动推导）                       |

## 偏特化实现

| 偏特化   | GmType                            | 说明                                    |
| :------- | :-------------------------------- | :-------------------------------------- |
| Fmap-A   | `GemmType<Element, NC1HWC0>`      | NC1HWC0→NC1HWC0，逐 Cin1 搬运           |
| Filter-B | `GemmType<Element, CI1KHKWCOCI0>` | CI1KHKWCOCI0→CI1KHKWCOCI0，含 Cout 对齐 |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,    // L1 目的
    AscendC::GlobalTensor<Element> srcTensor,   // GM 源
    LayoutDst const &layoutDst,
    LayoutSrc const &layoutSrc
)
```

## 调用示例

### Fmap（NC1HWC0）

```cpp
#include "catlass/conv/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Conv::Tile;

constexpr uint32_t Cin1 = 4, Hi = 28, Wi = 28, C0 = 16;

using Element = half;
using LayoutTagSrc = layout::NC1HWC0;
using LayoutTagDst = layout::NC1HWC0;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(1, Cin1, Hi, Wi, C0);
auto layoutDst = LayoutTagDst::MakeLayout<Element>(1, Cin1, Hi, Wi, C0);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using CopyOp = CopyGmToL1<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
