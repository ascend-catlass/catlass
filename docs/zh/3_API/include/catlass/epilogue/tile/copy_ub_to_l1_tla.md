# CopyUb2L1Tla

> [代码位置](../../../../../../../include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp)

[TOC]

## 功能说明

`CopyUb2L1Tla` 实现 epilogue 阶段从 UB 到 L1 的 zN 格式数据搬运（TLA 风格）。将 UB 上的 zN 格式数据搬运到 L1，保持 zN 格式不变。

- 适用范围：仅 `Arch::Ascend950`（条件编译 `CATLASS_ARCH == 3510`）
- 风格：TLA，使用 `tla::Tensor` 封装操作数
- 布局要求：源 UB 为 zN（非对齐），目的 L1 为 zN（对齐）
- 通过 `AscendC::DataCopy` 实现搬运

## 模板原型

```cpp
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct CopyUb2L1Tla;
```

底层通过 SFINAE 匹配 `iszNUnAlign<ElementSrc, LayoutSrc>` && `iszN<ElementDst, LayoutDst>` 且 `TPosition::VECCALC` → `TPosition::A1`。

| 模板参数    | 说明                                  |
| :---------- | :------------------------------------ |
| `ArchTag`   | 架构标签，仅 Ascend 950 有特化         |
| `TensorSrc` | 源 TLA Tensor，UB 位置，zN 非对齐布局 |
| `TensorDst` | 目的 TLA Tensor，L1 位置，zN 对齐布局 |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

| 参数        | 说明                                                  |
| :---------- | :---------------------------------------------------- |
| `dstTensor` | 目的 L1 TLA Tensor（zN 对齐布局，TPosition::A1）      |
| `srcTensor` | 源 UB TLA Tensor（zN 非对齐布局，TPosition::VECCALC） |

内部使用 `AscendC::DataCopy(dstData[dstOffset], srcData[srcOffset], dataCopyParams)`，按 zN 格式分段搬运。

## 调用示例

```cpp
#include "catlass/epilogue/tile/copy_ub_to_l1_tla.hpp"

using namespace Catlass::Epilogue::Tile;

constexpr uint32_t M = 128;
constexpr uint32_t N = 256;

// 源 U B : zN 非对齐
auto srcLayout = tla::MakeLayout<half, layout::zNUnAlign>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::zN>(M, N);

AscendC::LocalTensor<half> ubTensor;
AscendC::LocalTensor<half> l1Tensor;

auto srcTlaTensor = tla::MakeTensor(ubTensor, srcLayout, Arch::PositionUB{});
auto dstTlaTensor = tla::MakeTensor(l1Tensor, dstLayout, Arch::PositionL1{});

CopyUb2L1Tla<Arch::Ascend950, decltype(srcTlaTensor), decltype(dstTlaTensor)> copyOp;
copyOp(dstTlaTensor, srcTlaTensor);
```
