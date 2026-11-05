# CopyGm2UbTla

> [代码位置](../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub_tla.hpp)

[TOC]

## 功能说明

`CopyGm2UbTla` 实现 epilogue 阶段从 GM 到 UB 的 TLA 风格数据搬运。通过 `tla::Tensor` 封装操作数，利用 SFINAE 根据源/目的 Layout 自动选择搬运策略。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品（RowMajor）、Ascend 950PR&950DT系列产品（VectorLayout / RowMajor）
- 风格：TLA（`tla::Tensor` 接口）
- 与 [CopyGm2Ub](./copy_gm_to_ub/README.md) 的区别：TLA 风格，通过 `decltype` 推导模板参数

## 模板原型

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct CopyGm2UbTla;
```

## 偏特化实现

| 架构      | SFINAE 条件                          | 搬运方式                               |
| :-------- | :----------------------------------- | :------------------------------------- |
| Atlas A2   | `isRowMajor<Src> && isRowMajor<Dst>` | `DataCopyPad` + `DataCopyPadExtParams` |
| Ascend 950 | `isVector<Src> && isVector<Dst>`     | `DataCopyPad`，单 block 搬运           |
| Ascend 950 | `isRowMajor<Src> && isRowMajor<Dst>` | `DataCopyPad` + `DataCopyPadExtParams` |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

| 参数        | 说明                           |
| :---------- | :----------------------------- |
| `dstTensor` | 目的 TLA Tensor（UB, VECCALC） |
| `srcTensor` | 源 TLA Tensor（GM）            |

## 调用示例

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"

using namespace Catlass::Epilogue::Tile;

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(128, 256);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(128, 256);

AscendC::GlobalTensor<half> srcData;
AscendC::LocalTensor<half> dstData;

auto srcTensor = tla::MakeTensor(srcData, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstData, dstLayout, Arch::PositionUB{});

CopyGm2UbTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
