# CopyUb2GmTla

> [代码位置](../../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm_tla.hpp)

[TOC]

## 功能说明

`CopyUb2GmTla` 实现 epilogue 阶段从 UB 到 GM 的 TLA 风格数据搬运。通过 `tla::Tensor` 封装操作数，利用 SFINAE 根据源/目的 Layout 自动选择搬运策略。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品（RowMajor）、Ascend 950PR&950DT系列产品（RowMajor）
- 风格：TLA（`tla::Tensor` 接口）
- 与 [CopyUb2Gm](./copy_ub_to_gm/README.md) 的区别：TLA 风格，通过 `decltype` 推导模板参数

## 模板原型

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct CopyUb2GmTla;
```

## 偏特化实现

| 架构      | SFINAE 条件                          | 搬运方式                      |
| :-------- | :----------------------------------- | :---------------------------- |
| Atlas A2   | `isRowMajor<Src> && isRowMajor<Dst>` | `DataCopyPad`，C0 对齐 stride |
| Ascend 950 | `isRowMajor<Src> && isRowMajor<Dst>` | `DataCopyPad`，C0 对齐 stride |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

| 参数        | 说明                         |
| :---------- | :--------------------------- |
| `dstTensor` | 目的 TLA Tensor（GM）        |
| `srcTensor` | 源 TLA Tensor（UB, VECCALC） |

## 调用示例

```cpp
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"

using namespace Catlass::Epilogue::Tile;

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(128, 256);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(128, 256);

AscendC::LocalTensor<half> srcData;
AscendC::GlobalTensor<half> dstData;

auto srcTensor = tla::MakeTensor(srcData, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstData, dstLayout, Arch::PositionGM{});

CopyUb2GmTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
