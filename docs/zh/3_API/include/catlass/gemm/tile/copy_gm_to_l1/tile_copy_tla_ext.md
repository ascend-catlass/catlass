# TileCopyTlaExt

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## 功能说明

`TileCopyTlaExt` 是 `TileCopyTla` 的扩展版本，同样用于 TLA 风格的 GM 到 L1 数据搬运。与 `TileCopyTla` 的主要区别在于：

1. **模板参数不同**：`TileCopyTlaExt` 通过显式的 `LayoutTagSrc` 和 `LayoutTagDst` 模板参数来匹配偏特化，而非通过 `std::enable_if_t` + trait 检测
2. **调用接口不同**：`TileCopyTlaExt` 的 `operator()` 额外接收一个 `ActualShape` 参数，允许调用方指定实际搬运的数据块形状（而非使用 tensor 的完整 shape），适用于需要 Padding 或部分搬运的场景

当前仅支持 `Arch::AtlasA2` 架构。

## 模板原型

```cpp
template <
    class ArchTag,          // 架构标签
    class TensorSrc,        // 源操作数 TLA Tensor 类型
    class TensorDst,        // 目的操作数 TLA Tensor 类型
    class LayoutTagSrc,     // 源 Layout 标签（显式指定）
    class LayoutTagDst      // 目的 Layout 标签（显式指定）
>
struct TileCopyTlaExt
```

### 模板参数说明

| 参数           | 说明                                                                          |
| :------------- | :---------------------------------------------------------------------------- |
| `ArchTag`      | 架构标签，当前仅支持 `Arch::AtlasA2`                                          |
| `TensorSrc`    | 源 TLA Tensor，封装 GM GlobalTensor、layout、coord 和 TPosition::GM           |
| `TensorDst`    | 目的 TLA Tensor，封装 L1 LocalTensor、layout、coord 和 TPosition::A1          |
| `LayoutTagSrc` | 显式指定的源 layout 标签，如 `layout::RowMajor`、`layout::PaddingRowMajor` 等 |
| `LayoutTagDst` | 显式指定的目的 layout 标签，如 `layout::zN`、`layout::nZ` 等                  |

## 偏特化实现

所有偏特化仅适用于 `Arch::AtlasA2`。

| LayoutTagSrc                 | LayoutTagDst | 说明                                      |
| :--------------------------- | :----------- | :---------------------------------------- |
| `layout::RowMajor`           | `layout::zN` | RowMajor → zN，支持 ActualShape           |
| `layout::PaddingRowMajor`    | `layout::zN` | PaddingRowMajor → zN，支持 ActualShape    |
| `layout::ColumnMajor`        | `layout::nZ` | ColumnMajor → nZ，支持 ActualShape        |
| `layout::PaddingColumnMajor` | `layout::nZ` | PaddingColumnMajor → nZ，支持 ActualShape |
| `layout::zN`                 | `layout::zN` | zN → zN（保持格式），支持 ActualShape     |
| `layout::nZ`                 | `layout::nZ` | nZ → nZ（保持格式），支持 ActualShape     |

## 调用接口

所有偏特化使用统一的调用接口：

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,     // 目的 TLA Tensor
    TensorSrc const &srcTensor,     // 源 TLA Tensor
    ActualShape actualShape         // 实际搬运的数据块形状
)
```

其中 `ActualShape` 定义为 `tla::Shape<uint32_t, uint32_t>`。

| 参数          | 说明                                                               |
| :------------ | :----------------------------------------------------------------- |
| `dstTensor`   | 目的 TLA Tensor（L1, TPosition::A1）                               |
| `srcTensor`   | 源 TLA Tensor（GM, TPosition::GM）                                 |
| `actualShape` | 实际需要搬运的数据块形状（行数, 列数），可小于 tensor 的完整 shape |

## 与 TileCopyTla 的对比

| 特性             | TileCopyTla                     | TileCopyTlaExt                               |
| :--------------- | :------------------------------ | :------------------------------------------- |
| 偏特化匹配方式   | `std::enable_if_t` + trait 检测 | 显式 `LayoutTagSrc` / `LayoutTagDst`         |
| 支持的架构       | Atlas A2 / Ascend 950             | 仅 Atlas A2                                   |
| ActualShape 参数 | 不支持                          | 支持                                         |
| Padding layout   | 不支持                          | 支持（PaddingRowMajor / PaddingColumnMajor） |
| 适用场景         | 通用矩阵乘 tile 搬运            | 需要部分搬运或 Padding 场景                  |

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;
const uint32_t actualM = 128;
const uint32_t actualK = 128;

// 通过 tla::MakeLayout 创建 Layout
auto layoutSrc = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zN>(M, K);

// 通过 tla::MakeTensor 构造 TLA Tensor
AscendC::GlobalTensor<half> srcGmTensor;
AscendC::LocalTensor<half> dstL1Tensor;
auto srcTensor = tla::MakeTensor(srcGmTensor, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, layoutDst, Arch::PositionL1{});

// 实例化 TileCopyTlaExt（LayoutTagSrc/LayoutTagDst 决定搬运策略，与 tensor 的 layout 无关）
TileCopyTlaExt<Arch::AtlasA2,
    decltype(srcTensor), decltype(dstTensor),
    layout::RowMajor, layout::zN> copyOp;

// 指定实际搬运的数据块形状（可小于 tensor 的完整 shape）
tla::Shape<uint32_t, uint32_t> actualShape(actualM, actualK);
copyOp(dstTensor, srcTensor, actualShape);
```
