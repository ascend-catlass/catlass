# TileCopyTla（L0C → GM）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_l0c_to_gm.hpp)

[TOC]

## 功能说明

`CopyL0CToGmTla` 是 `CopyL0CToGm` 的 TLA 封装，负责将矩阵乘累加结果从 L0C（`CO1`）搬运到 GM，支持类型转换、per-tensor/per-channel 量化/反量化和 ReLU 激活。

与 [非 TLA 版本](./copy_l0c_to_gm.md) 的区别在于操作数使用 `tla::Tensor` 封装，偏特化通过 SFINAE 按目标 Layout 自动派发。

> **注意**：该结构体名为 `CopyL0CToGmTla`，与通用 `TileCopyTla` 不同，专用于 L0C→GM 通道。

## 模板原型

```cpp
template <
    class ArchTag,                                               // 架构标签：Arch::AtlasA2 或 Arch::Ascend950
    class TensorSrc,                                             // 源 TLA tensor（L0C, CO1）
    class TensorDst,                                             // 目标 TLA tensor（GM）
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,  // 量化粒度
    bool ReluEnable = false,                                     // 是否开启 ReLU
    class Enable = void                                          // SFINAE 分派
>
struct CopyL0CToGmTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm.");
};
```

## 偏特化实现

偏特化通过 `Enable` SFINAE 自动派发：

| 架构      | SFINAE 条件                              | 说明                                              |
| :-------- | :--------------------------------------- | :------------------------------------------------ |
| Atlas A2   | `isRowMajor<LayoutDst>`                  | GM RowMajor，`AscendC::Fixpipe` + `CFG_ROW_MAJOR` |
| Atlas A2   | `iszN<ElementDst, LayoutDst>`            | GM zN，`AscendC::Fixpipe` + `CFG_NZ`              |
| Ascend 950 | `isRowMajor<LayoutDst>` + NO_QUANT       | `AscendC::DataCopy` + `SetFixpipeNz2ndFlag`       |
| Ascend 950 | `iszN<ElementDst, LayoutDst>` + NO_QUANT | `AscendC::DataCopy`，zN 保持                      |
| Ascend 950 | `isRowMajor<LayoutDst>` + PER_TENSOR     | `AscendC::Fixpipe` + `deqScalar`                  |
| Ascend 950 | `isRowMajor<LayoutDst>` + PER_CHANNEL    | `AscendC::Fixpipe` 三参数，Scale 向量直接传入     |

## 调用接口

### NO_QUANT / PER_TENSOR（基础重载）

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // 目标 tensor（GM, RowMajor 或 zN）
    TensorSrc const &srcTensor,    // 源 tensor（L0C, CO1）
    uint8_t unitFlag = 0           // unit 标志位
);
```

### Ascend 950 批量搬运重载（NO_QUANT only）

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor,
    uint32_t l0Batch,              // L0C batch 数
    uint32_t dstNdStride           // 目标 ND stride
);
```

### PER_CHANNEL 重载（含 scale tensor）

```cpp
template <class TensorDst, class TensorSrc, class TensorQuant>
void operator()(
    TensorDst const &dstTensor,        // 目标 tensor
    TensorSrc const &srcTensor,        // 源 tensor
    TensorQuant const &quantTensor,    // per-channel scale tensor
    uint8_t unitFlag = 0               // unit 标志位
);
```

## 调用示例

### NO_QUANT RowMajor（Atlas A2）

```cpp
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using ElementAccumulator = float;
using ElementDst = half;

const int M = 128;
const int N = 256;

auto srcLayout = tla::MakeLayout<ElementAccumulator, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<ElementDst, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

CopyL0CToGmTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### NO_QUANT zN（Atlas A2）

```cpp
auto srcLayout = tla::MakeLayout<float, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<float, layout::zN>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

CopyL0CToGmTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### NO_QUANT RowMajor（Ascend 950）

```cpp
auto srcLayout = tla::MakeLayout<float, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

CopyL0CToGmTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### PER_TENSOR：int32 → half（Ascend 950）

```cpp
using ElementAccumulator = int32_t;
using ElementDst = half;

auto srcLayout = tla::MakeLayout<ElementAccumulator, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<ElementDst, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

using CopyOp = CopyL0CToGmTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    ScaleGranularity::PER_TENSOR>;
CopyOp::Params params(0.5f);
CopyOp copyOp(params);
copyOp(dstTensor, srcTensor);
```

### PER_CHANNEL：float → int8（Ascend 950）

```cpp
using ElementAccumulator = float;
using ElementDst = int8_t;

auto srcLayout = tla::MakeLayout<ElementAccumulator, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<ElementDst, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

auto quantLayout = tla::MakeLayout<uint64_t, layout::VectorLayout>(N);
auto quantTensor = tla::MakeTensor(scaleData, quantLayout, Arch::PositionL1{});

CopyL0CToGmTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    ScaleGranularity::PER_CHANNEL> copyOp;
copyOp(dstTensor, srcTensor, quantTensor);
```
