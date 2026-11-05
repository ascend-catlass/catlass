# CopyL0CToGm

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_l0c_to_gm.hpp)

[TOC]

## 功能说明

`CopyL0CToGm` 模板负责将矩阵乘累加结果从 L0C（Accumulator Buffer，`CO1`）搬运到 GM（Global Memory），支持：

- **纯类型转换**（Cast）：float → half / bfloat16_t，int32_t → int32_t 等
- **Per-Tensor 量化/反量化**：搬运时应用统一 scalar scale
- **Per-Channel 量化/反量化**：搬运时应用逐通道 scale 向量
- **ReLU 激活**：搬运时在 FixPipe 中直接应用 ReLU 非线性

该模板作为 [TileCopy](../tile_copy/README.md) 的 `CopyL0CToGm` 成员类型，通常由 blockMmad 自动管理。仅在自定义 kernel 模板组装时需要显式声明。

> **依赖**：本模块依赖 [copy_l0c_to_dst](../copy_l0c_to_dst/README.md) 中定义的 `ScaleGranularity` 枚举和 `CopyL0CToDstQuantMode`（Ascend 950）/ `CopyL0CToGmQuantMode`（Atlas A2）量化模式映射表。

## 模板原型

```cpp
template <
    class ArchTag,                                               // 架构标签：Arch::AtlasA2 或 Arch::Ascend950
    class ElementAccumulator,                                    // 累加器元素类型（通常为 float 或 int32_t）
    class GmType,                                                // GM 数据描述：Gemm::GemmType<ElementDst, LayoutDst>
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,  // 量化粒度
    bool ReluEnable = false                                      // 是否开启 ReLU
>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};
```

- `GmType` 的 `LayoutDst` 决定 GM 数据排布，不同 Layout 触发不同偏特化

## 偏特化实现

### NO_QUANT（纯类型转换）

| 架构      | 目标 Layout | 源 Layout | 搬运方式                                    | 说明                                            |
| :-------- | :---------- | :-------- | :------------------------------------------ | :---------------------------------------------- |
| Atlas A2   | RowMajor    | zN        | `AscendC::Fixpipe` + `CFG_ROW_MAJOR`        | 支持 float→half/bf16 等 cast                    |
| Atlas A2   | zN          | zN        | `AscendC::Fixpipe` + `CFG_NZ`               | zN 排布保持，float→float 时 `channelSplit=true` |
| Atlas A2   | NDC1HWC0    | zN        | `AscendC::Fixpipe` + `CFG_NZ`               | Conv 5D tensor                                  |
| Ascend 950 | RowMajor    | zN        | `AscendC::DataCopy` + `SetFixpipeNz2ndFlag` | NZ→RowMajor 转排布                              |
| Ascend 950 | zN          | zN        | `AscendC::DataCopy`                         | zN 排布保持，float→float 时 `channelSplit=true` |
| Ascend 950 | NDC1HWC0    | zN        | `AscendC::Fixpipe` + `CFG_NZ`               | Conv 5D tensor                                  |

### PER_TENSOR（per-tensor 量化/反量化）

| 架构      | 目标 Layout | 源 Layout | 搬运方式                                           | Params              |
| :-------- | :---------- | :-------- | :------------------------------------------------- | :------------------ |
| Atlas A2   | RowMajor    | zN        | `AscendC::Fixpipe` + `CFG_ROW_MAJOR` + `deqScalar` | `float scale = 1.0` |
| Ascend 950 | RowMajor    | zN        | `AscendC::Fixpipe` + `CFG_ROW_MAJOR` + `deqScalar` | `float scale = 1.0` |

### PER_CHANNEL（per-channel 量化/反量化）

| 架构      | 目标 Layout | 源 Layout | 搬运方式                                                  | scale 参数                       |
| :-------- | :---------- | :-------- | :-------------------------------------------------------- | :------------------------------- |
| Atlas A2   | RowMajor    | zN        | `AscendC::Fixpipe` + `CFG_ROW_MAJOR` + `SetFixPipeConfig` | `LocalTensor<uint64_t>` 旁路     |
| Ascend 950 | RowMajor    | zN        | `AscendC::Fixpipe` + `CFG_ROW_MAJOR` 三参数               | `LocalTensor<uint64_t>` 直接传入 |

## 调用接口

### NO_QUANT / PER_TENSOR

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &dst,   // GM 目标 tensor
    AscendC::LocalTensor<ElementSrc> const &src,    // L0C 源 tensor（CO1）
    LayoutDst const &dstLayout,                     // GM 数据 layout
    LayoutSrc const &srcLayout,                     // L0C 数据 layout（固定 zN）
    uint8_t unitFlag = 0                            // unit 标志位
);
```

### PER_CHANNEL（三参数，scale 向量）

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &dst,        // GM 目标 tensor
    AscendC::LocalTensor<ElementSrc> const &src,         // L0C 源 tensor（CO1）
    AscendC::LocalTensor<uint64_t> const &scale,         // per-channel scale tensor
    LayoutDst const &dstLayout,                          // GM 数据 layout
    LayoutSrc const &srcLayout,                          // L0C 数据 layout（固定 zN）
    uint8_t unitFlag = 0                                 // unit 标志位
);
```

## 调用示例

### NO_QUANT：float → half（Atlas A2）

```cpp
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"

using namespace Catlass::Gemm::Tile;

using ElementAccumulator = float;
using ElementDst = half;
using GmType = Gemm::GemmType<ElementDst, layout::RowMajor>;

const int M = 128;
const int N = 256;
auto dstLayout = layout::RowMajor::MakeLayout<ElementDst>(M, N);
auto srcLayout = layout::zN::MakeLayout<ElementAccumulator>(M, N);

AscendC::GlobalTensor<ElementDst> dstGmTensor;
AscendC::LocalTensor<ElementAccumulator> srcL0CTensor;

using CopyOp = CopyL0CToGm<Arch::AtlasA2, ElementAccumulator, GmType>;
CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### PER_TENSOR：int32 → half 反量化（Atlas A2）

```cpp
using ElementAccumulator = int32_t;
using ElementDst = half;
using GmType = Gemm::GemmType<ElementDst, layout::RowMajor>;
using CopyOp = CopyL0CToGm<Arch::AtlasA2, ElementAccumulator, GmType, ScaleGranularity::PER_TENSOR>;

auto dstLayout = layout::RowMajor::MakeLayout<ElementDst>(M, N);
auto srcLayout = layout::zN::MakeLayout<ElementAccumulator>(M, N);

CopyOp::Params params(0.5f);
CopyOp copyOp(params);
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### PER_CHANNEL：int32 → int8（Atlas A2）

```cpp
using ElementAccumulator = int32_t;
using ElementDst = int8_t;
using GmType = Gemm::GemmType<ElementDst, layout::RowMajor>;
using CopyOp = CopyL0CToGm<Arch::AtlasA2, ElementAccumulator, GmType, ScaleGranularity::PER_CHANNEL>;

AscendC::LocalTensor<uint64_t> scaleTensor;
CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, scaleTensor, dstLayout, srcLayout);
```

### ReLU 激活输出

```cpp
using CopyOp = CopyL0CToGm<Arch::AtlasA2, float, GmType,
    ScaleGranularity::NO_QUANT, true>;
CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### Ascend 950 RowMajor

```cpp
using GmType = Gemm::GemmType<half, layout::RowMajor>;
using CopyOp = CopyL0CToGm<Arch::Ascend950, float, GmType>;

auto dstLayout = layout::RowMajor::MakeLayout<half>(M, N);
auto srcLayout = layout::zN::MakeLayout<float>(M, N);

CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### Ascend 950 zN

```cpp
using GmType = Gemm::GemmType<float, layout::zN>;
using CopyOp = CopyL0CToGm<Arch::Ascend950, float, GmType>;

auto dstLayout = layout::zN::MakeLayout<float>(M, N);
auto srcLayout = layout::zN::MakeLayout<float>(M, N);

CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```
