# CopyL0CToUBTla

> [代码位置](../../../../../../../include/catlass/gemm/tile/copy_l0c_to_ub.hpp)

[TOC]

## 功能说明

`CopyL0CToUBTla` 模板负责将矩阵乘累加结果从 L0C（Accumulator Buffer，`CO1`）搬运到 UB（Unified Buffer，`VECCALC`），供 Vector 引擎执行逐元素后处理操作（如激活函数、自定义算子等）。

与 [CopyL0CToGm](./copy_l0c_to_gm/README.md) 不同，UB 搬运的目的地是 Vector 计算单元可直接访问的 Unified Buffer，可实现 GM 写回前的中间处理。

> **限制**：仅支持 TLA 风格，仅支持 Ascend 950 架构（`CATLASS_ARCH == 3510`）。Atlas A2 无 L0C→UB 通道。
>
> **依赖**：本模块依赖 [copy_l0c_to_dst](./copy_l0c_to_dst/README.md) 中定义的 `ScaleGranularity`、`CopyL0CToDstQuantMode` 和 `CopyL0CToUBMode`。

## 模板原型

```cpp
template <
    class ArchTag,                                               // 架构标签：仅 Arch::Ascend950
    class TensorSrc,                                             // 源 TLA tensor（L0C, CO1）
    class TensorDst,                                             // 目标 TLA tensor（UB, VECCALC）
    CopyL0CToUBMode CopyMode = CopyL0CToUBMode::NO_SPLIT,        // 搬运模式
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,  // 量化粒度
    bool ReluEnable = false,                                     // 是否开启 ReLU
    class Enable = void                                          // SFINAE 分派
>
struct CopyL0CToUBTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to ub.");
};
```

- `CopyMode`：搬运模式，选用 `NO_SPLIT`、`SPLIT_M`（M 维度对半拆）或 `SPLIT_N`（N 维度 32 对齐拆）
- `TensorDst` 位置必须为 `VECCALC`，Layout 只支持 `RowMajor`

## 偏特化实现

| CopyMode   | M 处理          | N 处理           | dualDstCtl | 搬运指令                                |
| :--------- | :-------------- | :--------------- | :--------- | :-------------------------------------- |
| `NO_SPLIT` | 原始 M          | 原始 N           | —          | `AscendC::Fixpipe` + `CFG_ROW_MAJOR_UB` |
| `SPLIT_M`  | `RoundUp(M, 2)` | 原始 N           | `1`        | `AscendC::Fixpipe` + `CFG_ROW_MAJOR_UB` |
| `SPLIT_N`  | 原始 M          | `RoundUp(N, 32)` | `2`        | `AscendC::Fixpipe` + `CFG_ROW_MAJOR_UB` |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // 目标 tensor（UB, VECCALC, RowMajor）
    TensorSrc const &srcTensor,    // 源 tensor（L0C, CO1）
    uint8_t unitFlag = 0           // unit 标志位
);
```

静态约束：

- `TensorDst::Layout` 为 `RowMajor`
- `TensorSrc::position == CO1`
- `TensorDst::position == VECCALC`

## 调用示例

### NO_SPLIT

```cpp
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

const int M = 128;
const int N = 256;

auto srcLayout = tla::MakeLayout<float, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<float, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstUBTensor, dstLayout, Arch::PositionUB{});

CopyL0CToUBTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### SPLIT_M

```cpp
CopyL0CToUBTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    CopyL0CToUBMode::SPLIT_M> copyOp;
copyOp(dstTensor, srcTensor);
```

### SPLIT_N

```cpp
CopyL0CToUBTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    CopyL0CToUBMode::SPLIT_N> copyOp;
copyOp(dstTensor, srcTensor);
```

### RE_QUANT + ReLU

```cpp
CopyL0CToUBTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, true> copyOp;
copyOp(dstTensor, srcTensor);
```
