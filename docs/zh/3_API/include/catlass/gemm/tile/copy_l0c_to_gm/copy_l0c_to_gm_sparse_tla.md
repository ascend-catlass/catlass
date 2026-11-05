# CopyL0CToGmSparseTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp)

[TOC]

## 功能说明

`CopyL0CToGmSparseTla` 是 TLA 风格的 Sparse GEMM L0C→GM 数据搬运模板。负责将 Sparse GEMM 的矩阵乘累加结果从 L0C（`CO1`）搬运到 GM，支持类型转换（`QuantMode_t::F322F16` 等）。

与普通 [CopyL0CToGmTla](./tile_copy_tla.md) 不同，`CopyL0CToGmSparseTla` 通过 `FixpipeParamsV220` 参数结构体（而非 `FixpipeParams`）控制搬运尺寸，适用于 Sparse GEMM 场景中与 dense 输出混合的场景。

> **限制**：该模板仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`），仅 NO_QUANT 模式。

## 模板原型

```cpp
template <
    class ArchTag,                                          // 架构标签
    class TensorSrc,                                        // 源 Tensor（L0C）
    class TensorDst,                                        // 目标 Tensor（GM）
    ScaleGranularity DEQUANT_GRANULARITY = NO_QUANT,        // 量化模式
    class Enable = void                                     // SFINAE 分发
>
struct CopyL0CToGmSparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};
```

| 模板参数              | 说明                                                                    |
| :-------------------- | :---------------------------------------------------------------------- |
| `ArchTag`             | 架构标签，仅支持 `Arch::AtlasA2`                                        |
| `TensorSrc`           | 源 Tensor：`tla::Tensor<LocalTensor<ElementSrc>, Layout, Coord, CO1>`   |
| `TensorDst`           | 目标 Tensor：`tla::Tensor<GlobalTensor<ElementDst>, Layout, Coord, GM>` |
| `DEQUANT_GRANULARITY` | 量化模式，仅 `NO_QUANT`                                                 |
| `Enable`              | SFINAE 条件                                                             |

## 偏特化实现

### Atlas A2

| 源        | 目标 Layout | 量化模式 | SFINAE 条件             | 说明                        |
| :-------- | :---------- | :------- | :---------------------- | :-------------------------- |
| L0C（zN） | RowMajor    | NO_QUANT | `isRowMajor<LayoutDst>` | Fixpipe v220，CFG_ROW_MAJOR |

通过 `FixpipeParamsV220` 参数结构体控制搬运尺寸（`nSize`、`mSize`），通过 `CopyL0CToGmQuantMode` 自动推导量化精度。

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

- `dstTensor`：GM 上的目标 Tensor（RowMajor）
- `srcTensor`：L0C 上的源 Tensor（zN 格式）

## 调用示例

### L0C → GM RowMajor（Atlas A2）

```cpp
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t N = 256;

auto srcLayout = tla::MakeLayout<float, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);

AscendC::LocalTensor<float> srcL0CTensor;
AscendC::GlobalTensor<half> dstGmTensor;

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

CopyL0CToGmSparseTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor);
```
