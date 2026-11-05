# TileCastInt8ToFp16Dequant

> [代码位置](../../../../../../../include/catlass/gemm/tile/cast_int8_to_fp16.hpp)

[TOC]

## 功能说明

`TileCastInt8ToFp16Dequant` 模板负责将 INT8 量化数据反量化（dequant）为 FP16（`half`），结果直接写回 GM。常用于 Prologue 阶段对量化权重进行反量化，在计算前完成精度转换。

**流水线**：GM(int8) → UB → Cast(int8→half) → Adds(+zeroPoint) → Muls(*scale) → UB → GM(half)，使用 2 级双缓冲（STAGES=2）和事件驱动实现流水并发。

与 [TileCastFp8ToFp16Dequant](./cast_fp8_to_fp16.md) 的区别在于元素类型为 `int8_t`（而非 FP8），Dequant 操作为 `Cast` + `Adds` + `Muls`（而非 FP8 专用的查表/位运算）。

> **限制**：仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`）。COMPUTE_LEN 不允许超过 32*1024。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：仅 Arch::AtlasA2
    class SrcType_,                   // 源类型：Gemm::GemmType<int8_t, LayoutSrc>
    class DstType_,                   // 目标类型：Gemm::GemmType<half, LayoutDst>
    uint32_t COMPUTE_LEN_,            // 每次计算长度
    uint32_t STAGES = 2               // 流水级数（默认 2）
>
struct TileCastInt8ToFp16Dequant {
    using ElementSrc = typename SrcType_::Element;   // int8_t
    using ElementDst = typename DstType_::Element;   // half
    using LayoutTagSrc  = typename SrcType_::Layout;
    using LayoutTagDst  = typename DstType_::Layout;
};
```

- `COMPUTE_LEN_`：单次 Cast 计算长度，上限 32*1024
- `LayoutSrc` / `LayoutDst`：只支持 RowMajor 或 ColumnMajor，且两者必须相同

## 构造与析构

### 构造函数

```cpp
TileCastInt8ToFp16Dequant(Arch::Resource<ArchTag> const &resource, Params const &params_);
```

从 Resource UB buffer 分配双缓冲：

- `ubInTensor[2]`*`COMPUTE_LEN` bytes（INT8 输入）
- `ubOutTensor[2]`*`COMPUTE_LEN * sizeof(half)` bytes（FP16 输出）

初始化事件标志位。

### Params

```cpp
struct Params {
    half deqScalar;      // 反量化 scale
    half deqZeroPoint;   // 反量化 zero point

    Params() = default;
    Params(half deqScalar_, half deqZeroPoint_);
};
```

### 析构函数

```cpp
~TileCastInt8ToFp16Dequant();
```

等待所有未完成的流水事件。

## 调用接口

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &gmDst, LayoutDst const &layoutDst,   // GM 目标（FP16）
    AscendC::GlobalTensor<ElementSrc> const &gmSrc, LayoutSrc const &layoutSrc    // GM 源（INT8）
);
```

49 个 SubBlock 并行处理，每轮最多 `COMPUTE_LEN / tileLenRoundInt8` 行，自动匹配合适的行数。

## 调用示例

```cpp
#include "catlass/gemm/tile/cast_int8_to_fp16.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = int8_t;
using ElementDst = half;
using SrcType = Gemm::GemmType<ElementSrc, layout::RowMajor>;
using DstType = Gemm::GemmType<ElementDst, layout::RowMajor>;

constexpr uint32_t COMPUTE_LEN = 16 * 1024;

const int M = 256;
const int K = 4096;
auto layoutSrc = layout::RowMajor::MakeLayout<ElementSrc>(M, K);
auto layoutDst = layout::RowMajor::MakeLayout<ElementDst>(M, K);

AscendC::GlobalTensor<ElementSrc> gmSrc;
AscendC::GlobalTensor<ElementDst> gmDst;

Arch::Resource<Arch::AtlasA2> resource;
TileCastInt8ToFp16Dequant<Arch::AtlasA2, SrcType, DstType, COMPUTE_LEN>::Params params(0.5, 0.0);

TileCastInt8ToFp16Dequant<Arch::AtlasA2, SrcType, DstType, COMPUTE_LEN> castOp(resource, params);
castOp(gmDst, layoutDst, gmSrc, layoutSrc);
```
