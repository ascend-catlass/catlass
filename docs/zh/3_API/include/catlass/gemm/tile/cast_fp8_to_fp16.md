# TileCastFp8ToFp16Dequant

> [代码位置](../../../../../../../include/catlass/gemm/tile/cast_fp8_to_fp16.hpp)

[TOC]

## 功能说明

`TileCastFp8ToFp16Dequant` 模板负责将 FP8（`int8_t` 存储）量化数据反量化（dequant）并转换为 FP16（`half`），结果直接写回 GM。常用于 A 矩阵（weight）的 Prologue 阶段，在计算前完成数据反量化。

**流水线**：GM(fp8) → UB → Dequant(fp8→fp16) → UB → GM(fp16)，使用双缓冲（BUFFER_NUM=2）和 4 个 EventId 实现 MTE2/V/MTE3 三级流水并发。

与 `TileCastInt8ToFp16Dequant` 的区别在于 Dequant 实现不同（FP8 使用查表/位运算），且支持 ColumnMajor 排布。

> **限制**：仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：仅 Arch::AtlasA2
    class SrcType_,                   // 源类型：Gemm::GemmType<ElementSrc, LayoutSrc>
    class DstType_,                   // 目标类型：Gemm::GemmType<ElementDst, LayoutDst>
    uint32_t COMPUTE_LENGTH           // 每次 Vector 引擎计算的长度
>
struct TileCastFp8ToFp16Dequant {
    using ElementSrc = typename SrcType_::Element;   // int8_t（FP8）
    using ElementDst = typename DstType_::Element;   // half（FP16）
    using LayoutTagSrc  = typename SrcType_::Layout;
    using LayoutTagDst  = typename DstType_::Layout;
};
```

- `COMPUTE_LENGTH`：单次 Dequant 计算长度，影响 UB buffer 分配大小
- `LayoutSrc` / `LayoutDst`：只支持 RowMajor 或 ColumnMajor，且两者必须相同

## 构造与析构

### 构造函数

```cpp
TileCastFp8ToFp16Dequant(Arch::Resource<ArchTag> &resource, Params const &params_);
```

从 `Arch::Resource` 的 UB buffer 中分配双缓冲：

- `inputBuffer[2]`*`COMPUTE_LENGTH`*1 byte（FP8 输入）
- `outputBuffer[2]`*`COMPUTE_LENGTH`*2 bytes（FP16 输出）
- `workspace[2]`*`COMPUTE_LENGTH`*2 bytes（计算 workspace）

### Params

```cpp
struct Params {
    half scalar;      // 反量化 scale
    half zeroPoint;   // 反量化 zero point

    Params() = default;
    Params(half scalar_, half zeroPoint_);
};
```

### 析构函数

无（UB buffer 由 Resource 管理）。

## 调用接口

### 主接口（FP8 → FP16 Dequant）

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> gmDst, LayoutDst const &layoutDst,   // GM 目标（FP16）
    AscendC::GlobalTensor<ElementSrc> gmSrc, LayoutSrc const &layoutSrc,   // GM 源（FP8）
    uint32_t &bufferIndex                                                   // 双缓冲索引（in/out）
);
```

### Epilogue 辅助接口（FP32 → FP16 Cast）

```cpp
void EpCastFp32ToFp16(
    AscendC::GlobalTensor<half> gmDst, LayoutRowMajor layoutDst,
    AscendC::GlobalTensor<float> gmSrc, LayoutRowMajor layoutSrc
);
```

用于 Epilogue 阶段将 float 累加结果 cast 为 half，仅支持 RowMajor。

## 调用示例

```cpp
#include "catlass/gemm/tile/cast_fp8_to_fp16.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = int8_t;
using ElementDst = half;
using SrcType = Gemm::GemmType<ElementSrc, layout::RowMajor>;
using DstType = Gemm::GemmType<ElementDst, layout::RowMajor>;

constexpr uint32_t COMPUTE_LENGTH = 16 * 1024;

const int M = 256;
const int K = 4096;
auto layoutSrc = layout::RowMajor::MakeLayout<ElementSrc>(M, K);
auto layoutDst = layout::RowMajor::MakeLayout<ElementDst>(M, K);

AscendC::GlobalTensor<ElementSrc> gmSrc;
AscendC::GlobalTensor<ElementDst> gmDst;

Arch::Resource<Arch::AtlasA2> resource;
TileCastFp8ToFp16Dequant<Arch::AtlasA2, SrcType, DstType, COMPUTE_LENGTH>::Params params(0.5, 0.0);

TileCastFp8ToFp16Dequant<Arch::AtlasA2, SrcType, DstType, COMPUTE_LENGTH> castOp(resource, params);

uint32_t bufferIndex = 0;
castOp(gmDst, layoutDst, gmSrc, layoutSrc, bufferIndex);
```
