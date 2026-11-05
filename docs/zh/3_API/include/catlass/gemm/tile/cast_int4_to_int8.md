# TileCastInt4ToInt8

> [代码位置](../../../../../../../include/catlass/gemm/tile/cast_int4_to_int8.hpp)

[TOC]

## 功能说明

`TileCastInt4ToInt8` 模板负责将 INT4 量化数据（`int8_t` 存储，每字节存 2 个 INT4）转换为 INT8（`int8_t`），结果直接写回 GM。常用于 A 矩阵（weight）在计算前完成 INT4→INT8 的 Prologue 类型转换。

**流水线**：GM(int4 packed in int8) → UB → Cast(int4→half) → Cast(half→int8) → UB → GM(int8)，使用 2 级双缓冲（STAGES=2）和事件驱动实现 MTE2/V/MTE3 三级流水并发。

转换通过 Vector 引擎的 `AscendC::Cast` 两次中转完成：INT4 → half → INT8。

> **限制**：仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`）。COMPUTE_LEN 不允许超过 24*1024。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：仅 Arch::AtlasA2
    class SrcType_,                   // 源类型：Gemm::GemmType<int8_t, LayoutSrc>
    class DstType_,                   // 目标类型：Gemm::GemmType<int8_t, LayoutDst>
    uint32_t COMPUTE_LEN_,            // 每次计算长度
    uint32_t STAGES = 2               // 流水级数（默认 2）
>
struct TileCastInt4ToInt8 {
    using ElementSrc = typename SrcType_::Element;   // int8_t（INT4 packed）
    using ElementDst = typename DstType_::Element;   // int8_t（INT8）
    using LayoutTagSrc  = typename SrcType_::Layout;
    using LayoutTagDst  = typename DstType_::Layout;
};
```

- `COMPUTE_LEN_`：单次 Cast 计算长度，上限 24*1024
- `SrcType_::Element` 和 `DstType_::Element` 的 sizeof 必须相同（均为 `int8_t`，`sizeof == 1`）
- `LayoutSrc` / `LayoutDst`：支持 RowMajor 或 ColumnMajor

## 构造与析构

### 构造函数

```cpp
TileCastInt4ToInt8(Arch::Resource<ArchTag> const &resource, Params const &params);
```

从 Resource UB buffer 分配双缓冲：

- `ubInTensor[2]`*`COMPUTE_LEN / 2` bytes（INT4 紧凑存储输入）
- `ubOutTensor[2]`*`COMPUTE_LEN` bytes（INT8 输出）
- `ubWorkspace[2]`*`COMPUTE_LEN * sizeof(half)` bytes（中间转换用 half workspace）

初始化事件标志位，确保流水线安全。

Params 为空结构体，无需额外参数。

### 析构函数

```cpp
~TileCastInt4ToInt8();
```

等待所有未完成的 MTE2/V/MTE3 事件，确保流水安全退出。

## 调用接口

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &gmDst, LayoutDst const &layoutDst,   // GM 目标（INT8）
    AscendC::GlobalTensor<ElementSrc> const &gmSrc, LayoutSrc const &layoutSrc    // GM 源（INT4 packed）
);
```

49 个 SubBlock 并行处理，每个 SubBlock 处理 `tilesPerAiv` 行，每轮最多 32 行（`tilesPerLoop=32`），不足 32 行时在最后一轮自适应。

## 调用示例

```cpp
#include "catlass/gemm/tile/cast_int4_to_int8.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = int8_t;   // INT4 packed in int8
using ElementDst = int8_t;   // INT8
using SrcType = Gemm::GemmType<ElementSrc, layout::RowMajor>;
using DstType = Gemm::GemmType<ElementDst, layout::RowMajor>;

constexpr uint32_t COMPUTE_LEN = 16 * 1024;

const int M = 256;
const int K = 8192;   // INT4 packed K，解包后为 16384
auto layoutSrc = layout::RowMajor::MakeLayout<ElementSrc>(M, K);
auto layoutDst = layout::RowMajor::MakeLayout<ElementDst>(M, K * 2);

AscendC::GlobalTensor<ElementSrc> gmSrc;
AscendC::GlobalTensor<ElementDst> gmDst;

Arch::Resource<Arch::AtlasA2> resource;
TileCastInt4ToInt8<Arch::AtlasA2, SrcType, DstType, COMPUTE_LEN>::Params params;

TileCastInt4ToInt8<Arch::AtlasA2, SrcType, DstType, COMPUTE_LEN> castOp(resource, params);
castOp(gmDst, layoutDst, gmSrc, layoutSrc);
```
