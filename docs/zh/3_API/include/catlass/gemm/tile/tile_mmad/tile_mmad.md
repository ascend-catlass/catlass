# TileMmad

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_mmad.hpp)

[TOC]

## 功能说明

`TileMmad` 使用 [AscendC::Mmad](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.md) 基础 API 完成矩阵乘加 `C += A * B`。操作数 A 在 L0A，B 在 L0B，C 在 L0C，排布格式分别为 zZ、nZ、zN。非 TLA 实现。

支持两种调用模式：

- 无 Bias：标准矩阵乘加
- 带 Bias：将 BT 中的 Bias 加载到 L0C 后执行矩阵乘加

与 TLA 版本 [TileMmadTla](./tile_mmad_tla.md) 的区别在于本模板直接操作 `AscendC::LocalTensor`，不使用 `tla::Tensor` 封装。

### 架构差异

| 架构             | `kDirectionAlign`                       | `disableGemv`                             | 说明           |
| :--------------- | :-------------------------------------- | :---------------------------------------- | :------------- |
| Atlas A2 (2201)   | `float` + `ColumnMajor`/`nZ` L1A 时开启 | —                                         | K 方向对齐优化 |
| Ascend 950 (3510) | —                                       | L1A 为 `VectorLayout` 时 false，其他 true | GEMV 模式控制  |

## 模板原型

```cpp
template <
    class ArchTag_,            // 架构标签
    class AType_,              // A 矩阵 GmType
    class BType_,              // B 矩阵 GmType
    class BiasType_            // Bias GmType
>
struct TileMmad;
```

## 调用接口

### 无 Bias

```cpp
void operator()(
    AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,   // L0C 累加结果
    AscendC::LocalTensor<ElementA> const &l0ATensor,             // L0A 左矩阵
    AscendC::LocalTensor<ElementB> const &l0BTensor,             // L0B 右矩阵
    uint32_t m,             // M 维度（对齐后）
    uint32_t n,             // N 维度（对齐后）
    uint32_t k,             // K 维度（对齐后）
    bool initC = true,      // true=覆盖, false=原子累加
    uint8_t unitFlag = 0    // L0C→GM 并行搬运标志
);
```

### 带 Bias

```cpp
void operator()(
    AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
    AscendC::LocalTensor<ElementA> const &l0ATensor,
    AscendC::LocalTensor<ElementB> const &l0BTensor,
    AscendC::LocalTensor<ElementAccumulator> const &l0BiasTensor,  // BT Bias 数据
    uint32_t m, uint32_t n, uint32_t k,
    bool initC = true,      // 带 Bias 时强制 false（内部覆盖）
    uint8_t unitFlag = 0
);
```

### 尾流水屏障

当 `(m / 16) * (n / 16) < 10` 时自动插入 `PipeBarrier<PIPE_M>()`，防止流水冲突。

## 调用示例

### 无 Bias

```cpp
#include "catlass/gemm/tile/tile_mmad.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<half, layout::zZ>;
using BType = Gemm::GemmType<half, layout::nZ>;
using BiasType = void;

AscendC::LocalTensor<half> l0ATensor;
AscendC::LocalTensor<half> l0BTensor;
AscendC::LocalTensor<float> l0CTensor;

Tile::TileMmad<Arch::AtlasA2, AType, BType, BiasType> mmadOp;
mmadOp(l0CTensor, l0ATensor, l0BTensor, 64, 64, 32);
```

### 带 Bias

```cpp
using AType   = Gemm::GemmType<half, layout::zZ>;
using BType   = Gemm::GemmType<half, layout::nZ>;
using BiasType = Gemm::GemmType<float, layout::VectorLayout>;

AscendC::LocalTensor<float> l0BiasTensor;

Tile::TileMmad<Arch::AtlasA2, AType, BType, BiasType> mmadOp;
mmadOp(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor, 64, 64, 32);
```

### unitFlag 并行搬运

```cpp
bool initC   = true;
uint8_t unitFlag = 1;  // 启用 L0C→GM 并行

// 第 1 次 mmad：初始化 C
mmadOp(l0CTensor, l0ATensor, l0BTensor, 64, 64, 32, initC, unitFlag);

// 后续 mmad：原子累加 + 继续并行
initC = false;
mmadOp(l0CTensor, l0ATensor, l0BTensor, 64, 64, 32, initC, unitFlag);
```
