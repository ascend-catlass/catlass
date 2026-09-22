# Block Scheduler 代码开发详解

## Block Scheduler 概述

Block Scheduler（块调度器）是CATLASS模板库中负责块级任务调度的核心组件，位于Block层，主要负责管理和分配矩阵乘法计算任务到不同的计算单元。它通过优化任务分配顺序和数据访问模式，提高计算资源利用率和缓存命中率，从而提升整体性能。

Block Scheduler采用模板化设计，支持多种调度策略和硬件架构，主要包括：

- 窗口化调度（Window-based Scheduling）
- 蛇形扫描（Snake Scanning）
- 动态任务分配（Dynamic Task Allocation）

本文将以`GemmIdentityBlockSwizzle`为例，详细讲解Block Swizzle的代码结构、主要接口和调度策略。

## 模板组装机制

GemmIdentityBlockSwizzle实现基于以下基础模板结构：

```cpp
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    // 核心实现
};
```

### 核心模板参数

| 参数名           | 描述                                            |
| ---------------- | ----------------------------------------------- |
| SwizzleOffset    | 窗口化调度的偏移量，默认为1                     |
| SwizzleDirection | 调度方向，0表示按Zn方向，1表示按Nz方向，默认为0 |

这些模板参数与调度策略算法**具体实现**相关，允许用户灵活配置调度策略，以适应不同的硬件架构和性能需求。

## 核心数据结构

GemmIdentityBlockSwizzle包含以下核心数据成员，用于维护调度状态和计算参数：

```cpp
/// Data members
GemmCoord problemShape;  // 问题形状，包含M、N、K三个维度的大小
MatrixCoord tileMN;      // Tile形状，包含M、N两个维度的大小
MatrixCoord loopsMN;     // 循环次数，包含M、N两个维度的循环次数
```

这些数据成员共同构成了调度器的状态，用于跟踪当前的计算进度和参数。

## 主要接口说明

### 构造函数

```cpp
CATLASS_DEVICE
GemmIdentityBlockSwizzle() {}
```

```cpp
CATLASS_DEVICE
GemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
    : problemShape(problemShape_), tileMN(tileMN_)
{
    loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
}
```

```cpp
CATLASS_DEVICE
GemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_,
    MatrixCoord const &loopsMN_) : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_) {}
```

GemmIdentityBlockSwizzle提供了三个构造函数，分别用于默认初始化、根据问题形状和Tile形状初始化，以及直接指定循环次数的初始化。

### Update方法

```cpp
CATLASS_DEVICE
void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
{
    problemShape = problemShape_;
    tileMN = tileMN_;
    loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
}
```

```cpp
CATLASS_DEVICE
void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_)
{
    problemShape = problemShape_;
    tileMN = tileMN_;
    loopsMN = loopsMN_;
}
```

Update方法用于更新调度器的参数，支持动态调整问题形状、Tile形状和循环次数。

### GetCoreLoops方法

```cpp
CATLASS_DEVICE
uint32_t GetCoreLoops() const
{
    return loopsMN.row() * loopsMN.column();
}
```

该方法返回核心循环次数，即M维度循环次数乘以N维度循环次数，表示总共有多少个块需要处理。

### GetBatchIdx方法

```cpp
CATLASS_DEVICE
uint32_t GetBatchIdx(uint32_t taskIdx)
{
    return taskIdx / (GetCoreLoops());
}
```

该方法根据任务索引获取批次索引，用于支持批处理模式。

### GetBlockCoord方法

```cpp
CATLASS_DEVICE
GemmCoord GetBlockCoord(uint32_t taskIdx)
{
    uint32_t innerIdx = taskIdx % GetCoreLoops();
    if constexpr (SwizzleDirection == 0) { // Zn
        uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
        uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
        uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());

        uint32_t nRow = SwizzleOffset;
        if (tileBlockIdx == tileBlockLoop - 1) {
            nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
        }
        uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
        uint32_t nIdx = inTileBlockIdx / nRow;
        if (tileBlockIdx % 2 == 1) {
            nIdx = loopsMN.column() - nIdx - 1;
        }
        return GemmCoord{mIdx, nIdx, 0};
    } else if constexpr (SwizzleDirection == 1) { // Nz
        // 类似的Nz方向处理逻辑
    }
}
```

GetBlockCoord是GemmIdentityBlockSwizzle的核心方法，根据任务索引计算块的坐标。它支持两种调度方向：

- Zn方向：按行分块，每块包含SwizzleOffset行
- Nz方向：按列分块，每块包含SwizzleOffset列

该方法还实现了蛇形扫描策略，当块索引为奇数时，反转N维度或M维度的索引，以优化内存访问模式。

### GetActualBlockShape方法

```cpp
CATLASS_DEVICE
GemmCoord GetActualBlockShape(GemmCoord blockCoord)
{
    uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
        (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
    uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
        (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
    uint32_t kActual = problemShape.k();
    return GemmCoord{mActual, nActual, kActual};
}
```

该方法根据块坐标计算块的实际大小，特别是处理尾部块的边界情况。当块是最后一行或最后一列时，需要调整块的大小以适应实际的问题规模。

## 调度策略分析

### 窗口化调度

GemmIdentityBlockSwizzle采用窗口化调度策略，将M维度或N维度的Tile划分为多个窗口进行处理：

```cpp
uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
```

窗口化调度可以提高缓存命中率，减少缓存抖动，特别是在处理大规模矩阵时效果显著。

### 蛇形扫描

在GetBlockCoord方法中实现了蛇形扫描策略：

```cpp
if (tileBlockIdx % 2 == 1) {
    nIdx = loopsMN.column() - nIdx - 1;
}
```

蛇形扫描可以优化内存访问模式，减少内存带宽压力，提高数据加载效率。

### 尾部块处理

调度器对尾部块进行特殊处理，确保计算的正确性：

```cpp
uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
    (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
```

对于不完整的尾部块，调度器会调整块大小，确保计算的正确性。

## 动态调度扩展

GemmIdentityBlockSwizzle还提供了动态调度的扩展版本`DynamicGemmIdentityBlockSwizzle`：

```cpp
struct DynamicGemmIdentityBlockSwizzle : public GemmIdentityBlockSwizzle<>
{
    uint32_t swizzleOffset{1};
    uint32_t swizzleDirection{0};

    // 构造函数和方法

    CATLASS_DEVICE
    void SetSwizzleParams(uint32_t swizzleOffset_, uint32_t swizzleDirection_)
    {
        swizzleOffset = swizzleOffset_;
        swizzleDirection = swizzleDirection_;
    }

    // 重写的GetBlockCoord方法
};
```

DynamicGemmIdentityBlockSwizzle允许在运行时动态调整SwizzleOffset和SwizzleDirection参数，提供了更灵活的调度控制。

## 执行流程分析

Block Scheduler的典型执行流程如下：

1. **初始化**：创建调度器实例，设置问题形状和Tile形状
2. **计算循环次数**：根据问题形状和Tile形状计算M和N维度的循环次数
3. **任务分配**：
   - 根据任务索引获取批次索引（如果支持批处理）
   - 根据任务索引计算块坐标
   - 根据块坐标计算块的实际大小

## 总结

Block Scheduler是CATLASS模板库中负责块级任务调度的核心组件，通过窗口化调度、蛇形扫描、尾部块处理等策略，优化内存访问模式和缓存命中率，从而提升整体性能。

Block Scheduler采用模板化设计，支持多种调度策略和硬件架构，能够灵活适应不同的计算需求。它还提供了动态调度的扩展版本，允许在运行时调整调度参数。

通过合理配置Block Scheduler的参数，可以显著提升矩阵乘法的性能，特别是在处理大规模矩阵时效果更加明显。
