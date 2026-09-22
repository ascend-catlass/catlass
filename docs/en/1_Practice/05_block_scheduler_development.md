# Block Scheduler Code Explained

## Block Scheduler Overview

Block scheduler is a core component in the CATLASS template library responsible for block-level task scheduling. It resides in the block layer, responsible for managing and distributing matrix multiplication tasks to different compute units. It improves compute resource utilization and cache hit rates by optimizing task distribution order and data access patterns, thereby enhancing overall performance.

Block scheduler adopts a template-based design and supports multiple scheduling policies and hardware architectures, including:

- Window-based scheduling
- Snake scanning
- Dynamic task allocation

This document uses `GemmIdentityBlockSwizzle` as an example to dive into the code structure, main interfaces, and scheduling policies of the block swizzle.

## Template Assembly Mechanism

The GemmIdentityBlockSwizzle implementation is based on the following basic template structure:

```cpp
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    // Core implementation
};
```

### Core Template Parameters

| Parameter        | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| SwizzleOffset    | Offset for window-based scheduling, which defaults to 1                |
| SwizzleDirection | Scheduling direction. 0 (default) for Zn direction, 1 for Nz direction |

These template parameters are related to the **specific implementation** of scheduling algorithms. They allow users to flexibly configure the scheduling policies to adapt to different hardware architectures and performance requirements.

## Core Data Structure

GemmIdentityBlockSwizzle contains the following core data members for maintaining scheduling state and computation parameters:

```cpp
/// Data members
GemmCoord problemShape;  // Problem shape, which contains M, N, K dimensions
MatrixCoord tileMN;      // Tile shape, which contains M and N dimensions
MatrixCoord loopsMN;     // Loop counts for M and N dimensions
```

These data members together form the state of the scheduler and track the current computation progress and parameters.

## Main Interfaces

### Constructor

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

GemmIdentityBlockSwizzle provides three constructors for default initialization, initialization based on problem shape and tile shape, and initialization that directly specifies the loop counts.

### Update Method

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

The Update method dynamically adjusts the problem shape, tile shape, and loop counts.

### GetCoreLoops Method

```cpp
CATLASS_DEVICE
uint32_t GetCoreLoops() const
{
    return loopsMN.row() * loopsMN.column();
}
```

This method returns the core loop count, which is the number of M-dimension loops multiplied by the number of N-dimension loops. It represents the total number of blocks to process.

### GetBatchIdx Method

```cpp
CATLASS_DEVICE
uint32_t GetBatchIdx(uint32_t taskIdx)
{
    return taskIdx / (GetCoreLoops());
}
```

Designed for batch processing, this method returns the batch index based on the task index.

### GetBlockCoord Method

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
        // Similar Nz direction processing logic
    }
}
```

GetBlockCoord is the core method of GemmIdentityBlockSwizzle. It calculates the block coordinate based on the task index. It supports two scheduling directions:

- Zn direction: Blocks are divided by row. Each block contains SwizzleOffset rows.
- Nz direction: Blocks are divided by column. Each block contains SwizzleOffset columns.

This method also implements snake scanning. When the block index is odd, it reverses the N-dimension or M-dimension index to optimize the memory access pattern.

### GetActualBlockShape Method

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

This method calculates the actual size of a block based on its coordinate, especially handling the boundaries of tail blocks. When a block is in the last row or last column, this method adjusts the block size to fit the actual problem size.

## Scheduling Policies

### Window-based Scheduling

GemmIdentityBlockSwizzle adopts window-based scheduling. It divides the tiles in the M or N dimension into multiple windows for processing:

```cpp
uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
```

Window-based scheduling improves the cache hit rate and reduces cache thrashing, especially when processing large-scale matrices.

### Snake Scanning

The GetBlockCoord method implements snake scanning:

```cpp
if (tileBlockIdx % 2 == 1) {
    nIdx = loopsMN.column() - nIdx - 1;
}
```

Snake scanning can optimize the memory access pattern, reduce memory bandwidth pressure, and improve data loading efficiency.

### Tail Block Processing

The scheduler performs special processing on tail blocks to ensure the correctness of computation:

```cpp
uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
    (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
```

For incomplete tail blocks, the scheduler adjusts the block size to ensure the correctness of computation.

## Dynamic Scheduling Extension

GemmIdentityBlockSwizzle also provides a dynamic version called `DynamicGemmIdentityBlockSwizzle`:

```cpp
struct DynamicGemmIdentityBlockSwizzle : public GemmIdentityBlockSwizzle<>
{
    uint32_t swizzleOffset{1};
    uint32_t swizzleDirection{0};

    // Constructor and method

    CATLASS_DEVICE
    void SetSwizzleParams(uint32_t swizzleOffset_, uint32_t swizzleDirection_)
    {
        swizzleOffset = swizzleOffset_;
        swizzleDirection = swizzleDirection_;
    }

    // Overridden GetBlockCoord method
};
```

DynamicGemmIdentityBlockSwizzle allows runtime adjustment of SwizzleOffset and SwizzleDirection, allowing more flexible scheduling control.

## Execution Flow Analysis

The typical execution flow of the Block Scheduler is as follows:

1. **Initialization**: Create a scheduler instance and set the problem shape and tile shape.
2. **Loop count calculation**: Calculate the loop count for the M and N dimensions based on the problem shape and tile shape.
3. **Task distribution**:
   - Obtain the batch index based on the task index (if batch processing is supported).
   - Calculate the block coordinates based on the task index.
   - Calculate the actual size of a block based on its coordinates.

## Summary

Block Scheduler is a core component in the CATLASS template library, responsible for block-level task scheduling. Through policies such as window-based scheduling, snake scanning, and tail block processing, it optimizes the memory access pattern and cache hit rate, thereby improving overall performance.

Block Scheduler adopts a template-based design and supports multiple scheduling policies and hardware architectures, allowing it to flexibly adapt to different computation requirements. It also provides an extended version for dynamic scheduling, allowing parameters to be adjusted at runtime.

By properly configuring the parameters of Block Scheduler, you can significantly improve the performance of matrix multiplication, especially when dealing with large-scale matrices.
