# Basic Matmul
>
> [Code location](../../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp)

## Description

Basic matrix multiplication, Cube Core operator, without AIV computation or TLA implementation

## Class Template Overview

- Template input parameters
    - class BlockMmad_: `blockMmad` class, which serves as the matrix multiplication component.
    - class BlockEpilogue_: `blockEpilogue` class, which serves as the epilogue component (not used in practice).
    - class BlockScheduler_: `blockScheduler` class, which strictly supports only [Gemm::Block::GemmIdentityBlockSwizzle](../block/block_swizzle/block_swizzle.md).
- Params:

```
struct Params {
    GemmCoord problemShape;     // Shape of the test case
    GM_ADDR ptrA;               // GM start address of input matrix A
    LayoutA layoutA;            // Storage layout of input matrix A
    GM_ADDR ptrB;               // GM start address of input matrix B
    LayoutB layoutB;            // Storage layout of input matrix B
    GM_ADDR ptrC;               // GM start address of output matrix C
    LayoutC layoutC;            // Storage layout of output matrix C
...
};
```

- Arguments:

```
struct Arguments {
    GemmCoord problemShape;     // Shape of the test case
    GM_ADDR ptrA;               // GM start address of input matrix A
    GM_ADDR ptrB;               // GM start address of input matrix B
    GM_ADDR ptrC;               // GM start address of output matrix C
};
```

## Example

Kernel assembly

```
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// kernel level
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
```

## Constraints

This kernel executes within the `void operator()<AscendC::AIC>` core function. Because the mechanism used to invoke `blockMmad` operates in a synchronous mode without pipeline preloading, it only supports simple blockMmad components such as [block_mmad_pingpong](../block/block_mmad_pingpong.md).
