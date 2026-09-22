# GEMM Kernel Code Explained

## Kernel Code Structure Overview

The GEMM Kernel in the CATLASS template library adopts a highly modular design. It assembles different components through template parameters to implement various matrix multiplication operations. This document uses `BasicMatmul` as an example to break down the core structure and key components of the Kernel code.

## Template Assembly Mechanism

All GEMM Kernels are defined in the form of template classes, which assemble different functional components through template parameters. Take `BasicMatmul` as an example:

```cpp
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BasicMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;
    // ...
};
```

### Core Template Parameters

| Parameter       | Description                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------------- |
| BlockMmad_      | The core computation component responsible for matrix multiplication                            |
| BlockEpilogue_  | Responsible for epilogues of the computation results (e.g., activation functions, quantization) |
| BlockScheduler_ | Responsible for scheduling and distributing computational tasks to different compute cores      |

### Type Export

The types exported through the template parameters form the Kernel's core type system, which includes:

- Architecture tag (ArchTag)
- L1 cache tile shape (L1TileShape)
- Data types (ElementA/B/C/Accumulator)
- Data layouts (LayoutA/B/C)

## Parameter Passing Mechanism

The Kernel uses a two-layer parameter structure: `Arguments` (user interface layer) and `Params` (kernel execution layer).

### Arguments

`Arguments` is the parameter structure used directly by users. It contains the most basic input and output information:

```cpp
struct Arguments {
    GemmCoord problemShape;
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrC;
};
```

### Params

`Params` is the parameter structure used during actual kernel execution. It contains more detailed execution information:

```cpp
struct Params {
    // Data members
    GemmCoord problemShape;
    GM_ADDR ptrA;
    LayoutA layoutA;
    GM_ADDR ptrB;
    LayoutB layoutB;
    GM_ADDR ptrC;
    LayoutC layoutC;

    // Methods
    CATLASS_HOST_DEVICE
    Params()
    {}

    CATLASS_HOST_DEVICE
    Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
           LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_)
        : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
          ptrC(ptrC_), layoutC(layoutC_) {}
};
```

### Parameter Conversion

The `ToUnderlyingArguments` function converts `Arguments` to `Params`:

```cpp
static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
{
    LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
    LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
    LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};
    Params params{args.problemShape, args.ptrA, layoutA, args.ptrB, layoutB, args.ptrC, layoutC};
    return params;
}
```

## Key Functions

### CanImplement

Checks whether the current hardware and environment support the implementation of this Kernel:

```cpp
static bool CanImplement(const Arguments &args)
{
    return true;
}
```

### GetWorkspaceSize

Gets the workspace size required for Kernel execution:

```cpp
static size_t GetWorkspaceSize(const Arguments &args)
{
    return 0;
}
```

### operator()

This is the Kernel's core execution function. It supports different core types (such as AIC, AIV) through template specialization:

```cpp
template <int32_t CORE_TYPE = g_coreType>
CATLASS_DEVICE
void operator()(Params const &params);

/// Executes one Matmul
template <>
CATLASS_DEVICE
void operator()<AscendC::AIC>(Params const &params) {
    BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
    uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

    Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad(resource);

    // Represent the full gm
    AscendC::GlobalTensor<ElementA> gmA;
    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
    AscendC::GlobalTensor<ElementB> gmB;
    gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

    for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        // Compute block location
        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

        // Compute initial location in logical coordinates
        MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
        MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
        MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
        int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
        int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
        int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

        // Compute block-scoped matrix multiply-add
        blockMmad(gmA[gmOffsetA], params.layoutA,
                  gmB[gmOffsetB], params.layoutB,
                  gmC[gmOffsetC], params.layoutC,
                  actualBlockShape);
    }

    AscendC::PipeBarrier<PIPE_ALL>();
}
```

## Execution Flow Analysis

The Kernel's execution flow divides into the following steps:

### Initializing the Scheduler

```cpp
BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
```

### Initializing Resources and Compute Components

```cpp
Arch::Resource<ArchTag> resource;
BlockMmad blockMmad(resource);
```

### Setting Global Memory Tensors

```cpp
AscendC::GlobalTensor<ElementA> gmA;
gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
// Set gmB and gmC...
```

### Looping Through Each Compute Block

```cpp
for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
    // 1. Compute block coordinates.
    GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
    GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

    // 2. Compute memory offsets.
    MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
    // Compute offsetB and offsetC...
    int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
    // Compute gmOffsetB and gmOffsetC...

    // 3. Execute block-level matrix multiplication.
    blockMmad(gmA[gmOffsetA], params.layoutA,
              gmB[gmOffsetB], params.layoutB,
              gmC[gmOffsetC], params.layoutC,
              actualBlockShape);
}
```

### Synchronization

```cpp
AscendC::PipeBarrier<PIPE_ALL>();
```

## Extensions and Differences Among Kernels

By comparing `BasicMatmul`, `BatchedMatmul`, `QuantMatmul`, and `OptimizedMatmul`, you can see their commonalities and differences in the base structure:

### BatchedMatmul Extension

`BatchedMatmul` adds batch processing support to `BasicMatmul`:

```cpp
struct Params {
    // Data members
    uint32_t batchCount; // Added batch count
    GemmCoord problemShape;
    GM_ADDR ptrA;
    LayoutA layoutA;
    int64_t strideA;      // Added batch stride for matrix A
    GM_ADDR ptrB;
    LayoutB layoutB;
    int64_t strideB;      // Added batch stride for matrix B
    GM_ADDR ptrC;
    LayoutC layoutC;
    int64_t strideC;      // Added batch stride for matrix C
    // ...
};
```

### QuantMatmul Extension

`QuantMatmul` adds quantization-related parameters and processing:

```cpp
struct Params {
    // Data members
    GemmCoord problemShape;
    __gm__ ElementA *ptrA;
    LayoutA layoutA;
    __gm__ ElementB *ptrB;
    LayoutB layoutB;
    __gm__ ElementScale *ptrScale;           // Added scale parameters
    LayoutScale layoutScale;
    __gm__ ElementPerTokenScale *ptrPerTokenScale;  // Added per-token scale parameters
    LayoutPerTokenScale layoutPerTokenScale;
    __gm__ ElementD *ptrD;                   // Added output matrix D
    LayoutD layoutD;
    GM_ADDR ptrWorkspace;                    // Added workspace
    // ...
};
```

### OptimizedMatmul Extension

`OptimizedMatmul` adds prologue processing and a more complex parameter structure:

```cpp
template <
    class PrologueA,         // Added prologue for matrix A
    class PrologueB,         // Added prologue for matrix B
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class OptimizedMatmul {
    // ...
    template<bool IsPaddingA = true, bool IsPaddingB = true>
    struct KernelParams : public ParamsBase {
        // Added padding-related parameters
        GM_ADDR ptrWA;
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;
        // ...
    };
    // ...
};
```

## Summary

The CATLASS GEMM Kernel adopts a highly modular and template-based design with the following characteristics:

1. **Template assembly**: Flexibly assembles different functional components through template parameters, enabling code reuse and function extension.
2. **Layered parameters**: Uses Arguments and Params to separate the user interface from kernel execution parameters.
3. **Unified execution process**: All Kernels follow a similar execution flow, including initialization, scheduling, computation, and synchronization.
4. **Scalability**: By extending the base structure, developers can easily implement advanced features such as batch processing, quantization, and optimization.

This design allows the CATLASS template library to efficiently support a wide range of GEMM operations while ensuring code maintainability and extensibility.
