# Epilogue Adaptation and Development Explained

## Epilogue Overview

Epilogue is the final stage of General Matrix Multiply (GEMM). It is responsible for post-processing operations on the result of matrix multiplication, such as activation functions, quantization/dequantization, and bias addition. In the CATLASS framework, epilogue adopts a modular design that supports flexible combination and extension of multiple post-processing operations.

## Host-layer Epilogue Adaptation

### Dispatch Policy Selection

Based on different epilogue operation requirements, an appropriate dispatch policy must be selected. For example, in a per-token dequantization scenario, we select the `EpilogueAtlasA2PerTokenDequant` policy:

```cpp
using DispatchPolicy = EpilogueAtlasA2PerTokenDequant;
```

### Data Type Definitions

Define the various data types involved in the epilogue based on specific computation requirements:

```cpp
using ElementScale = float;
using ElementPerTokenScale = float;
using ElementD = half;
using LayoutScale = Layout<ScaleType::Vector>;
using LayoutPerTokenScale = Layout<ScaleType::Vector>;
using LayoutD = RowMajor;
```

### Tile Component Configuration

Configure the corresponding tile components based on the type of epilogue operation:

```cpp
using TileRowBroadcastMul = TileRowBroadcastMul<...>;
using TileBroadcastOneBlk = TileBroadcastOneBlk<...>;
using TileCopy = TileCopy<...>;
```

### BlockEpilogue Assembly

Assemble the configured tile components into a complete BlockEpilogue.

```cpp
using BlockEpilogue = BlockEpilogue<DispatchPolicy, CType, ScaleType, PerTokenScaleType, DType, TileRowBroadcastMul, TileBroadcastOneBlk, TileCopy>;
```

### Kernel Integration

Integrate the BlockEpilogue into the kernel. For example, use `QuantMatmulMultiStageWorkspace`:

```cpp
using Kernel = QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;
```

## Kernel-layer Epilogue Adaptation

### Parameter Definition

Define a struct that contains the parameters required for epilogue operations:

```cpp
struct Params {
    GemmCoord problemShape;
    __gm__ ElementA *ptrA;
    LayoutA layoutA;
    __gm__ ElementB *ptrB;
    LayoutB layoutB;
    __gm__ ElementScale *ptrScale;
    LayoutScale layoutScale;
    __gm__ ElementPerTokenScale *ptrPerTokenScale;
    LayoutPerTokenScale layoutPerTokenScale;
    __gm__ ElementD *ptrD;
    LayoutD layoutD;
    GM_ADDR ptrWorkspace;
    // ...
};
```

### AIV Core Implementation

Implement epilogue operations of the AIV core:

```cpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIV>(Params const &params) {
    // ... AIV core epilogue implementation
}
```

### AIC/AIV Synchronization

Implement synchronization between the AIC core and the AIV core:

```cpp
Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
// ... Computation operation ...
Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);
```

### Workspace Adaptation

#### Workspace Size Calculation

```cpp
static size_t GetWorkspaceSize(const Arguments &args) {
    size_t lenWorkspace = static_cast<size_t>(L1TileShape::M) * L1TileShape::N *
        args.aicCoreNum * WORKSPACE_STAGES;
    size_t sizeWorkspace = lenWorkspace * sizeof(uint32_t);
    return sizeWorkspace;
}
```

**Code explanation**:

- `L1TileShape::M/N`: Represents the shape of the L1 tile, i.e., the size of the matrix block processed by each AIC core at once.
- `args.aicCoreNum`: Number of AIC cores involved in the computation.
- `WORKSPACE_STAGES`: Number of stages for the workspace, used to implement pipeline parallelism between AIC and AIV.
- `sizeof(uint32_t)`: Size of each element. This example assumes intermediate results are of type `uint32_t`.
- The final workspace size is the size for a single core at a single stage multiplied by the number of cores and the number of stages.

#### AIC Core Writing into Workspace

Implementation where the AIC core writes matrix multiplication results to the workspace:

```cpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIC>(Params const &params) {
    // ... Initialization and preparation ...

    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
    auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

    uint32_t stageId = 0;
    uint32_t stageUsed = 0;

    // Loop through each matrix block.
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        // ... Compute block location and offsets ...

        // Compute the workspace offset for the current stage.
        MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
        int64_t gmOffsetC = layoutC.GetOffset(offsetC);

        // Perform matrix multiplication and write the results to the workspace.
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmC[gmOffsetC], layoutC,
                actualBlockShape,
                callbackBeforeFixpipe, callbackAfterFixpipe
            );
        } else {
            callbackBeforeFixpipe();
            blockMmad(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmC[gmOffsetC], layoutC,
                actualBlockShape
            );
            callbackAfterFixpipe();
        }

        // Proceed to the next stage.
        stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
    }

    // ... Subsequent synchronization and cleanup ...
}
```

**Code explanation**:

- `gmC`: Global tensor pointing to the workspace
- `layoutC`: Workspace layout in the RowMajor format
- `stageId`: ID of the current workspace stage
- `offsetC`: Computes the workspace offset for the current core in the current stage.
- `gmOffsetC`: Converts matrix coordinates into global memory offsets.
- `blockMmad`: Performs matrix multiplication and writes the results to the specified location in the workspace.
- After matrix multiplication completes, `callbackAfterFixpipe` sets the completion flag.

#### AIV Core Reading from Workspace

Implementation where the AIV core reads results from the workspace and performs epilogues:

```cpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIV>(Params const &params) {
    // ... Initialization and preparation ...

    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
    auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

    uint32_t stageId = 0;

    // ... Configure epilogue parameters ...

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        // ... Obtain the block coordinates and actual block shape ...

        // Compute the workspace offset for the current stage.
        MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
        int64_t gmOffsetC = layoutC.GetOffset(offsetC);
        auto gmBlockC = gmC[gmOffsetC];
        auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

        // Wait for the AIC core to complete the computation in the current stage.
        Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);

        // Perform epilogues.
        blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);

        // Notify the AIC core that the computation in the current stage is complete.
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

        // Proceed to the next stage.
        stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
    }

    // ... Subsequent synchronization and cleanup ...
}
```

**Code explanation**:

- `gmC`: Global tensor pointing to the workspace, sharing the same workspace as the AIC core.
- `layoutC`: Layout of the workspace, consistent with the AIC core.
- `stageId`: ID of the current workspace stage, synchronized with the AIC core.
- `offsetC`: Computes the workspace offset for the current core in the current stage.
- `gmBlockC`: Obtains the starting address of the current block in the workspace.
- `Arch::CrossCoreWaitFlag`: Waits for the AIC core to complete computation in the current stage and write to the workspace.
- `blockEpilogue`: Performs epilogues by reading intermediate results from the workspace.
- `Arch::CrossCoreSetFlag`: Notifies the AIC core that epilogues in the current stage complete.

## BlockEpilogue Development

### Template Parameters

Using [EpilogueAtlasA2PerTokenDequant](../../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp) as an example, the template parameters of BlockEpilogue are shown in the table below:

| Parameter           | Type     | Description                                     |
| ------------------- | -------- | ----------------------------------------------- |
| DispatchPolicy      | struct   | Epilogue dispatch policy                        |
| CType               | typename | Element type of input matrix C                  |
| ScaleType           | typename | Type of the global scaling factor               |
| PerTokenScaleType   | typename | Type of the per-token scaling factor            |
| DType               | typename | Element type of output matrix D                 |
| TileRowBroadcastMul | typename | Tile component for row broadcast multiplication |
| TileBroadcastOneBlk | typename | Tile component for single-block broadcast       |
| TileCopy            | typename | Tile component for data copy                    |

### Core Methods

The core methods of BlockEpilogue include:

- `UpdateParams()`: Updates the epilogue parameters.
- `operator()`: Performs epilogues.

### UB Management

BlockEpilogue needs to manage Unified Buffer (UB) resources, including:

- UB storage of the input matrix C
- UB storage of the scaling factor
- UB storage of the output matrix D

## Tile Component Development

### Tile Types

Tile component types vary depending on the epilogues. For example:

- `TileRowBroadcastMul`: row broadcast multiplication
- `TileBroadcastOneBlk`: single-block broadcast
- `TileCopy`: data copy

### Tile Struct

The struct of a tile component typically includes:

- Template parameters: Define the tile shape, data type, and more.
- Core methods: Implement tile-level operations.
- UB management: Manage the UB resources used by tiles.

### Using Tiles

In BlockEpilogue, the use of tile components typically includes:

- Initializing tile components
- Configuring tile parameters
- Calling core tile methods to perform operations

## Summary

Epilogue adaptation and development is an important part of matrix multiplication computation in CATLASS. By selecting an appropriate dispatch policy, configuring tile components, assembling BlockEpilogue, and implementing collaborative work between AIC and AIV cores, you can efficiently perform various epilogues on matrix multiplication results. Additionally, properly managing the workspace and UB resources can further improve the epilogue performance.
