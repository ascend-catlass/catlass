# Block MMAD Code Explained

## Block MMAD Overview

Block Matrix Multiply-Add (Block MMAD) is a core component in the CATLASS template library responsible for block matrix multiplication. It resides in the middle layer of the compute architecture. It interfaces with the Kernel layer above and the Tile layer below, efficiently loading data from global memory (GM) into local memory (L1/L0) and scheduling tile matrix multiplication tasks.

Block MMAD adopts a highly modular and template-based design. It supports multiple scheduling strategies, tile shapes, and data types. This flexibility allows it to adapt to different hardware architectures and computational requirements. This document uses `BlockMmadPingpong` as an example.

## Template Assembly Mechanism

The Block MMAD implementation is based on the following basic template structure:

```cpp
template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmad {
    // Type definition and core implementation
};
```

### Core Template Parameters

| Parameter         | Description                                                                        |
| ----------------- | ---------------------------------------------------------------------------------- |
| DispatchPolicy    | Scheduling policy that controls task distribution and execution flow               |
| L1TileShape       | Shape of the tile at the L1 cache level, defining the M, N, and K dimensions       |
| L0TileShape       | Shape of the tile at the L0 cache level, defining the M, N, and K dimensions       |
| AType/BType/CType | Data types and layout information for input matrices A, B, and output matrix C     |
| BiasType          | (Optional) Bias data type                                                          |
| TileCopy          | Tile-level data copy component responsible for data transfer between memory levels |
| TileMmad          | Tile-level matrix multiplication component responsible for the actual computation  |

### Type Export

Block MMAD establishes a unified type interface through its type export system, making it easier for upper-layer components to use:

```cpp
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;
```

## Memory Management and Cache Design

Block MMAD manages different levels of memory, including global memory (GM), L1 cache, and L0 cache.

### Static Constant Definition

```cpp
static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
```

### Memory Check

```cpp
// Check LayoutC
static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only supports RowMajor yet!");

// Check L1TileShape
static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

// Check L0TileShape
static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
static_assert(L0C_TILE_SIZE <= L0C_SIZE, "L0TileShape exceeding the L0C space!");
```

### Multi-Stage Cache Design

Block MMAD adopts a multi-stage pipeline design, using ping-pong techniques to hide memory access latency:

```cpp
protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
```

## Core Interface Implementation

### Constructor

```cpp
CATLASS_DEVICE
BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
{
    uint32_t l1AOffset = l1BufAddrStart;
    uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
    // Init buffers
    for (uint32_t i = 0; i < STAGES; i++) {
        // Assign L1/L0A/L0B space for each stages
        l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
        l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
        l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
        l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);

        // Assign event ID for each stages
        l1AEventList[i] = i;
        l1BEventList[i] = i + STAGES;
        l0AEventList[i] = i;
        l0BEventList[i] = i + STAGES;

        // The event id that needs to be set before the loop
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
    }
    l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
}
```

### Destructor

```cpp
CATLASS_DEVICE
~BlockMmad()
{
    for (uint32_t i = 0; i < STAGES; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
    }
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
}
```

### operator()

operator() is the core interface of Block MMAD, responsible for executing block-level matrix multiplication.

```cpp
CATLASS_DEVICE
void operator()(
    AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
    AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
    AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
    GemmCoord const &actualShape)
{
    // 1. Initialize parameters and layout.
    // 2. Preload the first batch of data to the L1 cache.
    // 3. Main loop:
    //    a. Preload the next batch of data to the L1 cache.
    //    b. Load the data currently in the L1 cache to the L0 cache.
    //    c. Perform tile-level matrix multiplication.
    //    d. Manage the multi-stage pipeline.
    // 4. Write the result from the L0 cache back to the global memory.
}
```

## Execution Flow Analysis

Using BlockMmadPingpong as an example, the execution flow is as follows:

### Data Preloading

```cpp
// load first matrix A tile from GM to L1
AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

// load first matrix B tile from GM to L1
AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
copyGmToL1B(l1BTensorList[l1ListId], gmB, layoutBInL1, layoutTileB);
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
```

### Main Loop (Multi-Stage Pipeline)

```cpp
// main loop
uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
    uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
    uint32_t kActualNext{0};

    // Preload the next batch of data to the L1 cache.
    if (kLoopIdx < kTileCount - 1) {
        // ... Preloading logic ...
    }

    // Process the data currently in the L1 cache.
    // Get L1 tensor for current stage
    auto l1ATensor = l1ATensorList[l1ListId];
    auto l1BTensor = l1BTensorList[l1ListId];

    // Load L1 cache data to L0 cache and execute computation.
    // ... L0 processing logic ...

    // Proceed to the next stage.
    l1ListId = l1ListIdNext;
    kActual = kActualNext;
}
```

### L0 Processing and Computation

```cpp
for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
    // ... M-dimension loop ...
    for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
        // ... K-dimension loop ...
        for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
            // ... N-dimension loop ...

            // Execute tile-level matrix multiplication.
            bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
            uint8_t unitFlag = 0b00;
            //... unitFlag setting ...
            tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);
        }
    }
}
```

### Result Writeback

```cpp
// copy block out
LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

if constexpr (!ENABLE_UNIT_FLAG) {
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
} else {
    copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
}
```

## Multi-Stage Pipeline and Event Synchronization

Block MMAD uses event-driven synchronization to ensure the correct execution of the multi-stage pipeline.

```cpp
// Wait for an event.
AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);

// Execute the operation.
copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);

// Trigger an event.
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
```

## Commonalities and Differences Among Block MMAD Implementations

CATLASS provides multiple Block MMAD implementations. They share the following commonalities:

1. **Unified template interface**: All implementations are based on the same template parameter structure.
2. **Multi-stage pipeline design**: All implementations use a multi-stage design to hide memory access latency.
3. **Event-driven synchronization**: All implementations use events to ensure the correct ordering of data transfer and computation.
4. **Memory hierarchy management**: All implementations manage data transfers between GM, L1, and L0.

The major differences among implementations are:

1. **Scheduling policies**: ping-pong, preload, streamk, and more
2. **Hardware adaptation**: optimization for different hardware architectures
3. **Special features**: quantization, sparse computation, bias processing, and more
4. **Performance optimization**: different pipeline depths and memory access patterns

## Summary

Block MMAD is a key component in the CATLASS template library that connects the kernel layer and the tile layer. It achieves efficient block-level matrix multiplication through the following designs:

1. **Modular design**: Assembles different functional components through template parameters.
2. **Multi-stage pipeline**: Uses ping-pong techniques to hide memory access latency.
3. **Event-driven synchronization**: Ensures the correct ordering of data transfer and computation.
4. **Automatic memory management**: Efficiently manages memory resources at different levels.
5. **Flexible adaptation**: Supports different hardware architectures and computational requirements.

The design of Block MMAD reflects the core ideas of modern high-performance compute libraries. Through elaborate pipeline design and memory management, Block MMAD unlocks hardware computational power to unleash matrix multiplication efficiency.
