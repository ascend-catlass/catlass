# Block MMAD 代码开发详解

## Block MMAD 概述

Block MMAD（Block Matrix Multiply-Add）是CATLASS模板库中负责块级矩阵乘法的核心组件，位于计算架构的中间层。它上接Kernel层，下接Tile层，负责将全局内存（GM）中的数据高效地加载到本地内存（L1/L0），并调度Tile级别的矩阵乘法运算。

Block MMAD采用高度模块化和模板化的设计，支持多种调度策略、Tile形状和数据类型，能够灵活适应不同的硬件架构和计算需求，本文将以[BlockMmadPingpong](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)为例详细讲解（后文截取代码片段中，一些变量的构造方式可以查看源代码）。

## 模板组装机制

Block MMAD实现基于以下基础模板结构：

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
    // 类型定义和核心实现
};
```

### 核心模板参数

| 参数名            | 描述                                                 |
| ----------------- | ---------------------------------------------------- |
| DispatchPolicy    | 调度策略，控制计算任务的分配和执行流程               |
| L1TileShape       | L1缓存级别Tile的形状，定义M、N、K三个维度的大小      |
| L0TileShape       | L0缓存级别Tile的形状，定义M、N、K三个维度的大小      |
| AType/BType/CType | 输入矩阵A、B和输出矩阵C的数据类型和布局信息          |
| BiasType          | 偏置数据类型（可选）                                 |
| TileCopy          | Tile级别的数据拷贝组件，负责不同内存层级间的数据传输 |
| TileMmad          | Tile级别的矩阵乘法组件，负责实际的计算操作           |

### 类型导出

Block MMAD通过类型导出系统建立统一的类型接口，方便上层组件使用：

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

## 内存管理与缓存设计

Block MMAD负责管理不同层级的内存，包括全局内存（GM）、L1缓存和L0缓存：

### 静态常量定义

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

### 内存检查

```cpp
// Check LayoutC
static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

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

### 多阶段缓存设计

Block MMAD采用多阶段流水线设计，使用pingpong技术隐藏内存访问延迟：

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

## 核心接口实现

### 构造函数

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

### 析构函数

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

### 核心计算接口 operator()

operator()是Block MMAD的核心接口，负责执行块级矩阵乘法：

```cpp
CATLASS_DEVICE
void operator()(
    AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
    AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
    AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
    GemmCoord const &actualShape)
{
    // 1. 初始化参数和布局
    // 2. 预加载第一批数据到L1缓存
    // 3. 主循环：
    //    a. 预加载下一批数据到L1缓存
    //    b. 将当前L1缓存中的数据加载到L0缓存
    //    c. 执行Tile级矩阵乘法
    //    d. 管理多阶段流水线
    // 4. 将结果从L0缓存写回全局内存
}
```

## 执行流程分析

以BlockMmadPingpong为例，其执行流程如下：

### 数据预加载

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

### 主循环（多阶段流水线）

```cpp
// main loop
uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
    uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
    uint32_t kActualNext{0};

    // 预加载下一批数据到L1缓存
    if (kLoopIdx < kTileCount - 1) {
        // ... 预加载逻辑 ...
    }

    // 处理当前L1缓存中的数据
    // Get L1 tensor for current stage
    auto l1ATensor = l1ATensorList[l1ListId];
    auto l1BTensor = l1BTensorList[l1ListId];

    // 将L1缓存中的数据加载到L0缓存并执行计算
    // ... L0级处理逻辑 ...

    // 切换到下一个阶段
    l1ListId = l1ListIdNext;
    kActual = kActualNext;
}
```

### L0级处理与计算

```cpp
for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
    // ... M维度循环 ...
    for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
        // ... K维度循环 ...
        for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
            // ... N维度循环 ...

            // 执行Tile级矩阵乘法
            bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
            uint8_t unitFlag = 0b00;
            // ... unitFlag设置 ...
            tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);
        }
    }
}
```

### 结果写回

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

## 多阶段流水线与事件同步

Block MMAD使用事件驱动的同步机制确保多阶段流水线的正确执行：

```cpp
// 等待事件
AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);

// 执行操作
copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);

// 触发事件
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
```

## 不同Block MMAD实现的共性与差异

CATLASS提供了多种Block MMAD实现，它们具有以下共性：

1. **统一的模板接口**：所有实现都基于相同的模板参数结构
2. **多阶段流水线设计**：都采用多阶段设计来隐藏内存访问延迟
3. **事件驱动同步**：都使用事件机制确保数据传输和计算的正确顺序
4. **内存层级管理**：都负责管理GM、L1和L0之间的数据传输

不同实现的主要差异在于：

1. **调度策略**：如pingpong、preload、streamk等不同的调度方式
2. **硬件适配**：针对不同硬件架构的优化实现
3. **特殊功能支持**：如量化、稀疏计算、偏置处理等
4. **性能优化**：不同的流水线深度和内存访问模式

## 总结

Block MMAD是CATLASS模板库中连接Kernel层和Tile层的关键组件，它通过以下设计实现了高效的块级矩阵乘法：

1. **高度模块化**：通过模板参数组装不同的功能组件
2. **多阶段流水线**：使用pingpong技术隐藏内存访问延迟
3. **事件驱动同步**：确保数据传输和计算的正确顺序
4. **自动内存管理**：高效管理不同层级的内存资源
5. **灵活适配**：支持不同的硬件架构和计算需求

Block MMAD的设计体现了现代高性能计算库的核心思想，通过精心的流水线设计和内存管理，充分发挥硬件的计算能力，实现高效的矩阵乘法运算。
