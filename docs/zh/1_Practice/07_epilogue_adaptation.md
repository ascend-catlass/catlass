# Epilogue适配与开发详解

## Epilogue概述

Epilogue是矩阵乘法（GEMM）计算的最后阶段，负责对矩阵乘法的结果进行后处理操作，如激活函数、量化/反量化、偏置加法等。在Catlass框架中，Epilogue采用模块化设计，支持多种后处理操作的灵活组合和扩展。

## Host层Epilogue适配

### Dispatch Policy选择

根据不同的Epilogue操作需求，需要选择合适的Dispatch Policy。例如，在per-token反量化场景中，我们选择`EpilogueAtlasA2PerTokenDequant`策略：

```cpp
using DispatchPolicy = EpilogueAtlasA2PerTokenDequant;
```

### 数据类型定义

根据具体的计算需求，定义Epilogue涉及的各种数据类型：

```cpp
using ElementScale = float;
using ElementPerTokenScale = float;
using ElementD = half;
using LayoutScale = Layout<ScaleType::Vector>;
using LayoutPerTokenScale = Layout<ScaleType::Vector>;
using LayoutD = RowMajor;
```

### Tile组件配置

根据Epilogue的操作类型，配置相应的Tile组件：

```cpp
using TileRowBroadcastMul = TileRowBroadcastMul<...>;
using TileBroadcastOneBlk = TileBroadcastOneBlk<...>;
using TileCopy = TileCopy<...>;
```

### BlockEpilogue组装

将配置好的Tile组件组装成完整的BlockEpilogue：

```cpp
using BlockEpilogue = BlockEpilogue<DispatchPolicy, CType, ScaleType, PerTokenScaleType, DType, TileRowBroadcastMul, TileBroadcastOneBlk, TileCopy>;
```

### Kernel集成

将BlockEpilogue集成到Kernel中，例如使用`QuantMatmulMultiStageWorkspace`：

```cpp
using Kernel = QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;
```

## Kernel层Epilogue适配

### 参数定义

定义包含Epilogue操作所需参数的结构体：

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

### AIV核实现

实现AIV核的Epilogue操作：

```cpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIV>(Params const &params) {
    // ... AIV核Epilogue实现
}
```

### AIC/AIV同步

实现AIC核和AIV核之间的同步机制：

```cpp
Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
// ... 计算操作 ...
Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);
```

### Workspace适配

#### Workspace大小计算

```cpp
static size_t GetWorkspaceSize(const Arguments &args) {
    size_t lenWorkspace = static_cast<size_t>(L1TileShape::M) * L1TileShape::N *
        args.aicCoreNum * WORKSPACE_STAGES;
    size_t sizeWorkspace = lenWorkspace * sizeof(uint32_t);
    return sizeWorkspace;
}
```

**代码说明**：

- `L1TileShape::M/N`：表示L1 Tile的形状，即每个AIC核每次处理的矩阵块大小
- `args.aicCoreNum`：参与计算的AIC核数量
- `WORKSPACE_STAGES`：Workspace的多阶段数量，用于实现AIC和AIV的流水线并行
- `sizeof(uint32_t)`：每个元素的大小，这里假设中间结果是uint32_t类型
- 最终的Workspace大小是单个阶段单个核的大小乘以核数量和阶段数

#### AIC核写入Workspace

AIC核将矩阵乘法结果写入Workspace的实现：

```cpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIC>(Params const &params) {
    // ... 初始化和准备代码 ...

    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
    auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

    uint32_t stageId = 0;
    uint32_t stageUsed = 0;

    // 循环处理每个矩阵块
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        // ... 计算块位置和偏移 ...

        // 计算当前阶段的Workspace偏移
        MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
        int64_t gmOffsetC = layoutC.GetOffset(offsetC);

        // 执行矩阵乘法并将结果写入Workspace
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

        // 切换到下一个阶段
        stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
    }

    // ... 后续同步和清理代码 ...
}
```

**代码说明**：

- `gmC`：指向Workspace的全局张量
- `layoutC`：Workspace的布局，使用RowMajor格式
- `stageId`：当前使用的Workspace阶段ID
- `offsetC`：计算当前核在当前阶段的Workspace偏移量
- `gmOffsetC`：将矩阵坐标转换为全局内存偏移量
- `blockMmad`：执行矩阵乘法，并将结果写入Workspace的指定位置
- 矩阵乘法完成后，通过`callbackAfterFixpipe`设置完成标志

#### AIV核从Workspace读取

AIV核从Workspace读取结果并进行后处理的实现：

```cpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIV>(Params const &params) {
    // ... 初始化和准备代码 ...

    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
    auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

    uint32_t stageId = 0;

    // ... 配置Epilogue参数 ...

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        // ... 获取块坐标和实际块形状 ...

        // 计算当前阶段的Workspace偏移
        MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
        int64_t gmOffsetC = layoutC.GetOffset(offsetC);
        auto gmBlockC = gmC[gmOffsetC];
        auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

        // 等待AIC核完成当前阶段的计算
        Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);

        // 执行Epilogue后处理操作
        blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);

        // 通知AIC核当前阶段的计算已完成
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

        // 切换到下一个阶段
        stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
    }

    // ... 后续同步和清理代码 ...
}
```

**代码说明**：

- `gmC`：指向Workspace的全局张量，与AIC核使用同一个Workspace
- `layoutC`：Workspace的布局，与AIC核保持一致
- `stageId`：当前使用的Workspace阶段ID，与AIC核同步
- `offsetC`：计算当前核在当前阶段的Workspace偏移量
- `gmBlockC`：获取Workspace中当前块的起始地址
- `Arch::CrossCoreWaitFlag`：等待AIC核完成当前阶段的计算并写入Workspace
- `blockEpilogue`：执行Epilogue后处理操作，从Workspace读取中间结果进行处理
- `Arch::CrossCoreSetFlag`：通知AIC核当前阶段的后处理已完成

## BlockEpilogue开发

### 模板参数

以[EpilogueAtlasA2PerTokenDequant](../../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp)为例，BlockEpilogue的模板参数如下表所示：

| 参数名              | 类型     | 描述                  |
| ------------------- | -------- | --------------------- |
| DispatchPolicy      | struct   | Epilogue调度策略      |
| CType               | typename | 输入矩阵C的元素类型   |
| ScaleType           | typename | 全局缩放因子类型      |
| PerTokenScaleType   | typename | Per-token缩放因子类型 |
| DType               | typename | 输出矩阵D的元素类型   |
| TileRowBroadcastMul | typename | 行广播乘法Tile组件    |
| TileBroadcastOneBlk | typename | 单块广播Tile组件      |
| TileCopy            | typename | 数据复制Tile组件      |

### 核心方法

BlockEpilogue的核心方法包括：

- `UpdateParams()`：更新Epilogue参数
- `operator()`：执行Epilogue操作

### UB管理

BlockEpilogue需要管理UB（Unified Buffer）资源，包括：

- 输入矩阵C的UB存储
- 缩放因子的UB存储
- 输出矩阵D的UB存储

## Tile组件开发

### Tile类型

根据Epilogue操作的不同，需要开发不同类型的Tile组件，如：

- `TileRowBroadcastMul`：行广播乘法
- `TileBroadcastOneBlk`：单块广播
- `TileCopy`：数据复制

### Tile结构

Tile组件的结构通常包括：

- 模板参数：定义Tile的形状、数据类型等
- 核心方法：实现Tile级别的操作
- UB管理：管理Tile使用的UB资源

### Tile使用

在BlockEpilogue中，Tile组件的使用通常包括：

- 初始化Tile组件
- 配置Tile参数
- 调用Tile的核心方法执行操作

## 总结

Epilogue适配与开发是Catlass框架中矩阵乘法计算的重要组成部分。通过合理选择Dispatch Policy、配置Tile组件、组装BlockEpilogue，并实现AIC/AIV核的协同工作，可以高效地完成各种矩阵乘法后的处理操作。同时，通过合理管理Workspace和UB资源，可以进一步提高Epilogue操作的性能。
