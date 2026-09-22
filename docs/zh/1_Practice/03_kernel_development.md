# GEMM Kernel 代码开发详解

## Kernel代码结构概述

CATLASS模板库中的GEMM Kernel采用了高度模块化的设计，通过模板参数组装不同的组件来实现各种矩阵乘法功能。本文将以`BasicMatmul`为例，详细拆解Kernel代码的核心结构和关键组件。

## 模板组装机制

所有GEMM Kernel都采用模板类的形式定义，通过模板参数来组装不同的功能组件。以`BasicMatmul`为例：

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

### 核心模板参数

| 模板参数        | 描述                                       |
| --------------- | ------------------------------------------ |
| BlockMmad_      | 负责矩阵乘法的核心计算组件                 |
| BlockEpilogue_  | 负责计算结果的后处理（如激活函数、量化等） |
| BlockScheduler_ | 负责调度和分配计算任务到不同的计算核心     |

### 类型导出

通过模板参数导出的类型形成了Kernel的核心类型系统，包括：

- 架构标签（ArchTag）
- L1缓存 tile 形状（L1TileShape）
- 数据类型（ElementA/B/C/Accumulator）
- 数据布局（LayoutA/B/C）

## 参数传递机制

Kernel采用了两层参数结构：`Arguments`（用户接口层）和`Params`（内核执行层）。

### Arguments结构

`Arguments`是用户直接使用的参数结构，包含最基本的输入输出信息：

```cpp
struct Arguments {
    GemmCoord problemShape;
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrC;
};
```

### Params结构

`Params`是内核实际执行时使用的参数结构，包含更详细的执行信息：

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

### 参数转换

通过`ToUnderlyingArguments`函数将`Arguments`转换为`Params`：

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

`ToUnderlyingArguments`函数由`DeviceGemm`类的`Initialize`函数内部调用。

## 关键函数解析

### CanImplement

检查当前硬件和环境是否支持实现该Kernel：

```cpp
static bool CanImplement(const Arguments &args)
{
    return true;
}
```

### GetWorkspaceSize

获取Kernel执行所需的工作区大小：

```cpp
static size_t GetWorkspaceSize(const Arguments &args)
{
    return 0;
}
```

### operator()

Kernel的核心执行函数，通过模板特化支持不同的核心类型（如AIC、AIV）：

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

## 执行流程分析

Kernel的执行流程可以概括为以下几个步骤：

### 初始化调度器

```cpp
BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
```

### 初始化资源和计算组件

```cpp
Arch::Resource<ArchTag> resource;
BlockMmad blockMmad(resource);
```

### 设置全局内存张量

```cpp
AscendC::GlobalTensor<ElementA> gmA;
gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
// 设置gmB和gmC...
```

### 循环处理每个计算块

```cpp
for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
    // 1. 计算块坐标
    GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
    GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

    // 2. 计算内存偏移
    MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
    // 计算offsetB和offsetC...
    int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
    // 计算gmOffsetB和gmOffsetC...

    // 3. 执行块级矩阵乘法
    blockMmad(gmA[gmOffsetA], params.layoutA,
              gmB[gmOffsetB], params.layoutB,
              gmC[gmOffsetC], params.layoutC,
              actualBlockShape);
}
```

### 同步操作

```cpp
AscendC::PipeBarrier<PIPE_ALL>();
```

## 不同Kernel的扩展与差异

通过对比`BasicMatmul`、`BatchedMatmul`、`QuantMatmul`和`OptimizedMatmul`，我们可以看到它们在基础结构上的共性和扩展差异：

### BatchedMatmul扩展

`BatchedMatmul`在`BasicMatmul`的基础上增加了批处理支持：

```cpp
struct Params {
    // Data members
    uint32_t batchCount;  // 增加批处理计数
    GemmCoord problemShape;
    GM_ADDR ptrA;
    LayoutA layoutA;
    int64_t strideA;      // 增加A矩阵的批处理 stride
    GM_ADDR ptrB;
    LayoutB layoutB;
    int64_t strideB;      // 增加B矩阵的批处理 stride
    GM_ADDR ptrC;
    LayoutC layoutC;
    int64_t strideC;      // 增加C矩阵的批处理 stride
    // ...
};
```

### QuantMatmul扩展

`QuantMatmul`增加了量化相关的参数和处理：

```cpp
struct Params {
    // Data members
    GemmCoord problemShape;
    __gm__ ElementA *ptrA;
    LayoutA layoutA;
    __gm__ ElementB *ptrB;
    LayoutB layoutB;
    __gm__ ElementScale *ptrScale;           // 增加缩放参数
    LayoutScale layoutScale;
    __gm__ ElementPerTokenScale *ptrPerTokenScale;  // 增加每token缩放参数
    LayoutPerTokenScale layoutPerTokenScale;
    __gm__ ElementD *ptrD;                   // 增加输出D矩阵
    LayoutD layoutD;
    GM_ADDR ptrWorkspace;                    // 增加工作区
    // ...
};
```

### OptimizedMatmul扩展

`OptimizedMatmul`增加了Prologue处理和更复杂的参数结构：

```cpp
template <
    class PrologueA,         // 增加A矩阵预处理
    class PrologueB,         // 增加B矩阵预处理
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class OptimizedMatmul {
    // ...
    template<bool IsPaddingA = true, bool IsPaddingB = true>
    struct KernelParams : public ParamsBase {
        // 增加填充相关参数
        GM_ADDR ptrWA;
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;
        // ...
    };
    // ...
};
```

## 总结

CATLASS GEMM Kernel采用了高度模块化和模板化的设计，具有以下特点：

1. **模板组装**：通过模板参数灵活组装不同的功能组件，实现代码复用和功能扩展
2. **分层参数**：使用Arguments和Params两层结构，分离用户接口和内核执行参数
3. **统一执行流程**：所有Kernel遵循相似的执行流程，包括初始化、调度、计算和同步
4. **可扩展性**：通过扩展基础结构，可以轻松实现批处理、量化、优化等高级功能

这种设计使得CATLASS模板库能够高效地支持各种GEMM操作，同时保持代码的可维护性和可扩展性。
