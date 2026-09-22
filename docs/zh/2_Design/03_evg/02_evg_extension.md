# EVG 扩展说明

本文整理当前 EVG 的扩展边界，主要回答两件事：什么时候新增 `ComputeFn`，什么时候新增节点；扩展时需要遵守哪些约束。接入方式见 [evg_api](../../3_API/evg_api.md)，整体设计见 [01_evg_design](./01_evg_design.md)。

## 先判断扩展类型

优先做这一步，避免把本来只需新增一个算子的事情做成新节点。

| 场景                                                                      | 处理方式         |
| ------------------------------------------------------------------------- | ---------------- |
| 只是逐元素计算，输入输出都在 UB，不需要 GM、layout、workspace             | 新增 `ComputeFn` |
| 需要读 GM、写 GM、管理 layout、申请 UB，或需要自己的 `Arguments / Params` | 新增节点         |

可以直接按下面的口径判断：

- 只扩“怎么算”，加 `ComputeFn`
- 涉及“怎么取、怎么放、怎么占资源”，加节点

当前最典型的参考实现：

- `ComputeFn`：`Add`、`Muls`、`LeakyRelu`、`AddRelu`
- 节点：`VisitorAuxLoad`、`VisitorAuxStore`、`VisitorRowBroadcast`

## 新增 ComputeFn

### 放置位置

当前 `ComputeFn` 统一放在：

- `include/catlass/epilogue/fusion/operations.hpp`

`VisitorCompute` 通过模板参数 `ComputeFn` 实例化这里的算子。新增逐元素算子时，直接沿用这里的写法即可，通常不再单独新增节点。

### 需要保持的形态

最小形态如下：

```cpp
template <typename T>
struct SomeOp {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1
    ) const {
        // do compute
    }
};
```

固定规则：

- 第一个参数是输出 `dst`
- 第二个参数是 `compute_length`
- 后续参数才是输入
- 输入路数由算子语义决定

如果算子带标量参数，保持聚合类型即可：

```cpp
template <typename T>
struct ClampMin {
    T min_value;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src
    ) const {
        AscendC::Maxs(dst, src, min_value, compute_length);
    }
};
```

对应接入方式：

```cpp
using ClampMinOp =
    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::ClampMin, ElementC, ElementC>;

typename ClampMinOp::Arguments args{{0.0f}};
```

### 当前约束

#### 只做纯计算

当前实现中，`ComputeFn` 只承载 UB 内的计算，下面这些职责仍放在节点层：

- 访问 GM
- 管理 layout
- 申请 workspace
- 处理事件同步
- 依赖全局坐标

这些职责不放在 `ComputeFn` 内处理。

#### 类型规则沿用 `VisitorCompute`

当前 `VisitorCompute` 要求所有输入类型都等于 `ElementCompute`。因此：

- 混合精度场景通常先插 `VisitorCast`
- 输入类型兼容仍放在图里处理，不放在 `ComputeFn` 内展开

#### 多步 V 计算自己补 barrier

如果一个 `operator()` 内包含多步 V 计算，应像 `AddRelu` 一样在步骤之间补 `AscendC::PipeBarrier<PIPE_V>()`。

简单说：

- 单条原子指令通常不用额外补
- 多步串联计算通常要补齐

#### 多输入算子沿用现有展开方式

像 `Add`、`Mul` 这类多输入算子，当前实现采用链式展开。新增多输入算子时，尽量沿用现有模式，避免单独引入新的调用约定。

#### 保持聚合初始化友好

`VisitorCompute` 当前按 `Op<ElementCompute>{...}` 构造算子。新增 `ComputeFn` 时，保持简单字段和聚合初始化会更贴合现有实现。

## 新增节点

### 放置位置

当前节点实现统一放在：

- `include/catlass/epilogue/fusion/visitor_*.hpp`

新增节点后，还应接入：

- `include/catlass/epilogue/fusion/fusion.hpp`

### 优先参考的同级实现

新增节点时，与现有同级实现保持相同的结构和职责：

- `VisitorAuxLoad`：典型叶子节点，负责从 GM 读取 tile
- `VisitorAuxStore`：典型根节点，负责最终写回
- `VisitorRowBroadcast`：典型跨阶段节点，`LOAD` 取数、`COMPUTE` 扩展
- `VisitorCompute`：纯计算节点；如果需求和它一致，通常直接复用即可

### 最小骨架

```cpp
template <class Element>
struct VisitorSomeNode : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    using ElementOutput = Element;

    struct Arguments {};
    struct Params {};

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void* workspace) {
        return Params(/* ... */);
    }

    template <class ProblemShape>
    static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    static bool
    can_implement(ProblemShape const&, Arguments const&) {
        return true;
    }

    struct Callbacks : EmptyCallbacks {
        template <VisitStage Stage, class ArchTag, class TensorC, typename... Inputs>
        CATLASS_DEVICE auto visit(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            Inputs const&... inputs
        ) {
            // stage-specific work
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        return Callbacks(/* ... */);
    }

    Params params;
};
```

### 必备组成与职责

| 组成                      | 作用                         | 约束                            |
| ------------------------- | ---------------------------- | ------------------------------- |
| `ElementOutput`           | 告诉图组织器节点输出类型     | 与实际输出一致                  |
| `Arguments`               | 用户侧直接填写的参数         | 保持简单，支持花括号初始化      |
| `Params`                  | device 侧真正使用的参数      | 尽量只保留执行需要的信息        |
| `to_underlying_arguments` | 把 `Arguments` 转成 `Params` | 需要 workspace 时在这里完成映射 |
| `get_workspace_size`      | 声明节点自有 workspace       | 只计算自己那一段                |
| `can_implement`           | 轻量合法性检查               | 至少检查容易误用的参数          |
| `Callbacks::visit`        | 执行节点动作                 | 严格按阶段写逻辑                |
| `get_callbacks`           | 申请 UB 并构造回调           | 谁申请 UB，谁推进 `ub_offset`   |

### 当前约束

#### 阶段语义不能打乱

节点逻辑按三阶段组织：

- `LOAD`：准备输入、从 GM 读入 UB
- `COMPUTE`：UB 内计算、变换、广播
- `STORE`：把结果真正写出

明确规则：

- 真正写回外部地址的动作放在 `STORE`
- 跨阶段节点可以存在，但每一步分开放置
- 本该在 `STORE` 的动作通常不前置到 `COMPUTE`

#### 节点职责保持单一

当前实现更适合让一个节点只承担一类职责，例如读、算、写、广播。
“读 GM + 算 + 写 GM”这类复合行为拆成多个节点后，更贴合现有图组织方式。

#### layout 一律按完整张量理解

如果节点带 `layout`，它描述的是完整 GM 张量，不是当前 tile。当前实现都按这个口径处理，新增节点时保持一致即可。

#### UB 分配口径保持一致

`get_callbacks` 中如果申请 UB，通常按下面的口径处理：

- 谁申请，谁推进 `ub_offset`
- 只申请自己需要的那一段
- 大小按 `compute_length` 和元素大小推导
- 分配结果不超过当前架构允许的 UB 上限

#### 不在节点里接管外层事件

EVG 的双缓冲和事件同步由 `BlockEpilogue` 统一管理。节点内部只负责按阶段执行；如果节点内有多步 V 计算，可以补 `AscendC::PipeBarrier<PIPE_V>()`，外层同步节奏仍由 `BlockEpilogue` 管理。

#### `Arguments` 继续支持直写

新增节点后，用户侧通常仍可直接写出整张图的聚合初始化：

```cpp
typename EVG::Arguments args{
    {},
    {deviceX, layoutX},
    {{2.0f}},
    {deviceD, layoutD}
};
```

如果一个节点让整张图的参数初始化显著变复杂，通常说明接口设计和现有规范还有些距离。

#### 先复用现有节点，再决定是否新增

如果诉求只是多一个逐元素算子，通常继续走 `ComputeFn + VisitorCompute`。
只有当现有节点无法表达所需的数据访问、layout 或资源行为时，再新增节点。

### `visit` 签名

`visit` 可以写成可变参数，也可以写成固定签名，取决于节点职责：

- 叶子节点、未来可能扩展输入路数的节点：可用可变参数
- 严格单输入或固定多输入节点：可以直接写死输入签名

关键不在于形式统一，而在于节点自身要明确：

- 需要几路输入
- 输入顺序是什么
- 输入类型是否符合预期

### 实施顺序

新增 `ComputeFn` 时：

1. 在 `operations.hpp` 中新增算子
2. 保持 `operator()(dst, compute_length, inputs...)` 这一形态
3. 多步 V 计算时补 `PipeBarrier`
4. 用 `VisitorCompute<..., ElementType, Scalars...>` 接入
5. 用最小图验证参数和类型口径

新增节点时：

1. 新建 `visitor_xxx.hpp`
2. 继承 `VisitorImpl<>`
3. 定义 `ElementOutput / Arguments / Params`
4. 实现 `to_underlying_arguments / get_workspace_size / can_implement`
5. 在 `Callbacks::visit` 中按阶段填充逻辑
6. 在 `get_callbacks` 中完成 UB 申请
7. 接入 `fusion.hpp`
8. 用最小样例验证能在 `TreeVisitor` 或 `TopologicalVisitor` 中组合

## 实现前检查清单

新增 `ComputeFn` 前，至少确认：

- 它是不是纯逐元素计算
- 是否真的不需要 GM、layout、workspace
- 是否有多步 V 计算需要 barrier
- 是否没必要新增节点

新增节点前，至少确认：

- 不能只靠 `ComputeFn` 解决
- `Arguments` 能否保持简洁并支持花括号初始化
- `LOAD / COMPUTE / STORE` 是否分工明确
- 是否需要 UB 或 workspace
- layout 是否按完整张量解释
- 是否补了基本的 `can_implement`

## 参考实现

扩展时，可直接对照这些现有实现：

- `include/catlass/epilogue/fusion/operations.hpp`
- `include/catlass/epilogue/fusion/visitor_compute.hpp`
- `include/catlass/epilogue/fusion/visitor_aux_load.hpp`
- `include/catlass/epilogue/fusion/visitor_aux_store.hpp`
- `include/catlass/epilogue/fusion/visitor_row_broadcast.hpp`

可按下面顺序对照：

- 只加逐元素算子：先看 `operations.hpp`
- 加读写类节点：先看 `visitor_aux_load.hpp`、`visitor_aux_store.hpp`
- 加跨阶段节点：先看 `visitor_row_broadcast.hpp`
