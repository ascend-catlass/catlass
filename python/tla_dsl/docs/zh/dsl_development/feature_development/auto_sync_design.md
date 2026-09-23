---
nav_order: 40
---

# AutoSync 设计

## 背景与目标

Ascend AI Core 上的 MTE、Cube、Vector、Fixpipe 等流水可以异步执行。同一块片上存储被不同流水连续读写时，如果没有正确的同步，会产生 RAW、WAR 或 WAW 数据竞争。手工同步要求开发者同时维护 tensor 的存储关系、指令所属 pipe、双缓冲切换和 `unit_flag` 协议，代码量大，也容易因算法调整而遗漏。

TLA DSL 通过 `@tla.kernel(auto_sync="v0")` 提供实验性自动同步能力。

`v0` 只处理单个 AIC 或 AIV 内部各流水之间的核内同步，不生成 Cube/Vector 核间同步，也不生成Vector核编程中VF内的同步。


## 用户接口与使用约束

### 启用方式

启用方式如下：

```python
@tla.kernel(auto_sync="v0")
def kernel_func(...):
    # 不再手写set/wait_flag、mutex 或 mutex_guard。
    ...
```

当前公开接口只接受 `None` 和 `"v0"`：

| 配置 | 行为 |
|------|------|
| `@tla.kernel` / `@tla.kernel()` | 不启用 AutoSync，核内同步由开发者显式描述。 |
| `@tla.kernel(auto_sync="v0")` | 启用自动核内同步。 |

完整端到端样例见 [basic_matmul_auto_sync.py](../../../../examples/end_to_end/basic_mmad/basic_matmul_auto_sync.py)。该样例使用 L1 和 L0 双缓冲，kernel 中未显式编写同步原语。

### 使用约束

`auto_sync="v0"` 采用保守的静态分析策略。开发者需要遵循以下约束；当编译器无法证明同步安全时，会直接终止编译并给出诊断。

| 约束项 | 开发者需要遵循的规则 |
|--------|----------------------|
| 适用入口 | `auto_sync` 仅支持 `@tla.kernel`，不能用于 `@tla.jit` 或其他函数。当前仅支持版本值 `"v0"`。 |
| 同步范围 | 只自动处理单个 AIC 或 AIV 内部不同硬件流水之间的核内同步。AIC/AIV 核间同步仍需显式使用 `cross_flag` 和 `cross_core_*`；`tla.vec.func` 内部需要的线程级同步不会自动生成。 |
| 手写同步 | 启用后不能在同一个 kernel 中使用核内 `tla.flag`、`tla.set_flag`、`tla.wait_flag`、`tla.mutex` 或 `tla.mutex_guard`，否则编译失败。 |
| 指令覆盖 | `v0` 只分析 `tla.copy`、`tla.mmad` 和 `tla.vec.func` 内受支持的 UB 访问。当前 AutoSync 不支持将 UB `scalar_load` / `scalar_store` 直接写在 `tla.vector` 下；启用 `v0` 时须将其放在 `tla.vec.func` 内。不启用 AutoSync 时，这类写法仍是合法的。`tla.print_tensor`、`tla.debug_print` 不会获得自动同步。 |
| 片上内存来源 | 被自动同步保护的片上 tensor 必须能够回溯到由 `tla.allocate` 生成的静态 allocation root。允许基于该 root 构造 recast、offset、tensor 和 tile view；不支持通过 `tla.make_ptr` 从片上裸地址构造、因而无法回溯到 allocation root 的 tensor。 |
| 外部调用 | `v0` 不能与 `tla.call_extern` 组合使用，因为编译器无法分析外部函数内部的异步访存和完成语义。 |
| 动态 buffer | 支持通过运行时条件选择多个由 `tla.allocate` 创建的 buffer。当前 `v0` 不支持把片上 pointer 作为循环状态，并在迭代过程中将它切换到另一块 allocation；同一条指令使用多个条件选择的 buffer 时，各分支还必须保持一致的 allocation 顺序。 |
| `unit_flag` | `tla.mmad` 只接受可静态证明为 0 或始终位于 `{2, 3}` 的 `unit_flag`；L0C copy 的 `CopyL0C2DstParams.unit_flag` 只支持 0 或 3。运行时在 0 与 2/3 之间切换不受支持。 |

如果 kernel 无法满足上述约束，应保持 `auto_sync=None`，并由开发者显式描述所需的核内及核间同步。

## 设计边界

### `v0` 负责什么

- `tla.copy` 涉及的片上源、目的 tensor；
- `tla.mmad` 的 L0A、L0B 和按 `unit_flag` 规则决定是否纳入的 L0C；
- 一个 `tla.vec.func` 内 `tla.load`、`tla.store`、`tla.gather`、`tla.scalar_load`、`tla.scalar_store` 访问的 UB tensor；
- 同一 `tla.alloc_ptr` 派生出的 recast、offset、tensor 和 tile view 别名；
- 可静态追踪的运行时条件 buffer 选择。

### `v0` 不负责什么

- AIC 与 AIV 之间的核间同步；
- `tla.call_extern` 内部可能发生的异步访问；
- `tla.print_tensor` 和 `tla.debug_print` 的同步；
- 裸片上地址或无法回溯到静态 `tla.alloc_ptr` 的指针；
- 把片上 pointer 作为循环状态，并在迭代过程中将它切换到另一块 allocation；
- 手写核内同步与自动核内同步混用。

因此，启用 `v0` 后禁止在同一个 kernel 中使用核内 `mutex`、`mutex_guard`、`flag`、`set_flag` 或 `wait_flag`。

核间 `cross_flag`、`cross_core_set_flag` 和 `cross_core_wait_flag` 不受此限制。

## 编译链路

AutoSync 的控制信息从 Python 装饰器一路传到 MLIR pass：

```text
@tla.kernel(auto_sync="v0")
          │
          │ catlass/dsl.py 校验并保存 options
          ▼
execution_lowering.py 构造 tla.func
          │ 添加 {tla.auto_sync = "v0"}
          ▼
TlaLowerFuncPass
          │ tla.func -> func.func，保留属性并确定 AIC/AIV/MIX 类型
          ▼
TlaInsertAutoMutexPass
          │ 分析 allocation root、指令资源和 pipe
          │ 插入 tla.mutex / lock / unlock，移除 tla.auto_sync 属性
          ▼
TlaLowerPtrPass / TlaSplitMixedFuncPass / region lowering
          ▼
TlaLowerMutexToStdPass
          │ !tla.mutex -> i8 ID
          │ lock/unlock -> get_buf_<pipe> / rls_buf_<pipe> runtime call
          ▼
后端编译与 bitcode 链接
```

关键顺序是 `TlaInsertAutoMutexPass` 位于 `TlaLowerFuncPass` 之后、`TlaLowerPtrPass` 和 `TlaSplitMixedFuncPass` 之前：

- `TlaLowerFuncPass` 已经根据 `tla.cube` / `tla.vector` region 得到函数 core 类型；
- `tla.alloc_ptr`、tensor/view 链和结构化控制流尚未被破坏，适合做资源溯源；
- MIX kernel 尚未拆成 AIC/AIV 两个函数，pass 可以在同一份原始结构上分别建立 Cube 和 Vector 的 mutex ID 空间。

对应实现入口：

- Python 接口：[catlass/dsl.py](../../../../catlass/dsl.py)
- 属性生成：[catlass/execution_lowering.py](../../../../catlass/execution_lowering.py)
- pass 管线：[PassRegistry.cpp](../../../../csrc/mlir/lib/Passes/PassRegistry.cpp)
- AutoSync 主体：[TlaInsertAutoMutexPass.cpp](../../../../csrc/mlir/lib/Passes/TlaInsertAutoMutexPass.cpp)
- mutex lowering：[TlaLowerMutexToStdPass.cpp](../../../../csrc/mlir/lib/Passes/TlaLowerMutexToStdPass.cpp)

## 核心模型

### allocation root 是同步资源的身份

AutoSync 不以 tensor SSA value 作为资源身份，而以片上内存的 `tla.alloc_ptr` root 作为身份。原因是多个 tensor/view 可能引用同一块物理存储：

```text
%root = tla.alloc_ptr ...
   ├── tla.recast_ptr
   ├── tla.ptr_add
   └── tla.make_tensor / tla.make_tensor_like
          └── tla.tile_view
```

这些派生值最终都解析到 `%root`，共享同一个 mutex。tensor 的动态 shape、stride 或 coord 不改变 allocation root，因此不会因为 view 形态不同而重复分配 mutex。

`TlaScratchAllocation` 为每个 root 记录：

- address space；
- alignment；
- 静态分配的 `[base, end)` 字节区间；
- allocation size。

AutoSync 与后续 `TlaLowerPtrPass` 共用 `planTlaScratchAllocations()` 的结果，保证 mutex 描述的资源与最终 lower 出的物理片上地址一致。

GM 和 generic memref 被视为全局资源，不分配核内 mutex；片上资源如果无法解析到有效 root，则编译失败。

### 指令是加锁边界

每个支持的异步指令形成一条 `InstructionPlan`：

```text
InstructionPlan = {
    op,          // 加锁范围对应的指令
    pipe,        // 执行该指令的硬件流水
    idSpace,     // Cube 或 Vector mutex ID 空间
    resources,   // 指令访问的片上 allocation root
}
```

资源收集规则如下：

| 指令边界 | pipe | 纳入的资源 |
|----------|------|------------|
| `tla.copy`，源为 GM | `mte2` | 片上 dst |
| `tla.copy`，源为 L1 | `mte1` | 片上 src、dst |
| `tla.copy`，源为 UB | `mte3` | 片上 src、dst |
| `tla.copy`，源为 L0C | `fix` | 片上 src、dst，以及作为 tensor 传入的 quant scale；`unit_flag=3` 时 L0C 由 unit flag 协议负责，不重复加锁 |
| `tla.mmad` | `cube` | L0A、L0B；未启用 unit flag 时还包括 L0C accumulator |
| `tla.vec.func` | `vector` | region 内 load/store/gather/scalar load/store 访问的 UB root，去重后统一加锁 |

GM operand 会在资源规范化阶段被过滤。单条指令多次访问同一个 root 时只加一次锁。

### 插入形式

假设一次 L1 到 L0A 的 copy 同时访问 `%l1` 和 `%l0a`，pass 转换前后的核心结构为：

```mlir
// before
tla.copy %l0a_tensor, %l1_tensor

// after（示意）
%l1_mutex = tla.mutex "auto_l1_0_4096" {id = 0 : i64}
%l0a_mutex = tla.mutex "auto_l0a_0_4096" {id = 1 : i64}
tla.mutex_lock %l1_mutex[<mte1>]
tla.mutex_lock %l0a_mutex[<mte1>]
tla.copy %l0a_tensor, %l1_tensor
tla.mutex_unlock %l0a_mutex[<mte1>]
tla.mutex_unlock %l1_mutex[<mte1>]
```

mutex 声明统一放在函数入口。锁按稳定 ID 顺序获取，按相反顺序释放，形成栈式顺序，避免多资源指令之间产生锁序反转。

## mutex ID 分配

### Cube 与 Vector 使用独立空间

硬件 mutex ID 按 core side 独立管理。AutoSync 为 Cube 和 Vector 分别收集实际使用的 allocation root，并分别从 0 分配 ID。一个 MIX kernel 可以同时拥有 Cube 侧 `id=0` 和 Vector 侧 `id=0`，两者不是同一个跨核同步对象。

每侧最多使用 32 个自动 mutex ID。超过限制会在编译期报错。

### 确定性排序

同一侧的资源先按 address space 排序，再按静态 base 排序。address space 顺序固定为：

```text
L1 < L0A < L0B < L0C < UB
```

确定性排序有两个作用：

- IR 和测试结果不依赖遍历或哈希表顺序；
- 所有多资源指令共享统一锁序，降低死锁风险。

mutex 的调试名称为 `auto_<address-space>_<base>_<size>`，例如 `auto_ub_256_256`。名称用于可读性，真正传给 runtime 的是数值 ID。

## 动态控制流

### `scf.if` 选择

双缓冲通常会根据运行时条件选择不同 allocation：

```python
buf = buf0_ptr if index == 0 else buf1_ptr
tensor = tla.make_tensor_like(buf, source)
tla.copy(tensor, source)
```

lower 到 `scf.if` 后，pointer 的真实 root 只能在运行时确定。AutoSync 不会保守地同时锁住两个候选 mutex，而是给原 `scf.if` 增加一个并行的 `!tla.mutex` result：

```text
                  then: (buf0_ptr, mutex0)
condition ─ scf.if
                  else: (buf1_ptr, mutex1)
                             │
                             └── copy 使用被选中的 mutex
```

mutex 与 pointer 沿完全相同的控制流传播，既保留精确性，也避免在指令位置重新构造 branch-local condition 所造成的 dominance 问题。嵌套 `scf.if` 由内向外物化对应的 mutex result。

### 循环携带值

对于 `scf.for` 的 iter argument/result，只有初始值和 backedge 能解析为同一 storage expression 时才认为 root 稳定。循环每次迭代切换不同 root、或 provenance 无法证明一致时，`v0` 会拒绝编译。

### 多个动态 buffer 的组合约束

一条指令同时访问多个由运行时条件选择的 buffer 时，各分支必须保持一致的 allocation 顺序。例如，一个分支选择 `buffer0` 作为 src、`buffer1` 作为 dst，另一个分支反过来选择 `buffer1` 作为 src、`buffer0` 作为 dst，当前 `v0` 不支持这种写法。原因是 AutoSync 无法为所有运行时分支生成唯一且一致的 mutex 加锁顺序。这是 AutoSync 的当前能力限制，不是 TLA DSL 对条件 buffer 的通用限制。

## `unit_flag` 协同

MMAD/FIX 的 `unit_flag` 已经承担一部分 L0C 流水握手。AutoSync 必须避免对同一资源重复建立互斥协议。

### MMAD

- `unit_flag=0`：L0C accumulator 与 L0A、L0B 一起纳入 mutex；
- `unit_flag=2` 或 `3`：L0C 由 unit flag 协议保护，AutoSync 只锁 L0A、L0B；
- 允许通过 cast、`arith.select` 或 `scf.if` 得到 unit flag，但必须能静态证明其始终为 0，或始终位于 `{2, 3}`；
- 运行时可能在 0 与 2/3 之间切换时无法确定是否需要 L0C mutex，因此报错。

### L0C copy

`CopyL0C2DstParams.unit_flag` 只支持 0 或 3：

- `unit_flag=0`：L0C 和目的片上资源都按普通规则加锁；
- `unit_flag=3`：L0C 侧依赖 unit flag 协议，AutoSync 不再锁 L0C；目的片上资源仍需加锁。

这样 MMAD 的 enabled unit flag 与 Fixpipe 的 `unit_flag=3` 可以组成一套完整的 L0C 生产/消费协议，核内 mutex 负责剩余资源。

## 保守失败策略与已知限制

`v0` 的原则是“证明安全后生成，否则失败”。典型限制和诊断如下：

| 场景 | 原因/处理 |
|------|-----------|
| 混用手写 local flag/mutex | 自动与手动协议的相互作用不明确，拒绝编译。 |
| 使用 `tla.call_extern` | pass 看不到 extern 内部访问及完成语义，拒绝编译。 |
| 受保护的片上 tensor 由 `tla.make_ptr` 从裸地址构造 | 无法回溯到 allocation root，AutoSync 不能判断其存储归属和别名关系，拒绝编译。 |
| 启用 `v0` 时，在循环迭代间把片上 pointer 切换到另一块 allocation | 当前 AutoSync 无法为该循环状态确定唯一的受保护 buffer，拒绝编译。 |
| 启用 `v0` 时，将 UB `scalar_load` / `scalar_store` 直接写在 `tla.vector` 下 | 当前 AutoSync 不支持这类写法；不启用 AutoSync 时仍可合法使用。 |
| 同一条指令使用多个条件选择的 buffer，且不同分支的 allocation 顺序不一致 | 当前 AutoSync 无法生成适用于所有分支的一致加锁顺序，拒绝编译。 |

这些限制刻意保持 `v0` 的语义简单。后续版本若扩大覆盖范围，应通过新的版本值显式演进，而不是静默改变 `v0` 的含义。

## 测试设计

AutoSync 的测试分成三层：

1. Python 前端测试验证参数校验、默认关闭和 `tla.auto_sync` 属性传递，见 [test_local_autosync.py](../../../../tests/frontend/ops/sync/test_local_autosync.py)。
2. MLIR lit 测试直接检查 pass 输出和失败诊断：
   - [auto-mutex-instruction-pipes.mlir](../../../../tests/lit/sync/mutex/auto-mutex-instruction-pipes.mlir)：指令资源、pipe、锁序和 unit flag；
   - [auto-mutex-alias-root.mlir](../../../../tests/lit/sync/mutex/auto-mutex-alias-root.mlir)：别名 root；
   - [auto-mutex-dynamic-select.mlir](../../../../tests/lit/sync/mutex/auto-mutex-dynamic-select.mlir)：动态选择；
   - [auto-mutex-id-spaces.test](../../../../tests/lit/sync/mutex/auto-mutex-id-spaces.test)：独立 ID 空间和 32-ID 上限；
   - [auto-mutex-diagnostics.test](../../../../tests/lit/sync/mutex/auto-mutex-diagnostics.test)：保守失败路径。
3. `basic_matmul_auto_sync.py` 端到端验证双缓冲、MMAD/FIX unit flag 和最终数值结果。

## 扩展 AutoSync 时的检查项

新增指令或扩大自动同步覆盖范围时，至少需要回答以下问题：

1. 指令级同步边界是什么，异步完成发生在哪个 pipe？
2. 哪些 operand 是读、写或读写片上资源，是否存在隐含 tensor operand？
3. 资源能否沿所有合法 tensor/pointer 构造回溯到 `tla.alloc_ptr` root？
4. 指令是否已有 unit flag、barrier 或其他硬件握手，需要避免重复同步？
5. AIC、AIV 和 MIX kernel 中的 mutex ID 空间如何确定？
6. 单条指令涉及多个资源时，能否维持全局稳定锁序？
7. `scf.if` 和 `scf.for` 下的 provenance 是否仍可证明？
8. 不支持的输入是否有明确诊断，并有对应 negative lit test？

实现上应优先把新规则加入 `collectInstructionPlans()`，并保持 storage provenance、ID 分配和 lock materialization 三层职责分离，避免把某个 op 的特殊语义扩散到通用资源分析中。
