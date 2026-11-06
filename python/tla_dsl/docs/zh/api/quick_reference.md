---
nav_order: 5
---

# CATLASS Python API 速查

本页按客户任务组织常用 API，不是完整接口清单。完整签名与约束见
[程序生命周期参考](runtime/index.md) 与 [语言参考](language/index.md)。

上下文标记：**Host** 表示在 `@tla.kernel` 外执行的 Python 脚本；**Kernel** 表示
`@tla.kernel` 函数体；**Cube** / **Vector** 分别指 `tla.cube()` / `tla.vector()`
区域；**SIMD** / **SIMT** 指 `tla.vec.func` 的对应模式。

## 1. 开始使用

| 入口 | 用途 | 上下文或关键约束 |
| --- | --- | --- |
| [快速开始](../quick_start.md) | 安装软件包并运行第一个 kernel。 | Host；先确认产品、CANN 与 Python 版本要求。 |
| [语法约束](../kernel_development/core_concepts/syntax_guide.md) | 区分编译期值与展开、运行时值与控制流。 | Kernel 与 `@tla.jit` helper 只支持文档列出的 Python 写法。 |
| [Layout](../kernel_development/core_concepts/layout.md) | 理解静态 / 动态 layout、逻辑尺寸与物理存放。 | Host 侧 layout 元数据参与编译；Kernel 读取生成的 layout。 |
| [编译与启动](../kernel_development/core_concepts/compile_and_launch.md) | 按编译后启动的流程使用 kernel。 | Host；编译使用类型样本，启动使用绑定真实 NPU 缓冲的 tensor。 |

## 2. Host 生命周期

| API | 用途 | 上下文或关键约束 |
| --- | --- | --- |
| [`@tla.kernel`](runtime/kernel_entry.md#tlakernel) | 声明 kernel 入口并得到 `TlaJitFunction`。 | Host 装饰器；函数体不在 Host 执行，且禁止直接调用。启动前须编译。 |
| [`@tla.jit`](runtime/kernel_entry.md#tlajit) | 声明 kernel lowering 期间内联的 device helper。 | 在 Host 声明；无独立 `compile` / `dump_mlir`，helper 之间不可递归调用。 |
| [`@tla.extern`](runtime/kernel_entry.md#tlaextern) | 声明由内联源码实现的 Ascend C ABI 入口。 | Host 声明；调用位于单个 Cube 或 Vector 区域，且不在 `tla.vec.func` 内。 |
| [`tla.compile`](runtime/compile_launch.md#tlacompile) | 从类型样本编译 kernel，返回 `JitCompiledFunction`。 | Host；类型样本不必绑定 NPU 缓冲。`block_num` / `stream` 等启动参数写在编译结果上。 |
| [`tla.JitCompiledFunction.__call__`](runtime/compile_launch.md#tlajitcompiledfunction__call__) | 启动已编译 kernel，并复用 executor 与二进制。 | Host；tensor 实参须为已绑定的 NPU 缓冲，`make_fake_tensor` 结果不能启动。 |
| [`tla.from_dlpack`](runtime/host_tensor.md#tlafrom_dlpack) | 零拷贝绑定 DLPack tensor 为 TLA Host tensor。 | Host；`layout_tag` 必填。NPU 缓冲可启动，CPU 缓冲只提供编译元数据。 |
| [`tla.make_tensor`](runtime/host_tensor.md#tlamake_tensor) | Host 双模式：由 `tla.make_ptr` + `tla.make_layout` 绑定缓冲。 | Host 路径跳过 DLPack；Kernel 内仍发 IR。 |
| [`tla.make_fake_tensor`](runtime/host_tensor.md#tlamake_fake_tensor) | 创建仅含元数据的编译样本。 | Host；无设备缓冲，不能启动。shape 输入是 Python 整数树，不是 Kernel layout 值。 |

## 3. Tensor、Layout 与 Pointer

| API | 用途 | 上下文或关键约束 |
| --- | --- | --- |
| [`tla.make_shape`](language/layout_tensor.md#tlamake_shape) | 构造逻辑或嵌套物理 shape。 | Kernel / Host 同类型；`zN` 等使用显式嵌套物理 shape。 |
| [`tla.make_coord`](language/layout_tensor.md#tlamake_coord) | 构造 coord 树。 | Kernel / Host 同类型；坐标语义由消费接口决定。 |
| [`tla.make_stride`](language/layout_tensor.md#tlamake_stride) | 构造 stride 树。 | Kernel / Host 同类型；嵌套与数值须匹配 shape 与 layout tag。 |
| [`tla.make_layout`](language/layout_tensor.md#tlamake_layout) | 组合 shape、stride、逻辑 origin 与 layout tag。 | Kernel / Host 同类型；Host 用 `make_shape` / `make_stride`。 |
| [`tla.make_layout_with_tag`](language/layout_tensor.md#tlamake_layout_with_tag) | 按元素类型与 layout tag 重映射二维逻辑尺寸。 | Kernel；仅限片上 layout，供 `make_tensor` 使用。 |
| [`tla.tile_view`](language/layout_tensor.md#tlatile_view) | 不拷贝源缓冲创建 tile 视图。 | Kernel；坐标使用 tile 粒度。 |
| [`tla.get_tile`](language/layout_tensor.md#tlaget_tile) | 从源缓冲切出扁平二维子 tile。 | Kernel；坐标为元素偏移。裁剪只收窄尺寸，不限制越界起点。 |
| [`tla.make_tensor`](language/layout_tensor.md#tlamake_tensor) | 组合 pointer、layout 与可选 coord。 | Kernel；pointer dtype 与地址空间须匹配，可运行 kernel 须有底层存储。 |
| [`tla.make_tensor_like`](language/layout_tensor.md#tlamake_tensor_like) | 在另一 pointer 上复用参考 tile 元数据，可改 layout tag。 | Kernel；目标 pointer 指向片上。 |
| [`tla.make_ptr`](language/layout_tensor.md#tlamake_ptr) | 从地址构造带类型的 pointer。 | Kernel / Host 同类型；Host 接受 Python int 地址。 |
| [`tla.recast_ptr`](language/layout_tensor.md#tlarecast_ptr) | 重解释 pointer 的元素类型。 | Kernel；地址与 swizzle 保持不变。 |

## 4. 数据搬运与矩阵计算

| API | 用途 | 上下文或关键约束 |
| --- | --- | --- |
| [`tla.copy`](language/data_movement.md#tlacopy) | 在支持的全局与片上通路间拷贝 tile。 | Kernel / Cube 或 Vector；受通路、layout、dtype、对齐与 origin shape 约束，SIMT 内不可用。 |
| [`tla.Tensor.load`](language/data_movement.md#tlatensorload) | 将 UB tile 加载为 vector 或 mask SSA。 | Kernel / Vector / SIMD；须在 `tla.vec.func` 内，源为 UB tile。 |
| [`tla.Tensor.store`](language/data_movement.md#tlatensorstore) | 将 vector 或 mask SSA 存入 UB tile。 | Kernel / Vector / SIMD；须在 `tla.vec.func` 内，目标为 UB tile。 |
| [`tla.Tensor.fill`](language/data_movement.md#tlatensorfill) | 用零填充 L1 tile。 | Kernel / Cube；目标须为 L1 `zN` 或 `nZ`，元素类型须为受支持的打包 fp4/fp8。 |
| [`tla.Tensor.__getitem__`](language/data_movement.md#tlatensor__getitem__) | 读取一个元素为标量。 | Kernel；GM 或 UB，`RowMajor` 或 `ColumnMajor`，逻辑秩 1 或 2。UB 访问须在 Vector 内；SIMT 使用逐线程元素访问。 |
| [`tla.Tensor.__setitem__`](language/data_movement.md#tlatensor__setitem__) | 写入一个标量元素。 | Kernel；tensor 限制同元素读取。`Numeric` 值须与 tensor 元素类型完全一致。 |
| [`tla.mmad`](language/matrix.md#tlammad) | 矩阵 tile 乘法累加。 | Kernel / Cube；累加器与操作数须为匹配的 L0 tile，dtype 组合受限。 |
| [`tla.mmad_mx`](language/matrix.md#tlammad_mx) | 微缩放矩阵 tile 乘法累加。 | Kernel / Cube；匹配 L0 tile、fp32 累加器，两个操作数均由 `tla.copy(..., scale=...)` 加载。 |

## 5. SIMD 与 SIMT 向量计算

两种模式都在 `tla.vector()` 内的 `tla.vec.func` 中执行：

- **SIMD** 作用于整向量寄存器与 mask。数据先搬到 UB，再使用 `Tensor.load` /
  `Tensor.store`，适合规则稠密向量计算。
- **SIMT** 以标量算术与元素访问表达单线程程序。使用 `thread_block_dim`，
  只允许捕获标量运行期值；区域内不允许寄存器向量、mask 操作、
  `Tensor.load` / `Tensor.store` 或 `tla.copy`。

| API | 用途 | 上下文或关键约束 |
| --- | --- | --- |
| [`tla.vec.func`](language/scopes.md#tlavecfunc) | 选择 `simd` 或 `simt` 向量执行模式。 | Kernel / Vector；SIMT 需 `thread_block_dim`，每 block 至多 2048 线程；SIMD 拒绝该参数。 |
| [`tla.create_mask`](language/simd/mask.md#tlacreate_mask)、[`tla.update_mask`](language/simd/mask.md#tlaupdate_mask) | 创建固定模式或尾部 mask。 | Kernel / Vector / SIMD；SIMT 无寄存器向量 mask。 |
| [`tla.add`](language/simd/arithmetic.md#tlaadd)、[`tla.sub`](language/simd/arithmetic.md#tlasub)、[`tla.mul`](language/simd/arithmetic.md#tlamul)、[`tla.div`](language/simd/arithmetic.md#tladiv) | 逐元素算术。 | Kernel / Vector；SIMD 使用 `VectorSSA`，SIMT 使用逐线程 `Numeric` 标量；mask 只作用于向量形态。 |
| [`tla.max`](language/simd/arithmetic.md#tlamax)、[`tla.min`](language/simd/arithmetic.md#tlamin) | 逐元素最大值或最小值。 | Kernel / Vector；SIMD 向量与 SIMT 标量形态各有 dtype 规则。 |
| [`tla.exp`](language/simd/arithmetic.md#tlaexp)、[`tla.log`](language/simd/arithmetic.md#tlalog)、[`tla.sqrt`](language/simd/arithmetic.md#tlasqrt)、[`tla.abs`](language/simd/arithmetic.md#tlaabs) | 一元数学运算。 | Kernel / Vector；SIMD 使用向量，SIMT 使用标量。标量 `exp` / `log` 仅 SIMT 可用。 |
| [`tla.cmp`](language/simd/compare_select.md#tlacmp) | 比较向量元素并产生 `MaskSSA`。 | Kernel / Vector / SIMD；SIMT 使用标量比较运算符并产生 `Bool`。 |
| [`tla.where`](language/simd/compare_select.md#tlawhere) | 按条件选择两个值。 | Kernel / Vector；SIMD 条件为 `MaskSSA`，SIMT 使用逐线程 `Bool`。 |
| [`tla.full`](language/simd/fill.md#tlafull)、[`tla.arange`](language/simd/fill.md#tlaarange) | 创建常量向量或 lane 索引序列。 | Kernel / Vector / SIMD；SIMT 拒绝这些寄存器向量构造接口。 |
| [`tla.VectorSSA.reduce`](language/simd/discrete_aggregate.md#tlavectorssareduce) | 在 mask 下归约为单 lane。 | Kernel / Vector / SIMD；必须提供 mask，可用 `full` 广播该结果。 |
| [`tla.VectorSSA.to`](language/simd/type_conversion.md#tlavectorssato) | 转换寄存器向量元素类型。 | Kernel / Vector / SIMD；支持类型见完整条目。 |
| [`tla.Numeric.to`](language/values_types.md#tlanumericto) | 转换标量元素类型。 | Kernel；SIMT 中生成逐线程转换。寄存器向量 `tla.cast` 没有逐线程等价接口。 |

## 6. 同步、资源与调试

### 同步

不确定什么时候需要同步、该选 flag 还是 mutex 时，先看
[同步使用指南](../kernel_development/core_concepts/sync.md)。

| API | 用途 | 上下文或关键约束 |
| --- | --- | --- |
| [`tla.flag`](language/sync.md#tlaflag)、[`tla.set_flag`](language/sync.md#tlaset_flag)、[`tla.wait_flag`](language/sync.md#tlawait_flag) | 在核内生产者与消费者流水间同步。 | Kernel / Cube 或 Vector；set 与 wait 须成对作用于同一 flag。 |
| [`tla.cross_flag`](language/sync.md#tlacross_flag)、[`tla.cross_core_set_flag`](language/sync.md#tlacross_core_set_flag)、[`tla.cross_core_wait_flag`](language/sync.md#tlacross_core_wait_flag) | 跨核同步。 | Kernel / Cube 或 Vector；set 与 wait 须指定 pipe；mode 4 要求 `aiv_id` 为 0 或 1。 |
| [`tla.pipe_barrier`](language/sync.md#tlapipe_barrier) | 在硬件流水上插入 barrier。 | Kernel / Cube 或 Vector；SIMT 内不可用。 |
| [`tla.mutex`](language/sync.md#tlamutex)、[`tla.mutex_guard`](language/sync.md#tlamutex_guard)、[`tla.mutex_lock`](language/sync.md#tlamutex_lock)、[`tla.mutex_unlock`](language/sync.md#tlamutex_unlock) | 以显式加锁或作用域 guard 保护命名资源。 | Kernel / Cube 或 Vector；guard 块须产生 `copy` 或 `mmad`，且不得显式加锁或解锁。 |
| [`tla.local_mem_bar`](language/sync.md#tlalocal_mem_bar) | 在本地存储类型之间排序访问。 | Kernel / Vector / SIMD；须在 `tla.vec.func` 内并使用受支持的 `MemType` 组合；SIMT 不可用。 |
| [`tla.arch.sync_threads`](language/system.md#tlaarchsync_threads) | 同步一个 SIMT block 内的线程。 | Kernel / Vector / SIMT；所有线程都须到达 barrier；它不是跨核或 pipe barrier。 |

### 资源与调试

| API | 用途 | 上下文或关键约束 |
| --- | --- | --- |
| [`tla.allocate`](language/resources.md#tlaallocate) | 分配片上 scratch 并返回带类型 pointer。 | Kernel；仅支持静态 shape 与片上地址空间。SIMT scratch 在区域外分配。 |
| [`tla.arch.block_idx`](language/system.md#tlaarchblock_idx)、[`tla.arch.block_num`](language/system.md#tlaarchblock_num) | 读取启动 block 索引或 block 数。 | Kernel；描述 AI Core block，不是 SIMT 线程。 |
| [`tla.arch.thread_idx`](language/system.md#tlaarchthread_idx)、[`tla.arch.thread_block_dim`](language/system.md#tlaarchthread_block_dim) | 读取 SIMT 线程坐标或线程 block 尺寸。 | Kernel / Vector / SIMT。 |
| [`tla.arch.get_capacity_in_bytes`](language/system.md#tlaarchget_capacity_in_bytes) | 查询编译目标的片上存储容量。 | Host 或 Kernel；作为编译期常量折叠。仅限片上地址空间；UB 可用量仍受编译器预留影响。 |
| [`tla.arch.get_dyn_ub`](language/system.md#tlaarchget_dyn_ub) | 获取启动时指定大小的动态 UB pointer。 | Kernel；每 kernel 至多一个区域。启动传 `ub=` 指定字节数；kernel 内尺寸不做边界检查。 |
| [`tla.TlaJitFunction.dump_mlir`](runtime/compile_launch.md#tlatlajitfunctiondump_mlir) | 返回前端 TLA IR，不构建或启动二进制。 | Host；类型样本规则与编译相同。 |
| [`tla.print`](language/debug.md#tlaprint) | 打印标量、格式化字符串或 Tensor／寄存器前缀。 | Kernel / Cube 或 Vector，包括 `vec.func`。SIMD：标量／寄存器／UB；SIMT：标量／GM／UB。dtype、长度和同步规则见 API 说明。 |
