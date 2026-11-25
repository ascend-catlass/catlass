---
nav_order: 8
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/system.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 系统变量访问

[语言参考](index.md) | [API 速查](../quick_reference.md)

`tla.arch` / `tla.AddressSpace` 上的架构属性（布局标签、pipe、片上地址空间、
block 辅助等）。

## 布局标签

`tla.arch` 上的布局标签（`_LayoutTag`），供 `make_layout` / `make_tensor` /
`make_tensor_like` / `copy` 使用，作为 `layoutTag=` / `layout_tag=` 传入。

**源码：** [`catlass.core_api.arch`](../../../../catlass/core_api.py)

| Tag | 用途 |
| --- | --- |
| `tla.arch.RowMajor` | ND 行主序 |
| `tla.arch.ColumnMajor` | ND 列主序 |
| `tla.arch.VectorLayout` | 一维连续平铺 |
| `tla.arch.zN` | 分形 zN（Cube A/B 暂存） |
| `tla.arch.nZ` | 分形 nZ |
| `tla.arch.L0Clayout` | L0C 累加器布局 |
| `tla.arch.zNUnAlign` | 无 M 轴分形分块的 zN |
| `tla.arch.zZMxScale` | L0A 片上 MX scale |
| `tla.arch.nNMxScale` | L0B 片上 MX scale |
| `tla.arch.RowMajorMxScaleA` | GM MX scale A，行主序 |
| `tla.arch.ColMajorMxScaleA` | GM MX scale A，列主序 |
| `tla.arch.RowMajorMxScaleB` | GM MX scale B，行主序 |
| `tla.arch.ColMajorMxScaleB` | GM MX scale B，列主序 |

约束说明：

- token 是 `tla.arch` 上的普通属性，本身不生成 op。
- 各 API 可能只接受子集，见对应条目的约束说明。

调用示例：

```python
tag = tla.arch.RowMajor
```

## Pipe

`tla.arch` 上的 pipe 标识，供 `flag` / `pipe_barrier` / `mutex_*` / 跨核同步使用。

**源码：** [`catlass.core_api.arch`](../../../../catlass/core_api.py)

| Pipe | 用途 |
| --- | --- |
| `tla.arch.SCALAR` | Scalar pipe |
| `tla.arch.VECTOR` | Vector pipe |
| `tla.arch.CUBE` | Cube pipe |
| `tla.arch.MTE1` | MTE1 |
| `tla.arch.MTE2` | MTE2 |
| `tla.arch.MTE3` | MTE3 |
| `tla.arch.FIX` | Fixpipe |

约束说明：

- 可用 pipe 取决于所在 `tla.cube()` / `tla.vector()` 区域，见各同步接口约束。
- 这些接口不通过 `tla.arch` 暴露 `ALL`。

调用示例：

```python
pipe = tla.arch.MTE2
```

## 片上 AddressSpace

`tla.AddressSpace` 的片上成员，供 `allocate` / `get_capacity_in_bytes` 使用。

**源码：** [`catlass.address_space.AddressSpace`](../../../../catlass/address_space.py)

| Address space | 用途 |
| --- | --- |
| `tla.AddressSpace.l1` | L1 |
| `tla.AddressSpace.l0a` | L0A |
| `tla.AddressSpace.l0b` | L0B |
| `tla.AddressSpace.l0c` | L0C |
| `tla.AddressSpace.ub` | UB |
| `tla.AddressSpace.biasbuf` | Bias 表（C2），供 `mmad` / `mmad_mx` 的 `bias` |

约束说明：

- `allocate` / `get_capacity_in_bytes` 拒绝 `gm`、`generic`。
- 用 `GM`→`L1`→`biasbuf` 的 `tla.copy` 搬入 bias，再把 `biasbuf` tile
  传给 `mmad` / `mmad_mx` 的 `bias=`。
- `make_ptr` 还可使用 `tla.AddressSpace.gm` / `generic`，见该条目。

调用示例：

```python
scope = tla.AddressSpace.ub
```

## 架构辅助

`tla.arch` 上的可调用辅助：block / thread 索引、SIMT `sync_threads`、片上容量与动态 UB。

### `tla.arch.block_idx`

**源码：** [`catlass.core_api.arch_block_idx`](../../../../catlass/core_api.py)

功能说明：

返回当前 AI 核在本次 launch 中的 block 索引（`Int32`）。

函数原型：

```python
tla.arch.block_idx() -> Int32
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。

调用示例：

```python
bid = tla.arch.block_idx()
```

---

### `tla.arch.sub_block_idx`

**源码：** [`catlass.core_api.arch_sub_block_idx`](../../../../catlass/core_api.py)

功能说明：

返回当前 block 内的 sub-block 索引（`Int32`）。

函数原型：

```python
tla.arch.sub_block_idx() -> Int32
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。

调用示例：

```python
sid = tla.arch.sub_block_idx()
```

---

### 指令 Cache 预取

`tla.arch.icache_prefetch(prefetch_units)` 在 Ascend 950 上从调用点的
程序计数器地址预取指令。长度参数为编译期 Python 整数，单位为 2 KiB。
标量 Vector 区域支持 1～7，标量 Cube 区域支持 1～15。两个接口均须在
Kernel 内使用，并且不支持 SIMD 或 SIMT Vector 函数区域。

`tla.arch.get_icache_prefetch_status()` 读取当前核的预取状态。
返回 `tla.Int64`，可在运行期分支中与 `tla.params.ICachePrefetchStatus.COMPLETE`（0）或
`INCOMPLETE`（1）比较。其他状态码与这两个成员均不相等。查询不会等待预取完成。

### `tla.arch.warp_reduce_sync`

**源码：** [`catlass.core_api.arch_warp_reduce_sync`](../../../../catlass/core_api.py)

说明：

对活跃 warp 线程执行标量归约，包括不足 32 个线程的 warp。
每个参与线程均获得归约结果，结果类型与输入标量的类型相同。

原型：

```python
tla.arch.warp_reduce_sync(value: Numeric | int | float, kind: str = 'add') -> Numeric
```

参数：

- `value`：Int8/16/32、UInt8/16/32、Float16 或 Float32 标量。
- `kind`：编译期字符串 `"add"`、`"max"` 或 `"min"`。

约束：

- 只能在 kernel 内的 `tla.vec.func(mode="simt")` 中调用；不提供跨 warp 同步。
- 有符号/无符号窄整数分别符号扩展/零扩展为 i32，归约后再截断为原始位宽。
- Float16 使用原生 f16 归约；浮点加法的结合顺序由硬件决定。

示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=32):
    tid, _, _ = tla.arch.thread_idx()
    gm_out[tid] = tla.arch.warp_reduce_sync(gm_in[tid])
```

---

### `tla.arch.dcache_prefetch`

**源码：** [`catlass.core_api.arch_dcache_prefetch`](../../../../catlass/core_api.py)

在 Ascend 950 的标量 AIV 或 AIC 上，请求包含目标 GM 地址的一条数据缓存行。
`address` 为带类型的 GM 指针，`byte_offset` 为有符号字节偏移，默认为 0，
接受 Python 整数、`Int16` 或 `Int64`。调用者须确保最终地址有效。该提示不提供同步保证。分块视图提供底层指针。请在字节偏移中计入分块起点。

```python
with tla.vector():
    view = tla.tile_view(src, tla.make_shape(16), tla.make_coord(2))  # contiguous Int32
    tla.arch.dcache_prefetch(view.ptr, byte_offset=128)
```

---

### `tla.arch.dcci_cacheline`

**源码：** [`catlass.core_api.arch_dcci_cacheline`](../../../../catlass/core_api.py)

功能说明：

清理并失效持有 `address` 的那一条 dcache 行。

该行先写回 Global Memory，随后被置为无效，之后再读该行会重新从内存取数。
只影响一条行，即持有 `address` 的那一条，`address` 无需按行边界对齐。
单次操作覆盖多大范围不在此约定：要维护多条行，请逐地址各调用一次，或改用
`tla.arch.dcci_cachelines`。

本操作发布一条行，既不是 barrier，也不做跨核同步：消费方所需的 flag / wait
协议仍由调用方负责。要一次发布整个域，使用 `tla.arch.dcci_cachelines`。

函数原型：

```python
tla.arch.dcci_cacheline(address: Pointer) -> None
```

参数说明：

- `address`（`Pointer`）：指向待维护数据的 GM 指针。必选。被维护的是持有该地址
  的那一条行，与指针由什么产生无关。

  张量的 `ptr` 是其背后的指针，因此 `t.ptr` 寻址元素 0，`t.ptr + i` 寻址元素
  `i`。视图暴露的是所视张量的同一个背后指针，而非视图自身的首元素，因此需自行
  加上视图位移：tile 下标为 `c`、tile 形状为 `n` 的 `tla.tile_view`，其首元素
  位于 `view.ptr + c * n`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内，且在 `tla.vec.func` 之外。
- `address` 须为 `gm` 地址空间的 `Pointer`。UB 寻址的行属于另一条原生指令，
  此处不支持。
- Ascend 950（架构 3510）。

调用示例：

```python
with tla.vector():
    out[0] = tla.Int32(101)
    tla.arch.dcci_cacheline(out.ptr)
```

---

### `tla.arch.dcci_cachelines`

**源码：** [`catlass.core_api.arch_dcci_cachelines`](../../../../catlass/core_api.py)

功能说明：

清理并失效本核上某一个域内所有有效的 dcache 行。

所选域内每条行先写回再置为无效。没有地址也没有行数：“cachelines”指整个域，
不是一段地址区间。

域由 `tla.params.DcciTarget` 选择：`ALL` 为本核整个 dcache，`GM_L2` 为持有 GM
地址的行（原生 OUT），`ST_ATOMIC` 为标量原子存储使用的行，即文档所述原子饱和
切换所要求的域。

与 `tla.arch.dcci_cacheline` 一样，本操作只做 cache 维护：既不是 pipe barrier，
也不做跨核同步；`GM_L2` 不承诺刷新共享 L2。

函数原型：

```python
tla.arch.dcci_cachelines(target: Any = None) -> None
```

参数说明：

- `target`（`tla.params.DcciTarget`）：待处理的域，可按位置或关键字传入。
  可选，默认 `DcciTarget.ALL`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内，且在 `tla.vec.func` 之外。
- `target` 为编译期参数；不接受地址或行数。
- `DcciTarget.UB` 被拒绝：原生整个 UB 的形式仅 AIV 可用，且依赖通过 CTRL[49]
  启用 UB dcache，而 TLA 没有任何操作设置该位。
- Ascend 950（架构 3510）。

调用示例：

```python
with tla.vector():
    out[0] = tla.Int32(7)
    tla.arch.dcci_cachelines(tla.params.DcciTarget.GM_L2)
```

---

### `tla.arch.dci`

**源码：** [`catlass.core_api.arch_dci`](../../../../catlass/core_api.py)

功能说明：

失效本核整个 dcache，**不**写回。

每条行都被置为无效，其中的脏数据被丢弃，之后的访问重新从 Global Memory 取数。

本操作不是更快的发布手段。脏数据会丢失，包括 kernel 未显式写出的标量写，如
编译器溢出，因此仅在确实要丢弃这些值时使用。要发布数据请使用
`tla.arch.dcci_cachelines`，它先清理再失效。本操作不同步其他核、icache
或共享 L2。

函数原型：

```python
tla.arch.dci() -> None
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内，且在 `tla.vec.func` 之外。
- 仅 Ascend 950（架构 3510）；A2 与 A3 没有该指令。
- 调用前须发出 `DSB_ALL` 数据屏障，等待此前发出的内存访问完成。TLA 当前未提供该屏障的公开 API。

调用示例：

```python
# 标量区域中的调用顺序：
# 1. 发出 DSB_ALL 数据屏障。
# 2. 调用 tla.arch.dci()。
```

---

### `tla.arch.load_bypass_dcache`

**源码：** [`catlass.core_api.arch_load_bypass_dcache`](../../../../catlass/core_api.py)

功能说明：

绕过 DCache，从 GM 读取一个整数标量。每次调用执行一次独立读取。

函数原型：

```python
tla.arch.load_bypass_dcache(ptr: Pointer) -> Numeric
```

参数说明：

- `ptr`（`Pointer`）：指向 GM 中 8、16、32 或 64 位有符号或无符号整数的指针。必填。

返回值：

- `Numeric`：类型与指针所指元素一致的标量。

约束：

- 在 Ascend 950 上，须在 `tla.cube()` 或 `tla.vector()` 内调用，并位于 `tla.vec.func()` 外。
- 指针须指向有效的 GM 元素，地址须按元素类型自然对齐。

示例：

```python
@tla.kernel
def read_first(src: tla.Tensor, dst: tla.Tensor):
    with tla.cube():
        dst[0] = tla.arch.load_bypass_dcache(src.ptr)
```

---

### `tla.arch.warpsize`

**源码：** [`catlass.core_api.arch_warpsize`](../../../../catlass/core_api.py)

功能说明：

返回 SIMT warp 大小，即 `Int32` 常量 32。

warp 按 x 维优先的线性顺序将每 32 个连续线程分为一组。
最后一个 warp 的线程数不足 32 时，该常量仍为 32。

函数原型：

```python
tla.arch.warpsize() -> Int32
```

参数说明：

- 无。

返回值：

- `Int32`：常量 32。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅在 `tla.vec.func(mode="simt")` 内可用。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=48):
    tid, _, _ = tla.arch.thread_idx()
    gm_out[tid] = tla.arch.warpsize()
```

---

### `tla.arch.lane_idx`

**源码：** [`catlass.core_api.arch_lane_idx`](../../../../catlass/core_api.py)

功能说明：

返回调用线程在其 warp 内的 lane 索引，类型为 `Int32`。

索引范围为 `[0, 31]`，计算公式为
`(x + y * BX + z * BX * BY) % 32`，其中 `(x, y, z)` 为线程索引，
`(BX, BY, BZ)` 为 `thread_block_dim`。
分支不会重新编号 lane。该查询不执行线程同步。

函数原型：

```python
tla.arch.lane_idx() -> Int32
```

参数说明：

- 无。

返回值：

- `Int32`：调用线程的 lane 索引。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅在 `tla.vec.func(mode="simt")` 内可用。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=48):
    tid, _, _ = tla.arch.thread_idx()
    gm_out[tid] = tla.arch.lane_idx()
```

---

### `tla.arch.activemask`

**源码：** [`catlass.core_api.arch_activemask`](../../../../catlass/core_api.py)

功能说明：

查询当前 Warp 中执行到此指令的活跃 Lane。不足 32 个线程的末尾 Warp 只包含
实际启动的线程。发散分支各自观察其活跃集合。

函数原型：

```python
tla.arch.activemask() -> UInt32
```

参数说明：

- 无。

返回值：

- `UInt32` 掩码。Lane `j` 活跃时，第 `j` 位为 1。

约束：

- 仅在 Ascend 950 的 `tla.vec.func(mode="simt")` 中可用。
- 不接受显式成员掩码，不提供跨 Warp 同步。编译器省略未使用的查询。
- 返回值反映执行该指令时的活跃状态。后端可能将相同的分支体汇合。
  不要将此快照作为同步保证或持久成员掩码。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=(48, 1, 1)):
    tid, _, _ = tla.arch.thread_idx()
    gm_out[tid] = tla.arch.activemask()
```

---

### `tla.arch.lanemask_eq`

**源码：** [`catlass.core_api.arch_lanemask_eq`](../../../../catlass/core_api.py)

功能说明：

返回 `UInt32` 掩码，当 `j == lane_idx()` 时第 `j` 位置位。
位置范围为 0–31，包含非活跃或未启动的 lane；该查询不执行同步。

函数原型：

```python
tla.arch.lanemask_eq() -> UInt32
```

参数说明：

- 无。

约束说明：

- 仅在 `@tla.kernel` 内的 `tla.vec.func(mode="simt")` 中可用。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=32):
    mask = tla.arch.lanemask_eq()
```

---

### `tla.arch.lanemask_lt`

**源码：** [`catlass.core_api.arch_lanemask_lt`](../../../../catlass/core_api.py)

功能说明：

返回 `UInt32` 掩码，当 `j < lane_idx()` 时第 `j` 位置位。
位置范围为 0–31，包含非活跃或未启动的 lane；该查询不执行同步。

函数原型：

```python
tla.arch.lanemask_lt() -> UInt32
```

参数说明：

- 无。

约束说明：

- 仅在 `@tla.kernel` 内的 `tla.vec.func(mode="simt")` 中可用。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=32):
    mask = tla.arch.lanemask_lt()
```

---

### `tla.arch.lanemask_le`

**源码：** [`catlass.core_api.arch_lanemask_le`](../../../../catlass/core_api.py)

功能说明：

返回 `UInt32` 掩码，当 `j <= lane_idx()` 时第 `j` 位置位。
位置范围为 0–31，包含非活跃或未启动的 lane；该查询不执行同步。

函数原型：

```python
tla.arch.lanemask_le() -> UInt32
```

参数说明：

- 无。

约束说明：

- 仅在 `@tla.kernel` 内的 `tla.vec.func(mode="simt")` 中可用。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=32):
    mask = tla.arch.lanemask_le()
```

---

### `tla.arch.lanemask_gt`

**源码：** [`catlass.core_api.arch_lanemask_gt`](../../../../catlass/core_api.py)

功能说明：

返回 `UInt32` 掩码，当 `j > lane_idx()` 时第 `j` 位置位。
位置范围为 0–31，包含非活跃或未启动的 lane；该查询不执行同步。

函数原型：

```python
tla.arch.lanemask_gt() -> UInt32
```

参数说明：

- 无。

约束说明：

- 仅在 `@tla.kernel` 内的 `tla.vec.func(mode="simt")` 中可用。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=32):
    mask = tla.arch.lanemask_gt()
```

---

### `tla.arch.lanemask_ge`

**源码：** [`catlass.core_api.arch_lanemask_ge`](../../../../catlass/core_api.py)

功能说明：

返回 `UInt32` 掩码，当 `j >= lane_idx()` 时第 `j` 位置位。
位置范围为 0–31，包含非活跃或未启动的 lane；该查询不执行同步。

函数原型：

```python
tla.arch.lanemask_ge() -> UInt32
```

参数说明：

- 无。

约束说明：

- 仅在 `@tla.kernel` 内的 `tla.vec.func(mode="simt")` 中可用。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=32):
    mask = tla.arch.lanemask_ge()
```

---

### `tla.arch.thread_idx`

**源码：** [`catlass.core_api.arch_thread_idx`](../../../../catlass/core_api.py)

功能说明：

返回调用线程在 thread block 内的索引 `(x, y, z)`。

各分量为 `[0, thread_block_dim[i])` 的 `Int32`。未使用的分量解包到 `_`
不产生开销。线程通常据此划分工作，例如用 `tla.range(tid, n, nthreads)`
遍历自己负责的元素。

函数原型：

```python
tla.arch.thread_idx() -> tuple[Int32, Int32, Int32]
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅可在 `tla.vec.func(mode="simt")` 内使用。launch 级 AI 核索引见
  `tla.arch.block_idx()`。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    tid, _, _ = tla.arch.thread_idx()
    gm_out[tid] = gm_in[tid]
```

---

### `tla.arch.sync_threads`

**源码：** [`catlass.core_api.arch_sync_threads`](../../../../catlass/core_api.py)

功能说明：

对当前 SIMT `tla.vec.func` 内线程做 barrier。

本 block 内所有线程到达后才继续；此前写入对本 block 其他线程可见。
只同步同一 block 内线程，不是跨核 barrier，也不是 pipe barrier（后者见
同步控制中的 flag / `tla.pipe_barrier`）。

典型用法是两阶段归约：各线程把部分结果写到共享 UB，汇合后再由一线程合并。

函数原型：

```python
tla.arch.sync_threads() -> None
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅可在 `tla.vec.func(mode="simt")` 内使用。
- 本 block 每个线程都必须到达；放在仅部分线程进入的分支会导致挂死。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=THREADS):
    tid, _, _ = tla.arch.thread_idx()
    ub_partial[tid] = my_partial
    tla.arch.sync_threads()
    if tid == 0:
        ...  # thread 0 combines ub_partial[0:THREADS]
```

---

### `tla.arch.get_ar`

**源码：** [`catlass.core_api.arch_get_ar`](../../../../catlass/core_api.py)

功能说明：

读取当前 AIV 的 AR 特殊寄存器（`Int64`）。

AR 保存压缩搬出序列的字节数。该序列使用 `tla.squeeze(..., record_nbytes=True)` 和 `POST_MODE_UPDATE_NBYTES`。

本接口在标量单元上执行，读到的是寄存器当前值。它不等待矢量流水，因此完成
产生该计数的指令序列所需的同步握手仍由调用方负责，与读取同一序列写入的 UB
数据一样。

函数原型：

```python
tla.arch.get_ar() -> Int64
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vector()` 内、`tla.vec.func` 外。
- 仅 AIV 可用；`tla.cube()` 上没有 AR。
- Ascend 950（arch 3510）。

调用示例：

```python
count_ready = tla.flag("count_ready", tla.arch.VECTOR, tla.arch.SCALAR)
with tla.vector():
    with tla.vec.func(mode="simd"):
        tla.arch.clear_ar()
        ...  # 在 AR 中记录字节数的压缩指令序列
    tla.set_flag(count_ready)
    tla.wait_flag(count_ready)
    count_bytes = tla.arch.get_ar()
```

---

### `tla.arch.clear_ar`

**源码：** [`catlass.core_api.arch_clear_ar`](../../../../catlass/core_api.py)

功能说明：

将当前 AIV 的 AR 特殊寄存器清零。

在压缩指令序列开始前清零 AR，使字节计数从零开始。

写在持有这些搬出指令的 SIMD `tla.vec.func` 内时，清零指令与写 AR 的指令处于
同一个 Vector Function，这也是原生接口文档给出的位置。直接写在 `tla.vector()`
内的标量位置同样可用，清零的是同一个寄存器；该指令所需的 Vector Function 由
实现提供。

清零 AR 不是同步操作，也不等待尚未完成的序列。在序列仍在累计时清零会丢失
它的计数。

函数原型：

```python
tla.arch.clear_ar() -> None
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vector()` 内，可以直接写在该区域内，也可以写在其中嵌套的
  `tla.vec.func(mode="simd")` 内。
- 仅 AIV 可用；`tla.cube()` 上没有 AR。`tla.vec.func(mode="simt")` 会被拒绝。
- Ascend 950（arch 3510）。

调用示例：

```python
with tla.vector():
    with tla.vec.func(mode="simd"):
        tla.arch.clear_ar()
        ...  # 在 AR 中记录字节数的压缩指令序列
```

---

### `tla.arch.sub_block_num`

**源码：** [`catlass.core_api.arch_sub_block_num`](../../../../catlass/core_api.py)

说明：

返回当前配置中每个逻辑 AI Core 上与调用者同类型的核数。
在分离模式下，MIX kernel 的各侧分别计算自己的 i64 结果。

原型：

```python
tla.arch.sub_block_num() -> Int64
```

参数：

- 无。

约束：

- 在 kernel 标量代码中调用，包括 cube 和 vector 区域。要在 SIMD 或 SIMT 的 `tla.vec.func` 中使用结果，请在区域外计算，再传入标量。

示例：

```python
value = tla.arch.sub_block_num()  # tla.Int64
```

---

### `tla.arch.block_num`

**源码：** [`catlass.core_api.arch_block_num`](../../../../catlass/core_api.py)

功能说明：

返回本次 launch 的 block（AI 核）数量（`Int32`）。

SIMT 区域内的每 block 线程规模见 `tla.arch.thread_block_dim`。

函数原型：

```python
tla.arch.block_num() -> Int32
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。

调用示例：

```python
nblocks = tla.arch.block_num()
```

---

### `tla.arch.thread_block_dim`

**源码：** [`catlass.core_api.arch_thread_block_dim`](../../../../catlass/core_api.py)

功能说明：

返回当前 thread block 的 `(x, y, z)` 规模。

即 `tla.vec.func(thread_block_dim=...)` 声明的值，读回为 `Int32`，供索引
运算（常见作 `tla.range` 的步长）。这是单 block 的线程几何，不是 grid；
launch 的 block 数见 `tla.arch.block_num()`。

函数原型：

```python
tla.arch.thread_block_dim() -> tuple[Int32, Int32, Int32]
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅可在 `tla.vec.func(mode="simt")` 内使用。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    tid, _, _ = tla.arch.thread_idx()
    nthreads, _, _ = tla.arch.thread_block_dim()
    for i in tla.range(tid, N, nthreads):
        gm_out[i] = gm_in[i]
```

---

### `tla.arch.get_vec_len_in_bytes`

**源码：** [`catlass.core_api.arch_get_vec_len_in_bytes`](../../../../catlass/core_api.py)

功能说明：

以字节为单位返回向量寄存器的长度，返回类型为 Python `int`。

函数原型：

```python
tla.arch.get_vec_len_in_bytes() -> int
```

参数说明：

无参数。

约束说明：

- 编译目标为 Ascend 950（3510）时，返回值为 256 字节。
- 可在宿主端调用，也可在 kernel 追踪时作为常量求值。

调用示例：

```python
import catlass.tla as tla

vector_bytes = tla.arch.get_vec_len_in_bytes()
```

---

### `tla.arch.get_datablock_size_in_bytes`

**源码：** [`catlass.core_api.arch_get_datablock_size_in_bytes`](../../../../catlass/core_api.py)

功能说明：

以字节为单位返回一个 DataBlock 的大小，返回类型为 Python `int`。

函数原型：

```python
tla.arch.get_datablock_size_in_bytes() -> int
```

参数说明：

无参数。

约束说明：

- 编译目标为 Ascend 950（3510）时，返回值为 32 字节。
- 可在宿主端调用，也可在 kernel 追踪时作为常量求值。

调用示例：

```python
import catlass.tla as tla

datablock_bytes = tla.arch.get_datablock_size_in_bytes()
```

---

### `tla.arch.get_capacity_in_bytes`

**源码：** [`catlass.core_api.arch_get_capacity_in_bytes`](../../../../catlass/core_api.py)

功能说明：

返回编译目标上某片上存储空间的字节容量。

函数原型：

```python
tla.arch.get_capacity_in_bytes(mem_scope: AddressSpace) -> int
```

参数说明：

- `mem_scope`（`AddressSpace`）：要查询的片上空间，见
  [片上 AddressSpace](#片上-addressspace)。必填。

约束说明：

- 仅接受片上 `AddressSpace`；拒绝 `gm` / `generic`。
- host 侧常量查询：host 与 `@tla.kernel` 内均可调用；kernel 内在 trace
  时折叠为常量（不生成 runtime op）。

调用示例：

```python
l1_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.l1)
ub_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)
```

---

### `tla.arch.get_dyn_ub`

**源码：** [`catlass.core_api.arch_get_dyn_ub`](../../../../catlass/core_api.py)

功能说明：

返回动态 UB 区域指针（长度由 launch 的 `ub=` 指定）。

函数原型：

```python
tla.arch.get_dyn_ub(dtype: type[Numeric], byte_alignment: int | None = None) -> Pointer
```

参数说明：

- `dtype`（`type[Numeric]`）：指针指向的元素类型。必填。
- `byte_alignment`（`int | None`）：指针字节对齐，写法与 `tla.allocate` 相同。
  可选；默认按元素类型宽度。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅 UB；区域位于静态 `tla.allocate` 之上。
- 每个 kernel 至多一个区域；子缓冲用指针算术切分。
- 按实际用量申请（未占用的 UB 可变为 Data Cache）。
- kernel 内不做越界检查。

调用示例：

```python
base = tla.arch.get_dyn_ub(tla.Float32, byte_alignment=256)
lo = tla.make_tensor(base, layout)
hi = tla.make_tensor(base + HALF, layout)
...
artifact(out, block_num=1, ub=64 * 1024)
```

---

## Warp 函数

`tla.arch` 上的 warp 内数据交换，仅适用于 SIMT 区域。

一个 warp 为 32 个线程，由硬件按线性顺序从 thread block 中划分。这些接口在同一
warp 的 lane 寄存器之间搬运数值，因此与「UB 暂存 tensor 加
`tla.arch.sync_threads()`」的写法不同，它们不产生访存，也不需要 block 级
barrier。它们之间的唯一区别是各自读取哪个 lane。

每个接口都接受 `width`：它把 warp 划分为该长度的连续 lane 分组，因此一个 kernel
可以在同一 warp 内跑多路互不影响的归约。它不改变执行该指令的线程集合。每个 mode
只校验自身运算可能越过的那一侧边界，源 lane 在该方向上越界时，该 lane 保留自己的
值；`bfly` 两侧都可能越过，但只按本组末尾校验，因此掩码取 `width` 及以上时会读到
编号更小的分组。要把交换限制在组内，需由调用方把 index 取在 `width` 以下。

支持的产品：Ascend 950PR&950DT系列产品。Atlas A2系列产品、Atlas A3系列产品没有
warp shuffle 指令。

### `tla.arch.shuffle_sync`

**源码：** [`catlass.core_api.arch_shuffle_sync`](../../../../catlass/core_api.py)

功能说明：

返回同一 warp 内 lane `src_lane` 所持有的 `value`，类型与 `value` 相同。

源 lane 为组内的 `src_lane % width`，因此让所有 lane 指向同一个源，即可把该
lane 的值广播到整个组。

函数原型：

```python
tla.arch.shuffle_sync(value: Any, src_lane: Any, width: int = 32) -> Any
```

参数说明：

- `value`：要发布给其他 lane 的值。支持 `Int8`/`Int16`/`Int32`/`Int64` 及其
  无符号形式，以及 `Float16`、`Float32`、`BFloat16`；结果类型与 `value` 相同。
  不支持 `Bool`：它是 `i1`，没有可供其扩位的 1 bit shuffle。
- `src_lane`：`Int32`，要读取的 lane，取 `[0, 32)`。常量为负时在编译期报错；
  更大的取值按上文的 `src_lane % width` 当作 lane 号处理，常量与运行时值皆
  如此；运行时值为负时由调用方自行保证，硬件只取其低 5 bit，不作诊断。
- `width`：Python `int`，取 `[2, 32]` 内的 2 的幂。可选，默认 `32`
  （整个 warp）。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅可在 `tla.vec.func(mode="simt")` 内使用。
- warp 内参与交换的每个线程都必须执行到该指令。被分支挡住的线程不提供自己的
  值，从非活跃 lane 读取的结果未定义。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    # warp 内每个 lane 最终都拿到 lane 0 的值
    leader = tla.arch.shuffle_sync(my_value, 0)
```

---

### `tla.arch.shuffle_sync_up`

**源码：** [`catlass.core_api.arch_shuffle_sync_up`](../../../../catlass/core_api.py)

功能说明：

返回比当前 lane 低 `delta` 个位置的 lane 所持有的 `value`，类型与 `value` 相同。

源 lane 为 `laneid - delta`。源 lane 落在组起点之前的线程保留自己的值，这使它
成为实现 inclusive 前缀和的合适原语。

函数原型：

```python
tla.arch.shuffle_sync_up(value: Any, delta: Any, width: int = 32) -> Any
```

参数说明：

- `value`：要发布给其他 lane 的值。支持 `Int8`/`Int16`/`Int32`/`Int64` 及其
  无符号形式，以及 `Float16`、`Float32`、`BFloat16`；结果类型与 `value` 相同。
  不支持 `Bool`：它是 `i1`，没有可供其扩位的 1 bit shuffle。
- `delta`：`Int32`，向下跨越的 lane 数，取 `[0, 32)`。源 lane 落到本组起点之前是允许的
  用法，该 lane 保留自己的值。可以是运行时值；常量超出 `[0, 32)` 时在编译期
  报错，运行时值由调用方自行保证，硬件只取其低 5 bit，不作诊断，因此运行时
  `delta` 为 33 时读到的是距离为 1 的相邻 lane。
- `width`：Python `int`，取 `[2, 32]` 内的 2 的幂。可选，默认 `32`
  （整个 warp）。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅可在 `tla.vec.func(mode="simt")` 内使用。
- warp 内参与交换的每个线程都必须执行到该指令。被分支挡住的线程不提供自己的
  值，从非活跃 lane 读取的结果未定义。
- `delta` 必须取 `[0, 32)`。常量超出该范围时在编译期报错；运行时值由调用方
  自行保证。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    # warp 内的 Kogge-Stone inclusive scan
    lane, _, _ = tla.arch.thread_idx()
    total = my_value
    for offset in (1, 2, 4, 8, 16):
        other = tla.arch.shuffle_sync_up(total, offset)
        total = tla.where(lane % 32 >= offset, total + other, total)
```

---

### `tla.arch.shuffle_sync_down`

**源码：** [`catlass.core_api.arch_shuffle_sync_down`](../../../../catlass/core_api.py)

功能说明：

返回比当前 lane 高 `delta` 个位置的 lane 所持有的 `value`，类型与 `value` 相同。

源 lane 为 `laneid + delta`。源 lane 越过组末尾的线程保留自己的值。把 `delta`
从 16 逐次减半到 1，即可在 lane 0 得到整个 warp 的归约结果。

函数原型：

```python
tla.arch.shuffle_sync_down(value: Any, delta: Any, width: int = 32) -> Any
```

参数说明：

- `value`：要发布给其他 lane 的值。支持 `Int8`/`Int16`/`Int32`/`Int64` 及其
  无符号形式，以及 `Float16`、`Float32`、`BFloat16`；结果类型与 `value` 相同。
  不支持 `Bool`：它是 `i1`，没有可供其扩位的 1 bit shuffle。
- `delta`：`Int32`，向上跨越的 lane 数，取 `[0, 32)`。源 lane 越过本组末尾是允许的
  用法，该 lane 保留自己的值。可以是运行时值；常量超出 `[0, 32)` 时在编译期
  报错，运行时值由调用方自行保证，硬件只取其低 5 bit，不作诊断，因此运行时
  `delta` 为 33 时读到的是距离为 1 的相邻 lane。
- `width`：Python `int`，取 `[2, 32]` 内的 2 的幂。可选，默认 `32`
  （整个 warp）。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅可在 `tla.vec.func(mode="simt")` 内使用。
- warp 内参与交换的每个线程都必须执行到该指令。被分支挡住的线程不提供自己的
  值，从非活跃 lane 读取的结果未定义。
- `delta` 必须取 `[0, 32)`。常量超出该范围时在编译期报错；运行时值由调用方
  自行保证。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    total = my_value
    for offset in (16, 8, 4, 2, 1):
        total = total + tla.arch.shuffle_sync_down(total, offset)
    # 此时每个 warp 的 lane 0 持有该 warp 的和
```

---

### `tla.arch.shuffle_sync_bfly`

**源码：** [`catlass.core_api.arch_shuffle_sync_bfly`](../../../../catlass/core_api.py)

功能说明：

返回 lane id 为当前 lane 异或 `lane_mask` 的那个 lane 所持有的 `value`，类型与
`value` 相同。

源 lane 为 `laneid ^ lane_mask`，即蝶形交换。与 `down` 形式不同，它让每个 lane
都得到结果，因此据此构建的归约无需再做一次广播。

函数原型：

```python
tla.arch.shuffle_sync_bfly(value: Any, lane_mask: Any, width: int = 32) -> Any
```

参数说明：

- `value`：要发布给其他 lane 的值。支持 `Int8`/`Int16`/`Int32`/`Int64` 及其
  无符号形式，以及 `Float16`、`Float32`、`BFloat16`；结果类型与 `value` 相同。
  不支持 `Bool`：它是 `i1`，没有可供其扩位的 1 bit shuffle。
- `lane_mask`：`Int32`，要在 lane id 上翻转的比特，取 `[0, 32)`。分组边界只在一侧
  校验，因此掩码大于 `width` 是允许的用法，两个方向的行为并不相同：源 lane 落到
  本组末尾之后时，该 lane 保留自己的值；源 lane 落入编号更小的分组时，则读取该
  lane 的值。所以要把蝶形交换限制在组内，需取 `lane_mask < width`，这是调用方的
  选择，而非被拒绝的用法。可以是运行时值；常量超出 `[0, 32)` 时在编译期报错，
  运行时值由调用方自行保证，硬件只取其低 5 bit，不作诊断。
- `width`：Python `int`，取 `[2, 32]` 内的 2 的幂。可选，默认 `32`
  （整个 warp）。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅可在 `tla.vec.func(mode="simt")` 内使用。
- warp 内参与交换的每个线程都必须执行到该指令。被分支挡住的线程不提供自己的
  值，从非活跃 lane 读取的结果未定义。
- `lane_mask` 必须取 `[0, 32)`。常量超出该范围时在编译期报错；运行时值由调用方
  自行保证。

调用示例：

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    total = my_value
    for mask in (16, 8, 4, 2, 1):
        total = total + tla.arch.shuffle_sync_bfly(total, mask)
    # 此时 warp 内每个 lane 都持有该 warp 的和
```

## 系统周期计数器

### `tla.arch.get_system_cycle`

读取当前核的系统周期计数器，返回保留 64 位位模式的 `Int64`。

- 仅可在 Kernel 的标量 `tla.cube()` 或 `tla.vector()` 区域中调用。
- 读取不会插入同步。比较同一个物理核的两次读取时，差值按 2 的 64 次方取模。
- 差值单位为目标设备的系统计数器刻度。

```python
with tla.vector():
    before = tla.arch.get_system_cycle()
    after = tla.arch.get_system_cycle()
    ticks = after - before
```
