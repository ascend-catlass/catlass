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
