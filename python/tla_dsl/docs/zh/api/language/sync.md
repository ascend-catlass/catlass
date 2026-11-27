---
nav_order: 7
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/sync.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 同步控制

[语言参考](index.md) | [API 速查](../quick_reference.md)

核内 / 跨核 flag、pipe barrier、mutex 与本地内存屏障。

> 何时需要同步、如何选择接口、常见卡死 / 结果错误如何排查，见
> [同步使用指南](../../kernel_development/core_concepts/sync.md)。

## `tla.flag`

**源码：** [`catlass.core_api.flag`](../../../../catlass/core_api.py)

功能说明：

在两条 pipe 之间创建管内同步 flag。

函数原型：

```python
tla.flag(name: str, src_pipe: PipeLike, dst_pipe: PipeLike, id: int = -1) -> TlaFlag
```

参数说明：

- `name`（`str`）：管内 flag 名称。必填。
- `src_pipe`（`PipeLike`）：源 pipe 标识，见 [Pipe](system.md#pipe)。必填。
- `dst_pipe`（`PipeLike`）：目的 pipe 标识，见 [Pipe](system.md#pipe)。必填。
- `id`（`int`）：硬件事件 ID。可选，默认 `-1`，表示自动分配。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 创建的是 flag 句柄；`set_flag`/`wait_flag` 须在 cube/vector 区域内成对使用。
- `id` 必须为 `-1` 或 `0..7`。自动分配会跳过同一 pipe pair 上显式指定的 ID。
- 显式 ID 可以复用；调用方须保证对应同步的生命周期不重叠。

调用示例：

```python
# MTE2 完成 GM→UB 拷贝后，VECTOR 侧才可消费该 UB tile。
ub_loaded = tla.flag("ub_loaded", src_pipe=tla.pipes.MTE2, dst_pipe=tla.pipes.VECTOR)
with tla.vector():
    tla.copy(dst=x_ub, src=x_gm)
    tla.set_flag(ub_loaded)   # 拷贝后：标记 UB 数据就绪
    tla.wait_flag(ub_loaded)  # 计算前：等待就绪
```

---

## `tla.cross_flag`

**源码：** [`catlass.core_api.cross_flag`](../../../../catlass/core_api.py)

功能说明：

创建命名的跨核同步 flag。源/目的 pipe 由对应 set/wait 指定；`mode=4` 为 1:1 AIC↔AIV，并按 AIV0/AIV1 独立寻址。

函数原型：

```python
tla.cross_flag(name: str, *, mode: int = 2) -> TlaCrossFlag
```

参数说明：

- `name`（`str`）：跨核 flag 名称。必填。
- `mode`（`int`）：跨核同步模式。可选，默认 `2`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `mode` 仅支持 0/1/2/4；源/目的 pipe 由对应 set/wait 指定。

调用示例：

```python
cf = tla.cross_flag("aic_aiv", mode=2)
```

---

## `tla.cross_core_set_flag`

**源码：** [`catlass.core_api.cross_core_set_flag`](../../../../catlass/core_api.py)

功能说明：

从指定 `pipe` 置位跨核 flag；`mode=4` 时需提供 `aiv_id`（0 或 1）。

函数原型：

```python
tla.cross_core_set_flag(cross_flag_value: CrossFlagLike, pipe: PipeLike, aiv_id: int | None = None) -> None
```

参数说明：

- `cross_flag_value`（`CrossFlagLike`）：由 `tla.cross_flag` 得到的跨核 flag。必填。
- `pipe`（`PipeLike`）：设置 flag 的 pipe。必填。
- `aiv_id`（`int | None`）：目标 AIV 编号；缺省表示广播/默认路由。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用；`mode=4` 时 `aiv_id` 须为 0 或 1。

调用示例：

```python
with tla.cube():
    tla.cross_core_set_flag(cf, tla.pipes.CUBE)
    # mode=4: tla.cross_core_set_flag(cf, tla.pipes.CUBE, aiv_id=0)
```

---

## `tla.cross_core_wait_flag`

**源码：** [`catlass.core_api.cross_core_wait_flag`](../../../../catlass/core_api.py)

功能说明：

在指定 `pipe` 上等待跨核 flag；`mode=4` 时需提供 `aiv_id`（0 或 1）。

函数原型：

```python
tla.cross_core_wait_flag(cross_flag_value: CrossFlagLike, pipe: PipeLike, aiv_id: int | None = None) -> None
```

参数说明：

- `cross_flag_value`（`CrossFlagLike`）：由 `tla.cross_flag` 得到的跨核 flag。必填。
- `pipe`（`PipeLike`）：执行 wait 的 pipe。必填。
- `aiv_id`（`int | None`）：目标 AIV 编号；缺省表示广播/默认路由。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用；`mode=4` 时 `aiv_id` 须为 0 或 1。

调用示例：

```python
with tla.vector():
    tla.cross_core_wait_flag(cf, tla.pipes.VECTOR)
```

---

## `tla.set_flag`

**源码：** [`catlass.core_api.set_flag`](../../../../catlass/core_api.py)

功能说明：

置位同核同步 flag。

函数原型：

```python
tla.set_flag(flag_value: FlagLike) -> None
```

参数说明：

- `flag_value`（`FlagLike`）：由 `tla.flag` 得到的核内 pipe flag。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。

调用示例：

```python
with tla.vector():
    tla.set_flag(ub_loaded)
```

---

## `tla.wait_flag`

**源码：** [`catlass.core_api.wait_flag`](../../../../catlass/core_api.py)

功能说明：

等待同核同步 flag。

函数原型：

```python
tla.wait_flag(flag_value: FlagLike) -> None
```

参数说明：

- `flag_value`（`FlagLike`）：由 `tla.flag` 得到的核内 pipe flag。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。

调用示例：

```python
with tla.vector():
    tla.wait_flag(ub_loaded)
```

---

## `tla.pipe_barrier`

**源码：** [`catlass.core_api.pipe_barrier`](../../../../catlass/core_api.py)

功能说明：

插入指定硬件 pipe 的屏障。

函数原型：

```python
tla.pipe_barrier(pipe: PipeLike) -> None
```

参数说明：

- `pipe`（`PipeLike`）：要插入 barrier 的 pipe。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。
- 允许的 pipe 随所在区域变化：`tla.cube()` 内为 `CUBE`、`MTE1`、`MTE2`、`FIX`、`ALL`；
  `tla.vector()` 内为 `MTE2`、`MTE3`、`ALL`（3510 不支持 `pipe_barrier<PIPE_V>`）。
- kernel 末尾需要排空所有流水时使用 `tla.pipes.ALL`。

调用示例：

```python
with tla.vector():
    tla.pipe_barrier(tla.pipes.MTE2)
```

---

## `tla.sync_all_aiv`

**源码：** [`catlass.core_api.sync_all_aiv`](../../../../catlass/core_api.py)

功能说明：

同步参与本 kernel 的全部 AIV。每个 AIV 通过跨核 flag 发布到达信号，
再等待所有对端到达，因此是 AIV 的全核同步。

函数原型：

```python
tla.sync_all_aiv(trigger_pipe: PipeLike = tla.pipes.ALL, wait_pipe: PipeLike = tla.pipes.ALL) -> None
```

参数说明：

- `trigger_pipe`（`PipeLike`）：发出到达信号的流水。可选，默认 `tla.pipes.ALL`；
  仅支持 `tla.pipes.MTE2`、`tla.pipes.MTE3` 与 `tla.pipes.ALL`。
  `tla.pipes.ALL` 会先排空全部流水，再从 `MTE3` 发信号。
- `wait_pipe`（`PipeLike`）：等待对端的流水。可选，默认 `tla.pipes.ALL`；
  仅支持 `tla.pipes.MTE2`、`tla.pipes.MTE3` 与 `tla.pipes.ALL`。
  `tla.pipes.ALL` 使用标量流水等待。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vector()` 内调用：纯 AIV kernel 与 MIX kernel 的 vector 侧均可。
- 每个 AIV 的调用次数、顺序与 pipe 选择必须一致，否则会死锁。
- MIX kernel 需要 cube/vector 之间的屏障时，两侧改用 `tla.sync_all()`。

调用示例：

```python
with tla.vector():
    tla.sync_all_aiv()
    # 或选择承载数据的拷贝流水
    tla.sync_all_aiv(tla.pipes.MTE3, tla.pipes.ALL)
```

---

## `tla.sync_all`

**源码：** [`catlass.core_api.sync_all`](../../../../catlass/core_api.py)

功能说明：

同步 MIX kernel 的 cube 与 vector 核。cube 侧与 vector 侧各写一条调用，
两条都必须有：cube 等待全部 AIV 到达后放行，AIV 发布到达信号并等待放行。

函数原型：

```python
tla.sync_all() -> None
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 仅在声明了 `core_ratio` 的 MIX kernel 中有效；纯 AIV kernel 改用 `tla.sync_all_aiv()`。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用；cube 侧调用与同一 kernel 的
  vector 侧调用配对。

调用示例：

```python
@tla.kernel(core_ratio=tla.arch.C1V2)
def mixed(...):
    with tla.cube():
        ...
        tla.sync_all()
    with tla.vector():
        tla.sync_all()
        ...
```

---

## `tla.mutex`

**源码：** [`catlass.core_api.mutex`](../../../../catlass/core_api.py)

功能说明：

创建与语义资源关联的 mutex 句柄。

函数原型：

```python
tla.mutex(resource: str, id: int = -1) -> TlaMutex
```

参数说明：

- `resource`（`str`）：互斥资源名。必填。
- `id`（`int`）：互斥实例编号。可选，默认 `-1`，但当前 bitcode 调用 lowering 不支持自动分配，
  请显式传入 `0..31`，该范围内没有任何 id 被保留。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `resource` 非空；`id` 须在 `0..31`。`id=-1` 前端接受，但编译到 bitcode 调用时会报
  `mutex id auto allocation is not implemented for bitcode call lowering`。
- mutex 只在单个 AIC 或 AIV 内有效；跨核请使用 `tla.cross_flag`。

调用示例：

```python
mtx = tla.mutex("l1_buf", id=0)
```

---

## `tla.mutex_guard`

**源码：** [`catlass.core_api.mutex_guard`](../../../../catlass/core_api.py)

功能说明：

创建上下文管理器，对代码块自动推断 mutex 访问。

函数原型：

```python
tla.mutex_guard(*mutexes: MutexLike) -> _MutexGuard
```

参数说明：

- *`mutexes`*（`MutexLike`）：一个或多个 mutex 对象。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 块内须调用 `copy` 或 `mmad`；guard 内不可再显式 lock/unlock。

调用示例：

```python
with tla.mutex_guard(mtx):
    tla.copy(dst, src)
```

---

## `tla.mutex_lock`

**源码：** [`catlass.core_api.mutex_lock`](../../../../catlass/core_api.py)

功能说明：

从指定 pipe 获取 mutex。

函数原型：

```python
tla.mutex_lock(mutex_value: MutexLike, *, pipe: PipeLike) -> None
```

参数说明：

- `mutex_value`（`MutexLike`）：要加锁的 mutex。必填。
- `pipe`（`PipeLike`）：加锁所在 pipe。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用；须指定 `pipe`。
- 可用 `pipe` 取决于所在核：`tla.cube()` 内支持 `SCALAR`、`CUBE`、`MTE1`、`MTE2`、`MTE3`、`FIX`；`tla.vector()` 内支持 `SCALAR`、`VECTOR`、`MTE2`、`MTE3`。`ALL` 无对应 mutex 形式。该核不支持的 pipe 在 kernel 编译时报错，而非在此调用处报错。

调用示例：

```python
tla.mutex_lock(mtx, pipe=tla.pipes.MTE2)
# scalar pipe，用于保护标量单元读写的 buffer。
# `tla.mutex` 返回值提供该方法形式：
mtx.lock(pipe=tla.pipes.SCALAR)
```

---

## `tla.mutex_unlock`

**源码：** [`catlass.core_api.mutex_unlock`](../../../../catlass/core_api.py)

功能说明：

从指定 pipe 释放 mutex。

函数原型：

```python
tla.mutex_unlock(mutex_value: MutexLike, *, pipe: PipeLike) -> None
```

参数说明：

- `mutex_value`（`MutexLike`）：要解锁的 mutex。必填。
- `pipe`（`PipeLike`）：解锁所在 pipe。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用；须指定 `pipe`。
- 可用 `pipe` 取决于所在核：`tla.cube()` 内支持 `SCALAR`、`CUBE`、`MTE1`、`MTE2`、`MTE3`、`FIX`；`tla.vector()` 内支持 `SCALAR`、`VECTOR`、`MTE2`、`MTE3`。`ALL` 无对应 mutex 形式。该核不支持的 pipe 在 kernel 编译时报错，而非在此调用处报错。

调用示例：

```python
tla.mutex_unlock(mtx, pipe=tla.pipes.MTE2)
# scalar pipe，用于保护标量单元读写的 buffer。
# `tla.mutex` 返回值提供该方法形式：
mtx.unlock(pipe=tla.pipes.SCALAR)
```

---

## `tla.local_mem_bar`

**源码：** [`catlass.core_api.local_mem_bar`](../../../../catlass/core_api.py)

功能说明：

在 `vec.func` 内插入本地内存屏障（按 `MemType` 对编码）。

函数原型：

```python
tla.local_mem_bar(src: MemType, dst: MemType)
```

参数说明：

- `src`（`MemType`）：源本地内存类型。必填。
- `dst`（`MemType`）：目的本地内存类型。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`(src, dst)` 须为支持的 `MemType` 对。

调用示例：

```python
with tla.vec.func(mode="simd"):
    tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
```

---

## `tla.data_sync_barrier`

**源码：** [`catlass.core_api.data_sync_barrier`](../../../../catlass/core_api.py)

功能说明：

按存储域插入数据同步屏障。阻止后续指令执行，直至此前发出的 `domain`
存储域访存指令完成。

函数原型：

```python
tla.data_sync_barrier(domain: DsbTarget) -> None
```

参数说明：

- `domain`（`DsbTarget`）：等待的存储域。必填。`GM_UB` 等待此前发出的
  GM 与 UB 访存；`GM` 仅 GM；`UB` 仅 UB；`SEQ` 为序列器访问（含参数缓冲区）。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用；内核体、`tla.cube()` 与
  `tla.vector()` 均可。
- 不得在 `tla.vec.func()` 内调用；`dsb` 是作用于整个存储域的标量流水指令，
  不是按线程生效的视图。
- `tla.cube()` 内拒绝 `UB`：AIC 无 UB，不会下发该屏障，请改用 `GM`、`GM_UB`
  或 `SEQ`，或将调用移入 `tla.vector()`。混合内核标量体内 `UB` 域在 AIC
  副本上为空操作。
- `GM_UB` 只覆盖 GM 与 UB：不含 L1 与 L0A/L0B/L0C Buffer；这些位置的顺序
  用 `tla.pipe_barrier`（`MTE1`/`FIX`/`CUBE`）或 `tla.flag` /
  `set_flag` / `wait_flag` 处理。
- 「对其他核可见」的保证针对**绕过 D-cache** 的 GM 访问。普通标量流水的
  GM 写在内核结束前对其他引擎与其他核不可见；V-pipe 的 UB 写不被 `UB` /
  `GM_UB` 等待——`vec.func` 写 UB 后拷出到 GM 仍须 flag 对排序。
- 不参与 `auto_sync`；请放在热点内层循环之外，每次迭代多一次屏障等待。

调用示例：

```python
with tla.vector():
    # 官方跨核形态：旁路写数据 → 等待 GM 域 → 旁路写同步标记；
    # 消费者旁路轮询标记后再旁路读数据。完整样例见
    # tests/end_to_end/data_sync_barrier.py。
    bypass_write(src.ptr, 1, data)  # 用户 extern：WriteGmBypassDCache
    tla.data_sync_barrier(tla.params.DsbTarget.GM)
    bypass_write(src.ptr, 0, flag)  # 用户 extern：WriteGmBypassDCache
```
