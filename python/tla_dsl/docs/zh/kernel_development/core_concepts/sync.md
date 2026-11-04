---
nav_order: 25
---

# 同步使用指南

CATLASS DSL 默认**不会**自动插入流水（pipe）之间的同步：`tla.copy`、`tla.mmad`、
`tla.vec.func` 内部的计算都会被翻译成对应流水上的指令，但这些流水在硬件上异步执行。
只有显式写了 `tla.flag` / `set_flag` / `wait_flag`（或跨核 `cross_flag`、`mutex`），
才能保证生产者先完成、消费者后开始。

同步写错通常**不报编译错误**，而是在设备上表现为：

- 程序卡死（等一个永远不会被置位的 flag，或等一个已被消耗的 flag）；
- 结果偶发错误（大多数迭代结果正确，个别 tile 读到旧数据）；
- 编译期报错（event id 耗尽、重复 flag 名、`wait_flag` 找不到配对的 `set_flag`）。

编译期只做**单向配对检查**：有 `wait_flag` 但没有 `set_flag` 才会报错。多余的 `set_flag`、
set/wait 顺序反了、漏了环前 prime 或环后 drain、同一 flag 生命周期重叠复用，都不会被
拦下，因此不能用“能编过”来证明同步协议正确。

本文按“判断要不要同步 -> 选哪个接口 -> 怎么写 -> 怎么排查”的顺序说明，
接口签名见 [同步控制 API 参考](../../api/language/sync.md)。

## 1. 硬件模型：流水（pipe）

一个 AIC/AIV 内部有多条独立流水，指令在各自流水上顺序执行，不同流水之间没有顺序保证。

| 流水 | TLA 标识 | 负责的指令 |
| --- | --- | --- |
| PIPE_S | `tla.arch.SCALAR` | 标量运算、张量下标读写、`vec.func` 内的标量 load/store |
| PIPE_V | `tla.arch.VECTOR` | 向量计算（SIMD 算术、`tile.load/store` 等） |
| PIPE_M | `tla.arch.CUBE` | `tla.mmad` 矩阵乘 |
| PIPE_MTE1 | `tla.arch.MTE1` | L1 -> L0A/L0B 搬运 |
| PIPE_MTE2 | `tla.arch.MTE2` | GM -> L1/L0/UB 搬运（含 GM -> L1 的 im2col 等） |
| PIPE_MTE3 | `tla.arch.MTE3` | UB -> GM 搬运 |
| PIPE_FIX | `tla.arch.FIX` | L0C -> GM/UB 搬运 |

同一条流水内的指令天然有序，**跨流水才有依赖问题**。`tla.vec.func` 内部的向量/标量
可见性不属这一层，需要时按第 8 节的 `local_mem_bar` 处理。

## 2. 每个操作属于哪条流水

判断同步的第一步是找出生产者和消费者各自的流水。`tla.copy` 的流水由**源**地址空间决定：

| 操作 | 流水 | 说明 |
| --- | --- | --- |
| `tla.copy(dst_l1, src_gm)` | MTE2 | 源是 GM |
| `tla.copy(dst_l0, src_l1)` | MTE1 | 源是 L1 |
| `tla.copy(dst_ub, src_gm)` | MTE2 | 源是 GM |
| `tla.copy(dst_gm, src_ub)` | MTE3 | 源是 UB |
| `tla.copy(dst_gm, src_l0c)` | FIX | 源是 L0C |
| `tla.mmad(...)` | CUBE | |
| SIMD 计算、`tile.store` / `tile.load` | VECTOR | `tla.vec.func(mode="simd")` 内 |
| `tensor[i]` 读写、标量运算 | SCALAR | |
| `tla.vec.func` 内的标量 load/store | SCALAR | |

注意：`tla.copy` 的源是 L0A/L0B 时没有可自动推断的流水
（`auto_sync` 会直接报错），手写同步时也要自己确认它走哪条流水。

## 3. 判断是否需要同步

对每一对“生产者写、消费者读”的操作，按下面三步判断：

1. **是否同一条流水？** 同流水不需要同步。
2. **是否存在数据依赖？** 看下表。
3. **消费者有没有可能先跑？** 有可能就需要同步；不确定时按“需要”处理。

| 依赖 | 含义 | 典型场景 | 需要插入的 flag |
| --- | --- | --- | --- |
| RAW（写后读） | 消费者读生产者刚写的数据 | `copy` 到 UB 后立刻 `vec.func` 计算 | 生产者流水 -> 消费者流水 |
| WAR（读后写） | 生产者要覆盖消费者还在读的 buffer | 下一轮 `copy` 覆盖上一轮刚算完的 UB | 消费者流水 -> 生产者流水 |
| WAW（写后写） | 两次写同一块内存 | 两次 `copy` 写同一 UB tile | 前一次写 -> 后一次写 |

最常见的错误是只写 RAW、漏掉 WAR。双缓冲循环里两种都需要，见第 5 节。

## 4. 核内同步：`tla.flag` + `set_flag` / `wait_flag`

两类接口的语义如下：

- `set_flag`：等**源流水**的前序指令全部完成，把 flag 置位；它**不阻塞**源流水的后续指令；
- `wait_flag`：**目的流水**执行到这里时，若 flag 未置位就阻塞其后续指令；置位后清零并继续。

写法规则：

- `tla.flag(name, src_pipe, dst_pipe, id=-1)` 只创建句柄，建议放在 kernel 函数体顶部、
  进入 `tla.cube()` / `tla.vector()` 之前；
- `tla.set_flag` / `tla.wait_flag` 必须在 `tla.cube()` 或 `tla.vector()` 内，且成对出现；
- 同一个 flag 名只能定义一次，`set_flag` 与 `wait_flag` 必须引用同一个句柄；
- 一个 flag 是“一次性事件”：一次 `set` 对应一次 `wait`。同一 flag 连续 `set` 两次是
  未定义行为，可能在后续 `pipe_barrier(ALL)` 处卡死；
- 每个 `(src_pipe, dst_pipe)` 组合只有 **8 个硬件 event id（0..7）**。默认 `id=-1`
  自动分配并跳过已显式占用的 id；显式 `id` 可以复用，但必须保证两段同步的生命周期不重叠。

最小示例（GM -> UB -> 计算 -> GM）：

```python
@tla.kernel
def vadd(x: tla.Tensor, y: tla.Tensor, out: tla.Tensor) -> None:
    # 省略 buffer 与 tile view 的创建：x_ub / y_ub / out_ub 由 tla.make_tensor_like 得到
    # 数据就绪：MTE2 写完 UB 后，VECTOR 才能读
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    # 计算完成：VECTOR 写完 UB 后，MTE3 才能读
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    with tla.vector():
        tla.copy(x_ub, x_gm)          # MTE2
        tla.copy(y_ub, y_gm)          # MTE2
        tla.set_flag(ub_loaded)       # MTE2 前序完成
        tla.wait_flag(ub_loaded)      # VECTOR 等数据

        with tla.vec.func(mode="simd"):
            out_ub.store(x_ub.load() + y_ub.load())   # VECTOR

        tla.set_flag(vec_done)        # VECTOR 前序完成
        tla.wait_flag(vec_done)       # MTE3 等计算
        tla.copy(out_gm, out_ub)      # MTE3

        tla.pipe_barrier(tla.pipes.ALL)   # 收尾，见第 8 节
```

WAR 的场景（消费者先读完，生产者才能覆盖）：

```python
# VECTOR 读完 ub 之后，MTE2 才能写下一轮
ub_free = tla.flag("ub_free", tla.arch.VECTOR, tla.arch.MTE2)
...
tla.set_flag(ub_free)
tla.wait_flag(ub_free)
tla.copy(ub, gm_next)
```

## 5. 双缓冲（ping-pong）

双缓冲需要每个 buffer 一对 flag：

- `loaded[b]`：缓冲区数据就绪（生产者流水 -> 消费者流水）；
- `available[b]`：缓冲区已空闲、可被覆盖（**读走它的流水** -> 生产者流水）。

两个关键点，漏一个就会出问题：

1. **循环开始前必须预置 `available`**，否则第一轮 `wait_flag(available[0])` 永远等不到；
2. **循环结束后要 drain**：用 `pipe_barrier(tla.pipes.ALL)` 保证所有已发出的搬运/计算完成。

```python
@tla.kernel
def vadd_pingpong(x: tla.Tensor, y: tla.Tensor, out: tla.Tensor) -> None:
    num_tiles = 4

    loaded = (
        tla.flag("loaded_0", tla.arch.MTE2, tla.arch.VECTOR, id=0),
        tla.flag("loaded_1", tla.arch.MTE2, tla.arch.VECTOR, id=1),
    )
    store_done = (
        tla.flag("store_done_0", tla.arch.VECTOR, tla.arch.MTE3, id=0),
        tla.flag("store_done_1", tla.arch.VECTOR, tla.arch.MTE3, id=1),
    )
    available = (
        tla.flag("available_0", tla.arch.MTE3, tla.arch.MTE2, id=0),
        tla.flag("available_1", tla.arch.MTE3, tla.arch.MTE2, id=1),
    )

    with tla.vector():
        # (1) 预置初始状态：两个 buffer 首轮都可写
        tla.set_flag(available[0])
        tla.set_flag(available[1])

        # range_constexpr 在编译期展开，slot 是编译期整数，才能索引 flag 元组
        for i in tla.range_constexpr(num_tiles):
            slot = i % 2

            # (2) WAR：上一轮把该 buffer 读走/写回之后，MTE2 才能覆盖
            tla.wait_flag(available[slot])
            tla.copy(ub_x[slot], gm_x[i])     # MTE2
            tla.copy(ub_y[slot], gm_y[i])     # MTE2

            # (3) RAW：数据就绪，VECTOR 才能读
            tla.set_flag(loaded[slot])
            tla.wait_flag(loaded[slot])
            with tla.vec.func(mode="simd"):
                ub_out[slot].store(ub_x[slot].load() + ub_y[slot].load())

            # (4) RAW：计算完成，MTE3 才能读并写回 GM
            tla.set_flag(store_done[slot])
            tla.wait_flag(store_done[slot])
            tla.copy(gm_out[i], ub_out[slot])  # MTE3

            # (5) WAR：写回完成，buffer 重新可用
            tla.set_flag(available[slot])

        # (6) drain：保证最后一次写回真正完成
        tla.pipe_barrier(tla.pipes.ALL)
```

> 说明：这里把 x/y/out 三个 buffer 当作同一个槽位统一管理，`available` 等到 `MTE3`
> 写回完成才放行下一轮 `MTE2`，是最保守的写法。若只保护输入 buffer，方向应改为
> `VECTOR -> MTE2`，并在计算结束后 `set_flag`；此时输出 buffer 需要单独一对 flag。

约束：flag 句柄的选择必须能在编译期确定。循环边界是运行时变量（`tla.range`）时，
不能用循环变量去索引 Python 元组，需要把缓冲区展开成编译期分支，或改用
`tla.range_constexpr`（注意它会把循环体展开）。

另外，`set_flag` / `wait_flag` 的操作数必须直接由 `tla.flag` 定义。经 `scf.if` yield
或其它控制流把句柄变成 block argument 时，lowering 会报
`expected ... to have a 'flag' attribute or a tla.flag operand`。`cross_flag` 有结构化
转换，允许更动态的携带方式；核内 flag 没有对等路径。

## 6. 跨核同步：`tla.cross_flag`

AIC 与 AIV 之间、不同 AIV 之间共享数据时，核内 flag 无效，必须用跨核 flag。
`tla.cross_flag(name, mode=...)` 创建句柄，`cross_core_set_flag` / `cross_core_wait_flag`
在 `tla.cube()` / `tla.vector()` 内指定 pipe 并成对使用。

| mode | 同步范围 | 说明 |
| --- | --- | --- |
| 0 | 所有 AI Core 的 AIC（或所有 AIV） | 全核栅栏，所有参与者都 set 后 wait 才放行 |
| 1 | 单个 AI Core 内的 AIV 之间 | 两个 AIV 都 set 后互相放行 |
| 2（默认） | 单核内 AIC <-> 全部 AIV | AIC set 后两个 AIV 都放行；两个 AIV 都 set 后 AIC 才放行 |
| 4 | 单核内 AIC <-> 单个 AIV | 按 `aiv_id`（0/1）独立寻址，仅 Ascend 950 支持 |

规则与限制：

- mode 0/1/2 的 flag id 范围是 0..15；mode 4 下 AIC<->AIV0 占 0..15、AIC<->AIV1 占 16..31；
  一个 kernel 最多 **16 个 cross flag**，否则报 `cross flag id exhausted`；
- `cross_core_set_flag` / `cross_core_wait_flag` 的 `pipe` 不支持 `ALL`；
  常用 `tla.arch.CUBE`、`tla.arch.FIX`、`tla.arch.VECTOR`、`tla.arch.MTE2`；
- mode 4 必须显式给出 `aiv_id=0/1`。AIC 侧写 `aiv_id=i`，AIV 侧对 0 和 1 各写一次，
  编译器会按 `sub_block_idx` 生成 guard，运行期只执行属于本 AIV 的那一条；
- `@tla.kernel(core_ratio=tla.arch.C1V1)` 时：mode 4 只允许 `aiv_id=0`；
  `cross_core_wait_flag` 必须用 `tla.arch.SCALAR`；L0C 结果要先 `L0C->GM->UB`，
  不能紧耦合 `L0C->UB`。编译器不会替你改写错误的 wait pipe，也不会在编译期强制这条
  约束：写错 pipe 一般仍能编过，但会在设备侧握手失败或卡死（参考
  `basic_mixed_core_ratio_11.py`）。

mode 4 的两段式握手（AIC 交付 token -> AIV 回 ack -> AIC 再放行）。
下面是双 AIV（非 `C1V1`）示例；`C1V1` 下按上文改为只对 `aiv_id=0` 操作，
且 `cross_core_wait_flag` 用 `tla.arch.SCALAR`：

```python
aic_to_aiv = tla.cross_flag("aic_to_aiv", mode=4)
aiv_to_aic = tla.cross_flag("aiv_to_aic", mode=4)

with tla.cube():
    tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=0)
    tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=1)
    # 两个 AIV 都确认后才继续
    tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=0)
    tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=1)

with tla.vector():
    tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=0)
    tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=1)
    ...
    tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=0)
    tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=1)
```

完整可运行版本见
[`cross_flag_two_way.py`](https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/cross_flag_two_way/cross_flag_two_way.py)。

## 7. 资源互斥：`tla.mutex`

`mutex` 与 flag 解决的是不同问题：

| | flag（`tla.flag`） | mutex（`tla.mutex`） |
| --- | --- | --- |
| 语义 | 一次性事件：一次 set 对应一次 wait | 计数锁：Lock 阻塞到同 id 全部 Unlock |
| 方向 | 单向（源流水 -> 目的流水） | 任意方向，可表达“等所有读者退出” |
| 适用 | 固定生产/消费顺序 | 多消费者、多 buffer 复用、反向同步 |

```python
mtx = tla.mutex("l1_buf", id=0)     # id 必须显式给出

with tla.cube():
    with tla.mutex_guard(mtx):
        tla.copy(l1_a, gm_a)        # guard 块内必须有 copy 或 mmad
```

规则与限制：

- **`id` 必须显式指定**：当前 bitcode lowering 不支持自动分配，`id=-1` 会报
  `mutex id auto allocation is not implemented`。`id` 取值 `0..31`，该范围内没有
  保留的 id；
- `pipe` 与所在区域有关：`tla.cube()` 内支持 `SCALAR`/`CUBE`/`MTE1`/`MTE2`/`MTE3`/`FIX`；
  `tla.vector()` 内支持 `SCALAR`/`VECTOR`/`MTE2`/`MTE3`；没有 `ALL` 形式；
- mutex 只在单核内有效，跨核请用 `cross_flag`；
- `mutex_guard` 块内不要再次显式 `mutex_lock`/`mutex_unlock`。

## 8. `pipe_barrier` 与 `local_mem_bar`

`tla.pipe_barrier(pipe)` 阻塞**指定流水**，直到该流水上此前已发出的指令全部完成；它不会
等待其它流水。需要排空所有流水时用 `tla.pipes.ALL`。它必须在 `tla.cube()` /
`tla.vector()` 内，且 Ascend 950 上支持的 pipe 有限：

| 区域 | 允许的 pipe |
| --- | --- |
| `tla.cube()` | `CUBE`、`MTE1`、`MTE2`、`FIX`、`ALL` |
| `tla.vector()` | `MTE2`、`MTE3`、`ALL`（不支持 `pipe_barrier<PIPE_V>`） |

因此 vector 区域里不要写 `tla.pipe_barrier(tla.arch.VECTOR)`，需要全流水等待时用
`tla.pipe_barrier(tla.pipes.ALL)`。

`tla.local_mem_bar(src, dst)` 只能在 `tla.vec.func` 内使用，用于 SIMD 区域内部
向量与标量访问之间的可见性，支持的组合限定为 12 种，常见的有：

```python
tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.SCALAR_LOAD)
tla.local_mem_bar(tla.params.MemType.VEC_ALL, tla.params.MemType.SCALAR_ALL)
```

注意它**只作用于 `vec.func` 内部**：`vec.func` 结束后，外层还要用正常 flag 保证
标量/其他流水看到完整结果（见
[`load_and_store_scalar_after_reduction.py`](https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_and_store_scalar_after_reduction.py)）。
SIMT 区域内的线程同步用 `tla.arch.sync_threads()`，它不是核间或流水 barrier。

## 9. 自动同步：`@tla.kernel(auto_sync="v0")`

`auto_sync="v0"` 会为核内的 `tla.copy`、`tla.mmad`、`tla.vec.func` 访问自动插入
mutex 形式的同步，适合规则的搬运-计算流水。限制：

- 只生成**单个 AIC 或 AIV 内部**的流水同步；跨核同步和 `vec.func` 内线程同步仍需手写；
- 不能与手写 `tla.flag`/`set_flag`/`wait_flag`、`tla.mutex`/`mutex_guard` 混用；
  `tla.call_extern` 也不支持；
- 受保护的片上 tensor 必须来自 `tla.allocate`，不能用 `tla.make_ptr` 构造；
- 环内不能把片上 pointer 当作循环状态并切换到另一块 `tla.allocate` 的 allocation root；
- `tla.mmad` 的 `unit_flag` 必须能静态证明恒为 0 或恒为 2/3；L0C copy 的 `unit_flag`
  只支持 0 或 3；
- 同一条指令按条件选中多个 buffer 时，各分支的 allocation 加锁顺序必须一致；
- UB 的 `tla.scalar_load`/`scalar_store` 必须放进 `tla.vec.func`；
- `tla.print_tensor` / `tla.debug_print` 不参与自动同步。

设计细节与静态分析约束见
[AutoSync 设计文档](../../dsl_development/feature_development/auto_sync_design.md)。
不确定时优先手写同步：手写同步的行为是确定的，AutoSync 的覆盖范围有明确边界。

## 10. 常见错误速查

| 现象 / 报错 | 原因 | 修法 |
| --- | --- | --- |
| `event id exhausted (max 7)` | 同一 pipe pair 旗标超过 8 个 | 复用 flag、减少并发缓冲，或合并同向同步 |
| `cross flag id exhausted` | cross flag 超过 16 个 | 合并 mode 相同的同步，或复用 flag |
| `wait_flag used without set_flag for 'xxx'` | 只写了 wait，没有对应 set；或名字写错 | 检查名称拼写与配对关系 |
| `mutex id auto allocation is not implemented` | `tla.mutex` 用了默认 `id=-1` | 显式写 `id=0..31` |
| `in vector pipe_barrier only support [...]` | vector 区域写了 PIPE_V 等不支持的 pipe | 改用 `MTE2`/`MTE3`/`ALL` |
| 重复 flag 名编译报错 | 同名 `tla.flag` 定义了两次 | 每个 flag 名称唯一 |
| 运行卡死 | 初始 `available` flag 未预置；或 set/wait 不成对 | 循环前预置、检查配对 |
| 结果偶发错误 | 漏 WAR 同步；末尾缺 `pipe_barrier(ALL)` | 补 buffer 空闲 flag 与收尾 drain |
| 跨核握手失败 | mode 选错、`aiv_id` 缺失、1:1 下 wait pipe 用了 VECTOR | 按第 6 节检查 mode 与 `core_ratio` 约束 |

## 11. 提交前检查清单

- [ ] 每个 `tla.copy` / `tla.mmad` / `vec.func` 的消费者，都确认了它与生产者的流水；
- [ ] 所有跨流水数据依赖都有 set/wait，且 set 与 wait 引用同一个 flag；
- [ ] 双缓冲有初始 `available` flag，且每轮结束后重新 set；
- [ ] 同一 pipe pair 的 flag 数不超过 8，cross flag 不超过 16；
- [ ] `pipe_barrier` 的 pipe 在所在区域的白名单内；
- [ ] kernel 末尾有 `pipe_barrier(tla.pipes.ALL)`；
- [ ] 跨核场景检查 mode、`aiv_id` 与 `core_ratio`；
- [ ] 未同时混用 AutoSync 与手写 flag/mutex。
