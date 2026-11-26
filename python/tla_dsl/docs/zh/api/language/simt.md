---
nav_order: 6
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simt.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# SIMT 向量计算

[语言参考](index.md) | [API 速查](../quick_reference.md)

SIMT 是逐线程的向量执行模式，以 `tla.vec.func(mode="simt", thread_block_dim=...)` 进入。
函数体描述每个线程执行的代码。线程通过 `tla.arch.thread_idx()` 获取自身索引，使用 `tensor[index]` 读写 GM／UB 中的元素，并通过标量运算处理数据。支持的接口和限制见下方说明。

```python
with tla.vector():
    with tla.vec.func(mode="simt", thread_block_dim=400):
        tid, _, _ = tla.arch.thread_idx()
        nthreads, _, _ = tla.arch.thread_block_dim()
        for i in tla.range(tid, 400, nthreads):
            gm_c[i] = gm_a[i] + gm_b[i]
    tla.pipe_barrier(tla.pipes.ALL)
```

子节顺序与 [SIMD 向量计算](simd/index.md#simd-向量计算) 对齐：先说明区域能写什么，再说明线程如何搬数与计算。
线程索引 / block 几何 / `sync_threads` 见 [系统变量访问](system.md#系统变量访问)。

## 区域范围

区域由 `tla.vec.func` 声明（见 [作用域和控制流](scopes.md)），负责几何规则、
每 block 至多 2048 线程，以及捕获规则：标量运行期值，以及通过 `t.ptr` /
`t.ptr_with_coord` 读取的 tensor 指针。片上 scratch 用
`tla.allocate` 在区域外分配（见 [资源管理](resources.md)）；区域结束后在
`tla.vector()` 内做 pipe barrier，如上例。

**SIMT 区域内可写内容**

函数体是逐线程标量代码，可用：

- 对 `gm` / `ub` tensor 的元素访问（`tensor[i]`、`tensor[i] = value`）——见下方
  [元素访问](#元素访问)。
- tensor 指针：`tensor.ptr`（原始存储基址）与 `tensor.ptr_with_coord`（视图起始地址），
  支持元素偏移运算——见下方 [指针访问](#指针访问)。
- 对 `gm` / `ub` 元素的原子累加（`tla.arch.atomic_add(ptr, value)`）——见
  [系统变量访问](system.md#tlaarchatomic_add)。
- 对 `Numeric` 的标量算术、比较、选择与转换，各自生成逐线程 op——见下方
  [基础算术](#基础算术)、[比较与选择](#比较与选择)、[类型转换](#类型转换)。
- `tla.range` 循环与运行期 `if`。
- `tla.print`：逐线程标量、格式化字符串及 GM／UB 前缀，见 [调试与打印](debug.md)。
- `tla.arch.warp_reduce_sync`：对 Warp 内的活跃线程执行 add/max/min 归约——见
  [系统变量访问](system.md#系统变量访问)。
- Warp 查询 `tla.arch.warpsize` / `lane_idx` / `lanemask_eq` / `lanemask_lt` /
  `lanemask_le` / `lanemask_gt` / `lanemask_ge` / `activemask`——见 [系统变量访问](system.md#系统变量访问)。
- `tla.arch.thread_idx` / `thread_block_dim` / `sync_threads`，以及 launch 级
  `tla.arch.block_idx` / `block_num`——见 [系统变量访问](system.md#系统变量访问)。

编译时会拒绝：

- 寄存器向量与 mask 相关——`VectorSSA` / `MaskSSA`、[SIMD 向量计算](simd/index.md#simd-向量计算)
  下的算术与 mask 接口、`Tensor.load` / `Tensor.store`、`tla.full`、
  `tla.create_mask`。这些接口用于 SIMD 寄存器向量编程。
- `tla.copy`：DMA 需要捕获 tensor 本身，而 SIMT 区域只能捕获标量运行期值和
  tensor 指针。可在进入区域前完成搬数，或让各线程直接按索引访问全局内存。
- `tla.local_mem_bar`、`tla.data_sync_barrier`、`tla.pipe_barrier`。
- 任意 `tla.vec.func` 内的 `while`；迭代请用 `tla.range`。
- 嵌套的 `tla.vec.func`。

## 元素访问

线程通过对 `gm` / `ub` tensor 的元素访问搬数：`value = tensor[i]` 与
`tensor[i] = value`，生成逐线程 `tla.simt_load` / `tla.simt_store`。布局须为
`RowMajor` 或 `ColumnMajor`，逻辑秩为 1 或 2；除 `Bool` 外不支持亚字节元素类型。
二者文档见 [数据搬运](data_movement.md#数据搬运) 中的 `tensor[i]` / `tensor[i] = value`。

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    tid, _, _ = tla.arch.thread_idx()
    gm_out[tid] = gm_in[tid]
```

## 指针访问

`tensor.ptr` 返回 tensor 的原始存储基址（`Pointer`）；`tensor.ptr_with_coord`
返回视图起始地址，即该基址向前 `coord[0]*stride[0] + coord[1]*stride[1]` 个
元素处。二者在 kernel 标量代码与 SIMT 区域内均可用，且都支持元素偏移运算
（`t.ptr + offset`）。Dynamic-GM Tensor 形参以 `arg.ptr` /
`arg.ptr_with_coord` 暴露同样的两个属性。

`tla.tile_view` 的坐标按 tile 粒度给出，因此 8 行 shape 上的
`tla.make_coord(block, 0)` 使视图从第 `block*8` 行开始。运行期坐标（如
`tla.arch.block_idx()` 或循环变量）受支持：偏移由该视图的
`tile_view` / `tla.make_tensor` 操作数计算得出。经运行期控制流选择的视图没有
唯一来源：应在各分支先取 `t.ptr_with_coord`（或 `t.ptr` 取原始基址），再选择指针
（如 `va.ptr_with_coord if cond else vb.ptr_with_coord`），不要先选择 tensor。

`ptr_with_coord` 要求线性布局（`RowMajor`、`ColumnMajor` 或
`VectorLayout`）且逻辑秩为 1 或 2，因为偏移只覆盖两个维度。在 SIMT 区域内读取
`t.ptr` 会把 tensor 的存储作为一个裸指针 launch 参数传入（GM 或 UB，静态
shape）；`ptr_with_coord` 复用区域元素访问已有的视图指针，不占用额外参数。

打包的亚字节元素类型（如 `tla.Float4E2M1`）仍按逻辑元素计偏移：两个 FP4
元素共用一个字节，地址按每元素半个字节前进，因此视图起点必须是 2 的倍数。
编译期已知起点落在半字节处会报错；运行期起点同样须为偶数，并按打包系数换算，
与 Ascend C 的 FP4 地址换算一致。

```python
row = tla.arch.block_idx()
tail = tla.tile_view(gm_out, tla.make_shape(8, 16), tla.make_coord(row, 0))

with tla.vec.func(mode="simt", thread_block_dim=256):
    tid, _, _ = tla.arch.thread_idx()
    view_ptr = tail.ptr_with_coord  # 原始基址 + row*8 行
    raw_ptr = tail.ptr              # 原始存储基址
```

## 基础算术

`Numeric` 上的算术与一元 math 在 SIMT 区域内仍用与 SIMD 相同的写法，但生成逐线程
op：`+` / `-` / `*` / `/` 变为 `tla.simt_add` / `simt_sub` / `simt_mul` /
`simt_div`；`tla.exp` / `log` / `abs` / `sqrt` / `max` / `min` 变为对应的
`simt_` 形态。条目与 [SIMD 向量计算 / 基础算术](#基础算术) 共用，各条约束说明中写明
SIMT 形态。

`tla.exp` 与 `tla.log` *仅* 在此模式可用：核内标量单元没有它们，SIMT 区域外的
标量形态会被拒绝。

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    tid, _, _ = tla.arch.thread_idx()
    x = gm_in[tid]
    gm_out[tid] = tla.sqrt(tla.exp(x) + tla.abs(x))
```

## 比较与选择

比较生成 `tla.simt_cmp`，结果为 `Bool` 而非 `MaskSSA`；对 `Bool` 条件的
`tla.where` 生成逐线程 `tla.simt_where`。条目与
[SIMD 向量计算 / 比较与选择](#比较与选择) 共用。

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    tid, _, _ = tla.arch.thread_idx()
    a = gm_a[tid]
    gm_out[tid] = tla.where(a > 0.0, a, -a)
```

## 类型转换

`Numeric.to` 在 SIMT 区域内生成逐线程 `tla.simt_cast`，区域外走核内标量转换；
条目见 [标量计算](values_types.md)。寄存器向量的 `tla.cast` 没有逐线程对应物。

```python
with tla.vec.func(mode="simt", thread_block_dim=256):
    tid, _, _ = tla.arch.thread_idx()
    gm_f32[tid] = gm_f16[tid].to(tla.Float32)
```
