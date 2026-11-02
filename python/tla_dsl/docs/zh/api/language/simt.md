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

逐线程向量执行模式，以 `tla.vec.func(mode="simt", thread_block_dim=...)` 进入。
函数体是*单线程*程序：各线程通过普通元素访问按索引寻址，无 tile、无向量寄存器、
无 UB 暂存。

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
每 block 至多 2048 线程，以及仅允许捕获标量运行期值。片上 scratch 用
`tla.allocate` 在区域外分配（见 [资源管理](resources.md)）；区域结束后在
`tla.vector()` 内做 pipe barrier，如上例。

**SIMT 区域内可写内容**

函数体是逐线程标量代码，可用：

- 对 `gm` / `ub` tensor 的元素访问（`tensor[i]`、`tensor[i] = value`）——见下方
  [元素访问](#元素访问)。
- 对 `Numeric` 的标量算术、比较、选择与转换，各自生成逐线程 op——见下方
  [基础算术](#基础算术)、[比较与选择](#比较与选择)、[类型转换](#类型转换)。
- `tla.range` 循环与运行期 `if`。
- `tla.arch.thread_idx` / `thread_block_dim` / `sync_threads`，以及 launch 级
  `tla.arch.block_idx` / `block_num`——见 [系统变量访问](system.md#系统变量访问)。

编译时会拒绝：

- 寄存器向量与 mask 相关——`VectorSSA` / `MaskSSA`、[SIMD 向量计算](simd/index.md#simd-向量计算)
  下的算术与 mask 接口、`Tensor.load` / `Tensor.store`、`tla.full`、
  `tla.create_mask`。SIMT 线程没有自己的向量寄存器。
- `tla.copy`：DMA 需要捕获 tensor 本身，而 SIMT 区域只能捕获标量运行期值。
  可在进入区域前完成搬数，或让各线程直接按索引访问全局内存。
- `tla.local_mem_bar`、`tla.pipe_barrier`、`tla.print`。
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
