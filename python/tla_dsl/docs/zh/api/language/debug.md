---
nav_order: 10
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/debug.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 调试与打印

[语言参考](index.md) | [API 速查](../quick_reference.md)

kernel 内标量 / tensor 调试打印。

## `tla.print`

**源码：** [`catlass.core_api.print`](../../../../catlass/core_api.py)

功能说明：

在 kernel 内打印标量、格式化字符串、Tensor 连续前缀或 SIMD 寄存器值。

函数原型：

```python
tla.print(*args: object, **kwargs: object) -> None
```

参数说明：

- *`args`*（`object`）：要打印的值（可变位置参数）。必填。
- **`kwargs`**（`object`）：不接受关键字参数；传入则报错。可选。

约束说明：

- 须在 `@tla.kernel` 函数的 `tla.cube()`／`tla.vector()` 或其中的 SIMD／SIMT `tla.vec.func` 内调用。这些位置的打印可以组合；不同线程、核和打印通道之间不保证输出顺序。
- 支持的 dtype 为 `f16`、`f32`、`i8`、`i16`、`i32`、`u8`、`u16`、`u32`。
- 字符串须为编译期 ASCII 字符串。格式串最多有 8 个 `{}` 标量字段，`{{`、`}}` 输出花括号。
- 在 `vec.func` 外，Cube 支持 GM／L1／L0C Tensor，Vector 支持 GM／UB Tensor。SIMD 支持标量、寄存器和 UB Tensor；SIMT 支持线程标量和静态形状的 GM／UB Tensor。
- GM／UB Tensor 须为一维或二维。从视图起始地址连续取数，不按 stride 选取元素，也不重排布局。调用方保证数据已就绪，且这段连续内存可读。
- `length` 为整数，单位为元素。只有 Tensor 形状静态且全部元素能在一次调用内打印时，才可省略。静态长度须为正，且不超过视图元素数和单次上限。运行时长度按执行模式处理，具体规则见打印指南。
- 寄存器默认打印编译期有效元素数；该值为动态值时，默认使用物理容量。显式长度以物理容量为上限；未初始化的元素没有确定的打印值。
- L1 须位于 Cube，且为 32 字节对齐的 zN/nZ 稠密前缀，C0 维度占 32 字节，最多打印 8 个元素。L0C 须位于 Cube，使用 L0Clayout 和一个 32 字节对齐的 16x16 视图（256 个元素）。同一 kernel 不能同时打印 L1 和 L0C；L0A/L0B 不支持打印。
- SIMD 会安排好打印取数与前后 Vector store 的顺序。SIMT 打印不代替线程同步，共享 UB 需要全线程屏障。算法中的其他数据依赖由调用方表达。
- 打印增加缓冲区、传输和设备流同步开销。性能测量前移除调用；运行时条件跳过打印仍可能保留这些开销。
- 单次上限、对齐、运行时长度、同步示例、日志容量、截断及异常恢复规则见 [打印指南](../../kernel_development/evaluation/print.md)。

调用示例：

```python
# source_gm 和 source_ub 含有 64 个已就绪的 FP32 元素。
with tla.vector():
    tla.print("length={}", 64)
    tla.print(source_gm, 8)
    with tla.vec.func(mode="simd"):
        values = source_ub.load()
        tla.print(values, 4)
        tla.print(source_ub, 4)

    with tla.vec.func(mode="simt", thread_block_dim=32):
        thread, _, _ = tla.arch.thread_idx()
        if thread < 4:
            tla.print("thread={} value={}", thread, source_gm[thread])
        if thread == 0:
            tla.print(source_gm, 8)
```

---

## `tla.prof_start`

**源码：** [`catlass.core_api.prof_start`](../../../../catlass/core_api.py)

在当前位置启动尽力而为的设备性能采集。在 A5 上，通过 `msprof op` 的
`KernelScale` 指标采集。降低为原生序列：`PIPE_ALL` 屏障、设置 CTRL 位 0、
`PIPE_ALL` 屏障。

```python
tla.prof_start() -> None
```

无参数。每个 kernel 特化仅使用一对 start/end。调用方负责位置、执行次数和
同步；不提供计时保证。

```python
with tla.cube():
    tla.prof_start()
    tla.mmad(acc, lhs, rhs, init_c=True)
    tla.prof_end()
```

## `tla.prof_end`

**源码：** [`catlass.core_api.prof_end`](../../../../catlass/core_api.py)

结束 `tla.prof_start()` 开启的区间。降低为原生序列：`PIPE_ALL` 屏障、清除
CTRL 位 0、`PIPE_ALL` 屏障。

```python
tla.prof_end() -> None
```

无参数。构建 IR 时要求存在尚未结束的 start，但不证明动态配对。错误的位置可能
导致编译失败或 kernel 挂起。

```python
with tla.cube():
    tla.prof_start()
    tla.mmad(acc, lhs, rhs, init_c=True)
    tla.prof_end()
```

## `tla.prof_scope`

**源码：** [`catlass.core_api.prof_scope`](../../../../catlass/core_api.py)

在代码块前后发出相同的 start/end 对，不移动或扩大区间。仅添加原生性能采集
屏障；这些屏障会改变重叠关系，不保证得到隔离的计时。

```python
tla.prof_scope() -> _ProfScope
```

无参数。每个 kernel 特化仅使用一个区间。构建 IR 时拒绝可直接检测的嵌套或重复
配对，但不证明执行次数和同步。在 A5 上验证的 Vector 位置是在标量调用方中包围
完整的 `tla.vec.func` 调用；不支持的位置可能编译失败。不承诺支持 SIMT、多个
区间或嵌套区间。

```python
with tla.vector():
    with tla.prof_scope():
        with tla.vec.func(mode="simd"):
            dst.store(tla.add(lhs.load(), rhs.load()))
```
