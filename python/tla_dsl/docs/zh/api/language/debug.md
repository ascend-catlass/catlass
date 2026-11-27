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

kernel 内标量 / tensor 调试打印与异常终止。

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

- 须在 `tla.cube()`／`tla.vector()` 或其中的 SIMD／SIMT `tla.vec.func` 内调用；不同线程、核和打印通道之间不保证输出顺序。
- 支持的 dtype 为 `f16`、`f32`、`i8`、`i16`、`i32`、`u8`、`u16`、`u32`。Cube 打印 GM／L1／L0C Tensor，外层 Vector 打印 GM／UB Tensor，SIMD 打印标量、寄存器和 UB Tensor，SIMT 打印线程标量和静态形状的 GM／UB Tensor。
- 字符串须为 host 侧 `str`，仅含 ASCII 字符且不含 NUL 字节。格式串最多 8 个空 `{}` 字段，字段数须与参数个数一致；`{:d}`、`{0}`、`{name}` 等格式说明符、索引和名称会被拒绝。`{{`、`}}` 输出字面花括号，`%` 按字面输出。
- GM／UB Tensor 须为一维或二维；从视图起始地址连续取数，不按 stride 选取元素。调用方保证数据已就绪且这段内存可读。
- `length` 为整数，单位为元素，仅当 Tensor 形状静态且全部元素能在一次调用内打印时可省略。寄存器默认使用有效元素数，该值为动态值时使用物理容量。
- L1 须位于 Cube，为 32 字节对齐的 zN/nZ 稠密前缀，C0 维度占 32 字节，最多 8 个元素；L0C 须位于 Cube，使用 L0Clayout 和 32 字节对齐的 16x16 视图（256 个元素）。L1 与 L0C 不能同时打印，L0A／L0B 不支持。
- SIMD 会安排打印取数与前后 Vector store 的顺序。SIMT 打印不代替线程同步；共享 UB 需要全线程屏障。其他数据依赖由调用方负责。
- 打印增加缓冲区、传输和设备流同步开销；性能测量前移除调用。
- 单次上限、对齐、运行时长度、同步示例、日志容量、截断及异常恢复规则见打印指南。

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

## `tla.trap`

**源码：** [`catlass.core_api.trap`](../../../../catlass/core_api.py)

功能说明：

以 trap 异常终止 AI Core：设备上核心停止、kernel 启动失败，该 kernel 的全部
输出未定义。

函数原型：

```python
tla.trap() -> None
```

参数说明：

- 无。

约束说明：

- 须在 `@tla.kernel` 函数内调用：kernel 函数体本身（未包在 `tla.cube()` /
  `tla.vector()` 内），或这两个区域内的标量代码（AIC / AIV 标量单元）。写在
  SIMD / SIMT 的 `tla.vec.func` 内会在编译期被拒绝；`vec.func` 内的降级尚未支持，
  当前请把判断移到区域外。
- 仅在 kernel 必须失败时调用。`tla.trap` 无条件执行且不返回，其后的代码在
  运行期不可达。
- trap 之后 stream 与 context 不可恢复。应退出进程，不要复用 stream 或再次
  launch。

调用示例：

```python
with tla.vector():
    divisor = src_gm[0]
    if divisor == 0:
        tla.trap()
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
