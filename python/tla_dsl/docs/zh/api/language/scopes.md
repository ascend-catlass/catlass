---
nav_order: 11
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/scopes.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 作用域与控制流

[语言参考](index.md) | [API 速查](../quick_reference.md)

Cube / Vector / `vec.func` 区域以及 kernel 侧循环范围。

从这些 region 内调用的外层 `def`、`tla.Constexpr` 函数入参与 `@tla.jit` helper，须遵守本节各 region 约束，以及所用算子条目的「约束说明」。

## `tla.range`

**源码：** [`catlass.core_api.range`](../../../../catlass/core_api.py)

功能说明：

创建前端动态循环范围（支持 Python `range` 形态）。

函数原型：

```python
tla.range(start: IndexLike, end: IndexLike | None = None, step: IndexLike | None = None) -> _ast_helpers.FrontendRange
```

参数说明：

- `start`（`IndexLike`）：循环起点（或当 `end` 缺省时的终点，此时起点为 0）。必填。
- `end`（`IndexLike | None`）：循环终点（不含）；缺省时 `start` 视为终点。可选，默认 `None`。
- `step`（`IndexLike | None`）：步长；缺省为 1。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 动态范围；循环体须符合前端动态 for 约束。

调用示例：

```python
for i in tla.range(0, n, 1):
    ...
```

---

## `tla.range_constexpr`

**源码：** [`catlass.core_api.range_constexpr`](../../../../catlass/core_api.py)

功能说明：

创建前端静态范围，用于可展开的 Python 循环。

函数原型：

```python
tla.range_constexpr(start: int, end: int | None = None, step: int | None = None) -> range
```

参数说明：

- `start`（`int`）：编译期循环起点（或当 `end` 缺省时的终点）。必填。
- `end`（`int | None`）：编译期循环终点（不含）。可选，默认 `None`。
- `step`（`int | None`）：编译期步长；缺省为 1。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 起止与步长须为编译期常量，用于可展开循环。
- 边界也可来自 `tla.as_numeric(...)` 等编译期 Numeric 值。
- 迭代次数达到 64 次及以上时发出 `DSLOptimizationWarning`，但继续展开；大循环应优先用 `tla.range(...)`。

调用示例：

```python
for k in tla.range_constexpr(0, 4):
    ...
```

---

## `tla.cube`

**源码：** [`catlass.core_api.cube`](../../../../catlass/core_api.py)

功能说明：

进入 cube 核区域（矩阵乘与相关搬运）。

函数原型：

```python
tla.cube() -> TlaRegion
```

参数说明：

无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 用于包裹 cube 侧矩阵乘与相关搬运。

调用示例：

```python
with tla.cube():
    tla.mmad(acc=l0c, lhs=l0a, rhs=l0b, init_c=True)
```

---

## `tla.vector`

**源码：** [`catlass.core_api.vector`](../../../../catlass/core_api.py)

功能说明：

进入 vector 核区域（vector 搬运与同步）。

函数原型：

```python
tla.vector() -> TlaRegion
```

参数说明：

无。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 用于包裹 vector 侧搬运与同步；`tla.vec.func` 须嵌套在其内。

调用示例：

```python
with tla.vector():
    tla.copy(dst=x_ub, src=x_gm)
```

---

## `tla.vec.func`

**源码：** [`catlass.core_api._vec_func`](../../../../catlass/core_api.py)

功能说明：

进入 vector 函数区域（`tla.vec.func`），模式为 `simd` 或 `simt`。

函数原型：

```python
tla.vec.func(*, mode: str = 'simd', thread_block_dim: int | tuple[int, int, int] | list[int] | None = None) -> TlaRegion
```

参数说明：

- `mode`（`str`）：`simd`（默认，寄存器 vector / mask）或 `simt`（逐线程）。
  可选，默认 `"simd"`。
- `thread_block_dim`（`int | tuple[int, int, int] | list[int] | None`）：SIMT
  thread-block 形状；仅在 `mode="simt"` 时有效。传入 `int n` 表示 `(n, 1, 1)`。
  可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须嵌套在 `tla.vector()` 内。
- 寄存器 vector / mask API 与 `local_mem_bar` 须在此区域内调用。
- `mode` 须为 `"simd"` 或 `"simt"`。
- `thread_block_dim` 在 `mode="simt"` 时必填，在 `mode="simd"` 时禁止。
- `thread_block_dim` 各分量须为正，乘积不超过 2048。
- SIMT 区域只能从 kernel 体捕获标量运行期值。
- SIMT：片上 scratch 用 `tla.allocate` 在区域外分配；用
  `tla.arch.thread_idx` / `thread_block_dim` 索引，用 `tla.arch.sync_threads`
  同步。

调用示例：

```python
with tla.vector():
    with tla.vec.func(mode="simd"):
        z = tla.add(x_reg, y_reg)
```
