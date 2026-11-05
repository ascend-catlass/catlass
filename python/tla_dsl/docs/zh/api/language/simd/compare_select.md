---
nav_order: 54
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/compare_select.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 比较与选择

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

## `tla.where`

**源码：** [`catlass.core_api.where`](../../../../../catlass/core_api.py)

功能说明：

按 mask 在两路 vector 或 predicate mask 间逐元素选择；在 SIMT 区域内也可在两路 per-thread 标量间选择。

函数原型：

```python
tla.where(mask: Any, x: Any, y: Any) -> Any
```

参数说明：

- `mask`（`MaskSSA | Bool`）：选择掩码；真取 `x`，假取 `y`。可为整 vector 的 `MaskSSA`，或 SIMT 区域内比较得到的 `Bool`。必填。
- `x`（`VectorSSA | MaskSSA | Numeric`）：掩码为真时的取值。必填。
- `y`（`VectorSSA | MaskSSA | Numeric`）：掩码为假时的取值。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；SIMD 路径的操作数必须属于同一类别。
- MaskSSA 路径中，操作数和选择掩码必须具有完全相同的类型。
- per-thread 形式要求 `mode="simt"`，并生成 `tla.simt_where`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    z = tla.where(m, x_reg, y_reg)
```

## `tla.cmp`

**源码：** [`catlass.core_api.cmp`](../../../../../catlass/core_api.py)

功能说明：

vector 比较，返回 mask。

函数原型：

```python
tla.cmp(lhs: VectorSSA, rhs: VectorSSA | Numeric | bool | int | float, mode: str, *, mask: MaskSSA | None = None) -> MaskSSA
```

参数说明：

- `lhs`（`VectorSSA`）：比较左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：比较右操作数（可为标量或 vector）。必填。
- `mode`（`str`）：比较模式。必填。

  | 取值 | 含义 |
  | --- | --- |
  | `lt` | 小于 |
  | `le` | 小于等于 |
  | `gt` | 大于 |
  | `ge` | 大于等于 |
  | `eq` | 等于 |
  | `ne` | 不等于 |

- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`mode` 须为上表取值之一。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m = tla.cmp(x_reg, y_reg, mode="lt")
```
