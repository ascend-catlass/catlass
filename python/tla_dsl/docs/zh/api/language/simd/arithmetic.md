---
nav_order: 52
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/arithmetic.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 基础算术

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

## `tla.exp`

**源码：** [`catlass.core_api.exp`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素指数（需 f16/f32），或逐线程标量指数。

函数原型：

```python
tla.exp(operand: VectorSSA | Numeric, *, mask: MaskSSA | None = None) -> VectorSSA | Numeric
```

参数说明：

- `operand`（`VectorSSA | Numeric`）：源 vector 寄存器，或浮点标量。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。仅 vector 形态。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- vector 形态须在 `tla.vec.func()` 内调用；元素类型须为 f16/f32。
- 在 `tla.vec.func(mode="simt")` 区域内，标量操作数生成逐线程
  `tla.simt_exp`；区域外标量形态会被拒绝，因为核内标量单元没有指数运算。
- 标量形态仅支持浮点，且不接受 `mask`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.exp(x_reg)
```

---

## `tla.log`

**源码：** [`catlass.core_api.log`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素对数（需 f16/f32），或逐线程标量对数。

函数原型：

```python
tla.log(operand: VectorSSA | Numeric, *, mask: MaskSSA | None = None) -> VectorSSA | Numeric
```

参数说明：

- `operand`（`VectorSSA | Numeric`）：源 vector 寄存器，或浮点标量。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。仅 vector 形态。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- vector 形态须在 `tla.vec.func()` 内调用；元素类型须为 f16/f32。
- 在 `tla.vec.func(mode="simt")` 区域内，标量操作数生成逐线程
  `tla.simt_log`；区域外标量形态会被拒绝，因为核内标量单元没有对数运算。
- 标量形态仅支持浮点，且不接受 `mask`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.log(x_reg)
```

---

## `tla.sqrt`

**源码：** [`catlass.core_api.sqrt`](../../../../../catlass/core_api.py)

功能说明：

vector 或标量的平方根。

对 `VectorSSA`：逐元素 AVE 运算（需 f16/f32）。
对浮点 `Numeric`：走核内标量单元，可在 `tla.cube`、SIMD/SIMT 的
`tla.vec.func` 或 kernel 函数体中使用。SIMT 区域内为逐线程
`tla.simt_sqrt`；其余路径为 `math.sqrt`。浮点 `Numeric` 上的 `x.sqrt()`
是同一操作。

函数原型：

```python
tla.sqrt(operand: VectorSSA | Numeric, *, mask: MaskSSA | None = None) -> VectorSSA | Numeric
```

参数说明：

- `operand`（`VectorSSA | Numeric`）：源 vector 寄存器，或浮点标量。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。仅 vector 形态。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- vector 形态须在 `tla.vec.func()` 内调用；元素类型须为 f16/f32。
- 标量形态仅支持浮点，且不接受 `mask`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.sqrt(x_reg)

with tla.cube():
    s = tla.sqrt(scale)
```

---

## `tla.abs`

**源码：** [`catlass.core_api.abs`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素绝对值。

函数原型：

```python
tla.abs(operand: VectorSSA | Numeric, *, mask: MaskSSA | None = None) -> VectorSSA | Numeric
```

参数说明：

- `operand`（`VectorSSA | Numeric`）：源 vector 寄存器，或浮点标量。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。仅 vector 形态。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- vector 形态须在 `tla.vec.func()` 内调用。
- 在 `tla.vec.func(mode="simt")` 区域内，标量操作数生成逐线程
  `tla.simt_abs`；区域外对标量调用 `tla.abs` 会被拒绝。Python 内建 `abs(x)`
  在两种位置均可用：SIMT 区域内为 `tla.simt_abs`，其余位置为 `math.absf`。
- 标量形态仅支持浮点，且不接受 `mask`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.abs(x_reg)
```

---

## `tla.neg`

**源码：** [`catlass.core_api.neg`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素取负。

函数原型：

```python
tla.neg(operand: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `operand`（`VectorSSA`）：源 vector 寄存器。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.neg(x_reg)
```

---

## `tla.add`

**源码：** [`catlass.core_api.add`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素加法（支持 vector–vector 与 vector–scalar）。
`VectorSSA` 在不需要 `mask` 时也可通过 `+` / `__radd__` 调用本接口。

函数原型：

```python
tla.add(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；支持 vector–vector 与 vector–scalar。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 计算前：x_reg、y_reg 已从 UB load 到寄存器。
    z = x_reg + y_reg            # 等价于 tla.add(x_reg, y_reg)
    z = x_reg + 1.0              # vector–scalar，走 __add__/__radd__
    # 带 mask：先 create_mask 得到 MaskSSA，再传 mask=（不能用运算符重载）。
    m = tla.create_mask(pattern=tla.mask.VL16, dtype=tla.Float16)
    z = tla.add(x_reg, y_reg, mask=m)
    # 计算后：有效元素为和；其余元素不参与。
```

---

## `tla.sub`

**源码：** [`catlass.core_api.sub`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素减法。
`VectorSSA` 在不需要 `mask` 时也可通过 `-`（`__sub__`）调用本接口。

函数原型：

```python
tla.sub(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数（被减数）。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数（减数）。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 计算前：x_reg / y_reg 为源 vector。
    z = x_reg - y_reg                 # 等价于 tla.sub(x_reg, y_reg)
    z = tla.sub(x_reg, y_reg, mask=m) # 需要 mask 时用函数形式
    # 计算后：z 为逐元素差。
```

---

## `tla.mul`

**源码：** [`catlass.core_api.mul`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素乘法。
`VectorSSA` 在不需要 `mask` 时也可通过 `*` / `__rmul__` 调用本接口。

函数原型：

```python
tla.mul(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 计算前：x_reg 为激活；scale 可为 vector 或标量。
    z = x_reg * y_reg      # 等价于 tla.mul(x_reg, y_reg)
    z = x_reg * 2.0        # vector–scalar
    z = tla.mul(x_reg, y_reg, mask=m)
    # 计算后：z 为逐元素积。
```

---

## `tla.max`

**源码：** [`catlass.core_api.max`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素最大值。

函数原型：

```python
tla.max(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA | Numeric
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    z = tla.max(x_reg, y_reg)
```

---

## `tla.min`

**源码：** [`catlass.core_api.min`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素最小值。

函数原型：

```python
tla.min(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA | Numeric
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    z = tla.min(x_reg, y_reg)
```

---

## `tla.div`

**源码：** [`catlass.core_api.div`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素除法。
`VectorSSA` 在不需要 `mask` 时也可通过 `/`（`__truediv__`）调用本接口。

函数原型：

```python
tla.div(lhs: VectorSSA | Numeric | bool | int | float, rhs: VectorSSA | Numeric | bool | int | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA | Numeric | bool | int | float`）：左操作数（被除数）。必填。
- `rhs`（`VectorSSA | Numeric | bool | int | float`）：右操作数（除数）。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 计算前：x_reg 为被除数；y_reg / 标量为除数。
    z = x_reg / y_reg                 # 等价于 tla.div(x_reg, y_reg)
    z = tla.div(x_reg, y_reg, mask=m)
    # 计算后：z 为逐元素商。
```
