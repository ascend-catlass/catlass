---
nav_order: 53
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/logical.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 逻辑计算

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

Mask / Vector 上的按位与逻辑运算。

## `tla.shift_left`

**源码：** [`catlass.core_api.shift_left`](../../../../../catlass/core_api.py)

功能说明：

按 lane 或标量移位量，对 vector 做逐元素左移。

函数原型：

```python
tla.shift_left(source: VectorSSA, shift: Any, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `source`（`VectorSSA`）：待移位的源 vector。必填。
- `shift`（`VectorSSA | Numeric | int`）：每 lane 移位量。可为与 `source`
  同元素位宽的有符号 `VectorSSA`，或可转为 i16 的整型标量。必填。
- `mask`（`MaskSSA | None`）：执行掩码。可选，默认 `None`（全 lane 有效）；
  被 mask 掉的 lane 写零。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- 源元素须为有符号 8/16/32 位整数（i8/i16/i32）；不支持 64 位与无符号移位。
- 移位量须为非负。

调用示例：

```python
with tla.vec.func(mode="simd"):
    _tile = tla.shift_left(tile, bit_shift_reg, mask=mask)
```

---

## `tla.shift_right`

**源码：** [`catlass.core_api.shift_right`](../../../../../catlass/core_api.py)

功能说明：

按 lane 或标量移位量，对 vector 做逐元素右移。

有符号源 vector 使用算术右移。

函数原型：

```python
tla.shift_right(source: VectorSSA, shift: Any, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `source`（`VectorSSA`）：待移位的源 vector。必填。
- `shift`（`VectorSSA | Numeric | int`）：每 lane 移位量。可为与 `source`
  同元素位宽的有符号 `VectorSSA`，或可转为 i16 的整型标量。必填。
- `mask`（`MaskSSA | None`）：执行掩码。可选，默认 `None`（全 lane 有效）；
  被 mask 掉的 lane 写零。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- 源元素须为有符号 8/16/32 位整数（i8/i16/i32）；不支持 64 位与无符号移位。
- 移位量须为非负。

调用示例：

```python
with tla.vec.func(mode="simd"):
    _tile = tla.shift_right(tile, bit_shift_reg, mask=mask)
```

---

## `tla.bitwise_not`

**源码：** [`catlass.core_api.bitwise_not`](../../../../../catlass/core_api.py)

功能说明：

逐元素按位/逻辑非（Mask 或 Vector）。

函数原型：

```python
tla.bitwise_not(operand: VectorSSA | MaskSSA, *, mask: MaskSSA | None = None) -> MaskSSA | VectorSSA
```

参数说明：

- `operand`（`VectorSSA | MaskSSA`）：按位取反的源操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m2 = tla.bitwise_not(m)
```

---

## `tla.bitwise_and`

**源码：** [`catlass.core_api.bitwise_and`](../../../../../catlass/core_api.py)

功能说明：

逐元素按位与（Mask 或 Vector）。

函数原型：

```python
tla.bitwise_and(src0_reg: VectorSSA | MaskSSA, src1_reg: VectorSSA | MaskSSA, *, mask: MaskSSA | None = None) -> MaskSSA | VectorSSA
```

参数说明：

- `src0_reg`（`VectorSSA | MaskSSA`）：按位与左操作数。必填。
- `src1_reg`（`VectorSSA | MaskSSA`）：按位与右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m3 = tla.bitwise_and(m0, m1)
```

---

## `tla.bitwise_or`

**源码：** [`catlass.core_api.bitwise_or`](../../../../../catlass/core_api.py)

功能说明：

逐元素按位或（Mask 或 Vector）。

函数原型：

```python
tla.bitwise_or(src0_reg: VectorSSA | MaskSSA, src1_reg: VectorSSA | MaskSSA, *, mask: MaskSSA | None = None) -> MaskSSA | VectorSSA
```

参数说明：

- `src0_reg`（`VectorSSA | MaskSSA`）：按位或左操作数。必填。
- `src1_reg`（`VectorSSA | MaskSSA`）：按位或右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m3 = tla.bitwise_or(m0, m1)
```

---

## `tla.bitwise_xor`

**源码：** [`catlass.core_api.bitwise_xor`](../../../../../catlass/core_api.py)

功能说明：

逐元素按位异或（Mask 或 Vector）。

函数原型：

```python
tla.bitwise_xor(src0_reg: VectorSSA | MaskSSA, src1_reg: VectorSSA | MaskSSA, *, mask: MaskSSA | None = None) -> MaskSSA | VectorSSA
```

参数说明：

- `src0_reg`（`VectorSSA | MaskSSA`）：按位异或左操作数。必填。
- `src1_reg`（`VectorSSA | MaskSSA`）：按位异或右操作数。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m3 = tla.bitwise_xor(m0, m1)
```
