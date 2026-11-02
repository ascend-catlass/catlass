---
nav_order: 2
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/values_types.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 值与数据类型

[语言参考](index.md) | [API 速查](../quick_reference.md)

`Numeric` 是 CATLASS Python 在 `@tla.kernel` 内使用的**标量**类型与值的基类。
具体类型包括整数（`tla.Int8` / `Int16` / `Int32` / `Int64` 及无符号变体）、
浮点（`tla.Float16` / `BFloat16` / `Float32`，以及更窄格式）和 `tla.Bool`。
用类型构造值，例如 `tla.Int32(0)`、`tla.Float32(1.5)`。

循环下标、坐标，以及会在运行时 `if` / `tla.range` / `while` 中更新的单元素状态，
请使用 `Numeric` 值。Python 字面量用 `as_numeric` 提升；改类型用 `Numeric.to`；
按位宽重解释用 `Numeric.bitcast`；按位 / 数学运算如 `~Integer`、`Float.sqrt`。

运算符 `+` `-` `*` `/` `//` `**` 与比较运算走核内标量单元；但在
`tla.vec.func(mode="simt")` 区域内各自变为对应的逐线程 op（`tla.simt_add`、
`tla.simt_cmp` 等）。见 [SIMT 向量计算](simt.md#simt-向量计算)。

约束说明：

- 优先使用具体类型（`Int32`、`Float32` 等）。抽象基类 `Numeric` / `Integer` /
  `Float` 不可直接构造。
- 参与设备侧控制流或计算时，须在 `@tla.kernel` 函数体内使用。
- 在运行时 `if` / `tla.range` / `while` 中对名字重新赋值时，该值须是 `Numeric`
  （由 `as_numeric` 或具体类型如 `tla.Int32(...)` 得到）。普通 Python
  `int` / `float` / `bool` 无法作为运行时状态跨过上述控制流。

## `tla.cast`

**源码：** [`catlass.base_dsl.typing.cast`](../../../../catlass/base_dsl/typing.py#L661)

功能说明：

由 `obj` 构造类型为 `type_` 的具体 `Numeric`。

函数原型：

```python
tla.cast(obj: Any, type_: type['Numeric']) -> 'Numeric'
```

参数说明：

- `obj`（`Any`）：源值（Python 标量或 `Numeric`）。必填。
- `type_`（`type[Numeric]`）：具体目标 `Numeric` 子类。必填。

约束说明：

- 建议在 `@tla.kernel` 函数体内调用。
- `type_` 须为具体 `Numeric` 子类（不是抽象基类 `Numeric` / `Integer` /
  `Float`），除非 `obj` 已属于该抽象族。
- 不支持的源/目标组合会抛出 `TypeError`。

调用示例：

```python
@tla.kernel
def kernel() -> None:
    x = tla.cast(0, tla.Int32)
    y = tla.cast(1.5, tla.Float32)
    tla.make_coord(x, y.to(tla.Int32))
```

---

## `tla.Numeric.to`

**源码：** [`catlass.base_dsl.typing.Numeric.to`](../../../../catlass/base_dsl/typing.py#L891)

功能说明：

将标量 `Numeric` 转为另一具体类型。

函数原型：

```python
tla.Numeric.to(dtype: Any, *, round_mode: Any = None) -> Any
```

参数说明：

- `dtype`（`type[Numeric] | type[int] | type[float] | type[bool]`）：
  目标类型。具体 `Numeric` 子类得到另一个 `Numeric`；传入 Python
  `int` / `float` / `bool` 时，若值仍是 Host 常量，则提取 Host 侧 Python
  标量。必填。
- `round_mode`（`ScalarRoundMode | None`）：浮点舍入模式：
  `NEAREST_EVEN`、`NEAREST_AWAY`、`FLOOR`、`CEIL`、`ODD`；省略或
  `TRUNC` 表示 float→int 向零截断。可选，默认 `None`。

约束说明：

- 优先在 `@tla.kernel` 函数体内调用。
- 产出 Python 标量时，`dtype` 须为具体 `Numeric` 子类，不能是抽象的
  `Integer` / `Float` / `Numeric`。
- 此处使用 `tla.params.ScalarRoundMode`；不要传入 `tla.params.RoundMode`。
- 不支持的源/目标组合会抛出 `TypeError`。

调用示例：

```python
@tla.kernel
def kernel(n: int) -> None:
    x = tla.as_numeric(n).to(tla.Int32)
    y = tla.Float32(1.5).to(
        tla.Int32, round_mode=tla.params.ScalarRoundMode.FLOOR
    )
    tla.make_coord(x, y)
```

---

## `tla.Numeric.bitcast`

**源码：** [`catlass.base_dsl.typing.Numeric.bitcast`](../../../../catlass/base_dsl/typing.py#L1030)

功能说明：

将标量 `Numeric` 的位型重新解释为**相同位宽**的另一种具体类型（不做数值转换）。

函数原型：

```python
tla.Numeric.bitcast(dtype: type['Numeric']) -> 'Numeric'
```

参数说明：

- `dtype`（`type[Numeric]`）：相同位宽的具体目标类型。必填。

约束说明：

- 优先在 `@tla.kernel` 函数体内调用。
- `dtype` 须为具体 `Numeric`，且与源值位宽相同；否则抛出 `TypeError` /
  `ValueError`。
- 同类型 bitcast 为 no-op，返回 `self`。

调用示例：

```python
bits = tla.Float32(1.0).bitcast(tla.Int32)
```

---

## `tla.as_numeric`

**源码：** [`catlass.base_dsl.typing.as_numeric`](../../../../catlass/base_dsl/typing.py#L1204)

功能说明：

将 Python 标量转为 Python `Numeric`。

函数原型：

```python
tla.as_numeric(obj: Any) -> Numeric
```

参数说明：

- `obj`（`bool | int | float | Numeric`）：Python 标量或已有 `Numeric`。必填。

约束说明：

- 结果用作运行时状态（控制流重绑定、循环边界等）时，优先在 `@tla.kernel`
  函数体内调用。
- 也可在 Host 侧从 Python 标量构造，用于 staging。
- 不接受用户自定义对象与非标量容器；无法转换时抛出 `ValueError`。

调用示例：

```python
@tla.kernel
def kernel(limit: int) -> None:
    index = tla.as_numeric(0)
    if limit > 0:
        index = index + 1
    tla.make_coord(index, 0)
```

---

## `tla.Integer.__invert__`

**源码：** [`catlass.base_dsl.typing.Integer.__invert__`](../../../../catlass/base_dsl/typing.py#L1253)

功能说明：

整数 `Numeric` 的按位取反：写 `~Integer`。

函数原型：

```python
~Integer -> 'Integer'
```

参数说明：

- 无。对具体整数 `Numeric` 使用一元 `~` 运算符。

约束说明：

- 建议在 `@tla.kernel` 函数体内调用。
- 接收者须为具体整数 `Numeric`。

调用示例：

```python
@tla.kernel
def kernel() -> None:
    flags = ~tla.Int32(0)
    tla.make_coord(flags, 0)
```

---

## `tla.Float.sqrt`

**源码：** [`catlass.base_dsl.typing.Float.sqrt`](../../../../catlass/base_dsl/typing.py#L1306)

功能说明：

标量浮点的平方根。

函数原型：

```python
tla.Float.sqrt() -> 'Float'
```

参数说明：

- （除 `self` 外无额外参数）

约束说明：

- 设备侧值须在 `@tla.kernel` 函数体内调用。
- 在非 `mode="simt"` 时仅支持 `Float32`（`Float16` / `BFloat16` 抛 `TypeError`）；
  请先 `.to(tla.Float32)`，或在 `tla.vec.func(mode="simt")` 内调用。

调用示例：

```python
@tla.kernel
def kernel() -> None:
    r = tla.Float32(4.0).sqrt()
    tla.make_coord(r.to(tla.Int32), 0)
```
