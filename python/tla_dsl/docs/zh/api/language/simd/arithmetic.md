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

## `tla.arch.sub_exp`

**源码：** [`catlass.core_api.arch_sub_exp`](../../../../../catlass/core_api.py)

功能说明：

使用原生 ExpSub 指令逐元素计算差值的自然指数。对每个输出下标 `i`，`reg_slot`
选定输入下标 `j`。当 `j` 位于两个输入的有效范围内且掩码启用该位置时，结果按下式计算。

```text
out[i] = exp(fp32(lhs[j]) - fp32(rhs[j]))
```

`fp32(x)` 表示在相减前将 `x` 转为 FP32。其余输出元素填入 FP32 正零（`+0.0`，ZEROING）。

函数原型：

```python
tla.arch.sub_exp(lhs: VectorSSA, rhs: VectorSSA, *, reg_slot: RegSlot = RegSlot.ZERO, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `lhs`（`VectorSSA`）：FP16 或 FP32 输入。必填。
- `rhs`（`VectorSSA`）：元素类型和物理宽度均与 `lhs` 相同的输入寄存器。必填。
- `reg_slot`（`RegSlot`）：决定每个输出位置 `i` 对应的输入位置 `j`，下标从 0 开始。
  对于 FP16 输入，`ZERO` 取偶数下标 `j = 2*i`，`ONE` 取奇数下标 `j = 2*i + 1`。
  例如，`ONE` 将输入位置 1、3、5 的计算结果依次放在输出位置 0、1、2。
  对于 FP32 输入，两种取值都用 `j = i`。默认值为 `RegSlot.ZERO`，仅接受关键字传参。
- `mask`（`MaskSSA | None`）：按输入元素位置指定的可选掩码。
  默认值为 `None`，启用有效输入范围内的所有位置。仅接受关键字传参。

约束说明：

- 须在 `@tla.kernel` 和 `tla.vec.func(mode="simd")` 内调用。
- 两个输入的元素类型须同为 FP16 或同为 FP32。某个输入的有效元素数已知时，
  其有效范围从首个元素开始，长度为该数量。两个输入的有效元素数均已知时，取较小值。
- 输出包含 64 个 FP32 元素。
- ZEROING 也适用于输入为非有限值的情况。FP16 输入仅首个元素有效时，
  `RegSlot.ONE` 输出全零。
- 不支持标量和 SIMT 形态、其他输入类型、其他寄存器 slot，以及掩码合并结果。

调用示例：

```python
with tla.vec.func(mode="simd"):
    even = tla.arch.sub_exp(lhs_half, rhs_half)
    odd = tla.arch.sub_exp(
        lhs_half, rhs_half, reg_slot=tla.params.RegSlot.ONE
    )
    result = tla.arch.sub_exp(lhs_float, rhs_float, mask=None)
```

---

## `tla.log`

**源码：** [`catlass.core_api.log`](../../../../../catlass/core_api.py)

功能说明：

vector 逐元素对数（需 f16/f32），或逐线程标量对数。

函数原型：

```python
tla.log(operand: VectorSSA | Numeric, *, mask: MaskSSA | None = None, mode: LogAlgo = LogAlgo.FLUSH_TO_ZERO) -> VectorSSA | Numeric
```

参数说明：

- `operand`（`VectorSSA | Numeric`）：源 vector 寄存器，或浮点标量。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示全有效。仅 vector 形态。可选，默认 `None`。
- `mode`（`tla.params.LogAlgo`）：vector 形态的次正规数处理方式。
  `tla.params.LogAlgo.FLUSH_TO_ZERO`（默认）沿用 AVE `vln` 路径：在
  Ascend 950 上，f32 的 `vln` 会把正次正规输入（`0 < x < 2^-126`）按 0
  处理并返回 `-inf`，而 f16 的 `vln` 路径本身就能正确计算正次正规输入；
  两种模式下负输入均返回 NaN，`+/-0.0` 均返回 `-inf`。
  `tla.params.LogAlgo.SUBNORMAL` 始终执行软件实现，两种 dtype 都能计算
  正次正规输入。可选，默认 `tla.params.LogAlgo.FLUSH_TO_ZERO`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- vector 形态须在 `tla.vec.func()` 内调用；元素类型须为 f16/f32。
- 在 `tla.vec.func(mode="simt")` 区域内，标量操作数生成逐线程
  `tla.simt_log`；区域外标量形态会被拒绝，因为核内标量单元没有对数运算。
- 标量形态仅支持浮点，且不接受 `mask`。
- `mode=SUBNORMAL` 仅适用于 SIMD vector 形态：逐线程标量形态与
  `tla.vec.func(mode="simt")` 都会拒绝。

调用示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.log(x_reg)
    z = tla.log(x_reg, mode=tla.params.LogAlgo.SUBNORMAL)
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

## `tla.addc`

**源码：** [`catlass.core_api.addc`](../../../../../catlass/core_api.py)

功能说明：

在每个有效 lane 上将操作数的位模式与输入进位相加，返回 `(sum, carry_out)`，即和的低 32 位与进位。

函数原型：

```python
tla.addc(lhs: VectorSSA, rhs: VectorSSA, carry_in: MaskSSA, *, mask: MaskSSA | None = None) -> tuple[VectorSSA, MaskSSA]
```

参数说明：

- `lhs`（`VectorSSA`）：左操作数。必填。
- `rhs`（`VectorSSA`）：右操作数。必填。
- `carry_in`（`MaskSSA`）：逐 lane 的输入进位谓词。必填。
- `mask`（`MaskSSA | None`）：执行掩码。默认 `None`，表示所有 lane 有效。

约束说明：

- 在 Ascend 950 上的 `tla.vec.func(mode="simd")` 内调用。
- 两个操作数须为同类型的 64-lane `i32` 或 `u32` 向量。显式传入的 `mask` 须有 64 个 lane。
- `carry_in` 和 `carry_out` 均为 64-lane `MaskSSA` 谓词（`!tla.mask<64>`），每个 lane 占 1 bit。
- `carry_in` 为 `1` 表示有输入进位，该 lane 的和加一。为 `0` 表示无输入进位。
- `carry_out` 为 `1` 表示该 lane 的和向第 33 位进位。为 `0` 表示无输出进位。
- 对 `mask` 未启用的 lane，`carry_out` 未定义。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # ALLF 将所有输入位设为 0，表示无输入进位。
    carry_in = tla.create_mask(pattern=tla.mask.ALLF, dtype=tla.UInt32)
    total, carry_out = tla.addc(lhs, rhs, carry_in)
```

---

## `tla.subc`

**源码：** [`catlass.core_api.subc`](../../../../../catlass/core_api.py)

功能说明：

在每个有效 lane 上计算 `lhs - rhs - (1 - carry_in)`，返回 `(diff, carry_out)`，即差的低 32 位与进位。

函数原型：

```python
tla.subc(lhs: VectorSSA, rhs: VectorSSA, carry_in: MaskSSA, *, mask: MaskSSA | None = None) -> tuple[VectorSSA, MaskSSA]
```

参数说明：

- `lhs`（`VectorSSA`）：左操作数，被减数。必填。
- `rhs`（`VectorSSA`）：右操作数，减数。必填。
- `carry_in`（`MaskSSA`）：逐 lane 的输入进位谓词。必填。
- `mask`（`MaskSSA | None`）：执行掩码。默认 `None`，表示所有 lane 有效。

约束说明：

- 在 Ascend 950 上的 `tla.vec.func(mode="simd")` 内调用。
- 两个操作数须为同类型的 64-lane `i32` 或 `u32` 向量。显式传入的 `mask` 须有 64 个 lane。
- `carry_in` 和 `carry_out` 均为 64-lane `MaskSSA` 谓词（`!tla.mask<64>`），每个 lane 占 1 bit。
- `carry_in` 为 `1` 表示无输入借位。为 `0` 表示有输入借位，该 lane 多减一。
- `carry_out` 为 `1` 表示无输出借位。为 `0` 表示有输出借位。
- 对 `mask` 未启用的 lane，`carry_out` 未定义。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # ALL 将所有输入位设为 1，表示无输入借位。
    carry_in = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt32)
    diff, carry_out = tla.subc(lhs, rhs, carry_in)
```

---

## `tla.mull`

**源码：** [`catlass.core_api.mull`](../../../../../catlass/core_api.py)

功能说明：

返回每个有效 lane 的 64 位乘积的低、高 32 位 `(low, high)`。`i32` 按有符号整数相乘，`u32` 按无符号整数相乘。

函数原型：

```python
tla.mull(lhs: VectorSSA, rhs: VectorSSA, *, mask: MaskSSA | None = None) -> tuple[VectorSSA, VectorSSA]
```

参数说明：

- `lhs`（`VectorSSA`）：左操作数。必填。
- `rhs`（`VectorSSA`）：右操作数。必填。
- `mask`（`MaskSSA | None`）：执行掩码。默认 `None`，表示所有 lane 有效。

约束说明：

- 在 Ascend 950 上的 `tla.vec.func(mode="simd")` 内调用。
- 两个操作数须为同类型的 64-lane `i32` 或 `u32` 向量。显式传入的 `mask` 须有 64 个 lane。

调用示例：

```python
with tla.vec.func(mode="simd"):
    low, high = tla.mull(lhs, rhs)
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

## `tla.relu`

**源码：** [`catlass.core_api.relu`](../../../../../catlass/core_api.py)

功能说明：

逐元素计算 ReLU，返回与 `src` 相同类型的向量。FP16、FP32 和 Int32 使用原生 `VRELU` 指令。Int64 使用有符号 64 位最大值计算，与零取最大值。

函数原型：

```python
tla.relu(src: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `src`（`VectorSSA`）为 FP16、FP32、Int32 或 Int64 输入寄存器，必需。
- `mask`（`MaskSSA | None`）为对应输入寄存器的执行掩码。默认值 `None` 启用寄存器中的所有元素位置。掩码未启用的位置输出零。

约束：

- 在 `@tla.kernel` 的 `tla.vec.func(mode="simd")` 作用域中调用。
- FP16 对应 128 个掩码位置，FP32 和 Int32 对应 64 个，Int64 对应 32 个。Int64 的输入向量需占满一个 32 元素寄存器。输入只含部分有效元素时，通过掩码限制参与计算的位置。

示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.relu(x, mask=mask)
```

---

## `tla.prelu`

**源码：** [`catlass.core_api.prelu`](../../../../../catlass/core_api.py)

功能说明：

使用原生 `VPRELU` 指令逐元素计算 PReLU。正值和 `+0.0` 原样返回。负值和 `-0.0` 与对应元素的斜率相乘。斜率为 `+inf` 时，`-0.0` 的结果为 `NaN`。

函数原型：

```python
tla.prelu(src: VectorSSA, slope: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `src`（`VectorSSA`）为 FP16 或 FP32 输入寄存器，必需。
- `slope`（`VectorSSA`）为逐元素斜率寄存器，元素类型须与 `src` 一致，必需。
- `mask`（`MaskSSA | None`）为对应输入寄存器的执行掩码。默认值 `None` 启用寄存器中的所有元素位置。掩码未启用的位置输出零。

约束：

- 在 `@tla.kernel` 的 `tla.vec.func(mode="simd")` 作用域中调用。
- 返回值与 `src` 类型相同。FP16 对应 128 个掩码位置，FP32 对应 64 个掩码位置。输入寄存器只含部分有效元素时，传入掩码，仅启用两个输入都有效的位置。

示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.prelu(x, slopes, mask=mask)
```

---

## `tla.leaky_relu`

**源码：** [`catlass.core_api.leaky_relu`](../../../../../catlass/core_api.py)

功能说明：

使用原生 `VLRELU` 指令逐元素计算 Leaky ReLU。正值和 `+0.0` 原样返回。负值和 `-0.0` 与共享标量斜率相乘。斜率为 `+inf` 时，`-0.0` 的结果为 `NaN`。

函数原型：

```python
tla.leaky_relu(src: VectorSSA, slope: Float16 | Float32 | float, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `src`（`VectorSSA`）为 FP16 或 FP32 输入寄存器，必需。
- `slope`（`Float16 | Float32 | float`）为标量斜率，必需。带类型的斜率须与 `src` 的元素类型一致。Python `float` 字面量按 `src` 的元素类型生成。
- `mask`（`MaskSSA | None`）为对应输入寄存器的执行掩码。默认值 `None` 启用寄存器中的所有元素位置。掩码未启用的位置输出零。

约束：

- 在 `@tla.kernel` 的 `tla.vec.func(mode="simd")` 作用域中调用。
- 返回值与 `src` 类型相同。FP16 对应 128 个掩码位置，FP32 对应 64 个掩码位置。输入寄存器只含部分有效元素时，传入掩码，仅启用有效位置。

示例：

```python
with tla.vec.func(mode="simd"):
    y = tla.leaky_relu(x, 0.25, mask=mask)
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
