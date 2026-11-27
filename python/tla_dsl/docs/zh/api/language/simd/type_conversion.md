---
nav_order: 59
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/type_conversion.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 类型转换

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

寄存器 vector 上的元素类型转换（`VectorSSA.to`）与按位重解释（`VectorSSA.bitcast`）。

## `tla.VectorSSA.to`

**源码：** [`catlass.core_api.VectorSSA.to`](../../../../../catlass/core_api.py)

功能说明：

将 `VectorSSA` 中的元素转换为 `dst_type`，返回转换后的 `VectorSSA`。

### 支持的转换

| 源类型 | 目标类型 |
| --- | --- |
| `Float16` | `Float16`、`BFloat16`、`Float32`、`Float8E4M3FN`、`Float8E5M2`、`HiFloat8`、`Int4x2`、`Int8`、`Int16`、`Int32`、`UInt8` |
| `BFloat16` | `Float16`、`BFloat16`、`Float32`、`Float8E4M3FN`、`Float8E5M2`、`Float8E8M0`、`Float4E2M1x2`、`Float4E1M2x2`、`Int32` |
| `Float32` | `Float16`、`BFloat16`、`Float32`、`Float8E4M3FN`、`Float8E5M2`、`HiFloat8`、`Int16`、`Int32`、`Int64` |
| `Float8E4M3FN` | `Float16`、`BFloat16`、`Float32` |
| `Float8E5M2` | `Float16`、`BFloat16`、`Float32` |
| `HiFloat8` | `Float16`、`Float32` |
| `Float8E8M0` | `BFloat16` |
| `Float4E2M1x2` | `BFloat16` |
| `Float4E1M2x2` | `BFloat16` |
| `Int4x2` | `Int16`、`Float16`、`BFloat16` |
| `Int8` | `Int16`、`Int32`、`Float16` |
| `Int16` | `Int4x2`、`Int8`、`Int32`、`UInt8`、`UInt32`、`Float16`、`Float32` |
| `Int32` | `Int8`、`Int16`、`Int64`、`UInt8`、`UInt16`、`Float32` |
| `Int64` | `Int32`、`Float32` |
| `UInt8` | `UInt16`、`UInt32`、`Float16`、`Float32` |
| `UInt16` | `UInt8`、`UInt32` |
| `UInt32` | `UInt8`、`UInt16`、`Int16` |

### 舍入模式

`round_mode` 使用 `RoundMode` 中的值，默认为 `CAST_NONE`。下表的舍入规则适用于目标类型范围内的有限值。

| 浮点转换 | 舍入到什么值 | `CAST_RINT` 在中点怎么取偶 |
| --- | --- | --- |
| 同类型（`Float16`、`BFloat16` 或 `Float32`） | 舍入到整数值，结果仍用原浮点类型表示。 | 选偶数。以 `Float32` → `Float32` 为例，`2.5` 得到 `2.0`，`3.5` 得到 `4.0`。 |
| 不同精度（需要舍入时） | 舍入到目标浮点类型能精确表示的值。 | 选二进制尾数末位为 `0` 的值。以 `Float32` → `Float16` 为例，`256.375` 刚好在相邻的 `Float16` 值 `256.25` 和 `256.5` 中间。它们的二进制尾数分别以 `01` 和 `10` 结尾，所以得到 `256.5`。 |

| 模式 | 怎么舍入 |
| --- | --- |
| `CAST_NONE` | 使用该转换方向的默认舍入规则。 |
| `CAST_RINT` | 在候选值中选离输入最近的一个。输入刚好在两个候选值中间时取偶。转为整数类型时，`2.5` 得到 `2`，`3.5` 得到 `4`。 |
| `CAST_ROUND` | 在候选值中选离输入最近的一个。输入刚好在两个候选值中间时，选离零较远的那个。转为整数时，2.5 得到 3，-2.5 得到 -3。 |
| `CAST_FLOOR` | 向下舍入。转为整数时，1.25 得到 1，-1.25 得到 -2。 |
| `CAST_CEIL` | 向上舍入。转为整数时，1.25 得到 2，-1.25 得到 -1。 |
| `CAST_TRUNC` | 向零舍入。转为整数时，1.75 得到 1，-1.75 得到 -1。 |
| `CAST_ODD` | 目标类型能精确表示输入时，保留原值。需要丢弃精度时，先向零截断，再把结果保留的最低位设为 1。 |
| `CAST_HYBRID` | 转为 `HiFloat8` 时，部分输入范围使用 `CAST_ROUND`。在其他范围，输入的二进制位决定是否向远离零的方向进位。结果可能偏离最近的可表示值。 |

### `CAST_NONE` 的默认行为

| 转换方向 | 省略 `round_mode` 或设为 `CAST_NONE` 时 |
| --- | --- |
| 支持 `CAST_RINT` 的转换 | 按 `CAST_RINT` 舍入，同类型浮点转换也会取整。 |
| `Float16`、`Float32` → `HiFloat8` | 使用 `CAST_ROUND`。 |
| `Float8E4M3FN` → `Float16`、`BFloat16`<br>`Float8E5M2` → `BFloat16`<br>`UInt8` → `Float32` | API 参数用 `CAST_NONE`。最后一条指令用 `CAST_RINT`，可精确转换输入的有限值。 |
| 其他仅允许 `CAST_NONE` 的方向 | `CAST_NONE` 是占位值，转换按固定规则执行。 |

### 饱和模式

`sat_mode` 使用 `SatMode` 中的值，默认为 `UNKNOWN`。

| 模式 | 含义 |
| --- | --- |
| `SatMode.UNKNOWN` | 转换方向没有可选的饱和模式时，用这个值占位。API 不会将 `UNKNOWN` 自动转换为 `SAT` 或 `NOSAT`。要求显式指定饱和模式的方向传入 `UNKNOWN` 时，编译会报错。 |
| `SatMode.SAT` | 结果超出目标类型的有限值范围时，取离结果最近的边界值。 |
| `SatMode.NOSAT` | 结果超出目标类型范围时，按该转换方向的非饱和规则处理。 |

要让 `CastParams` 中指定的 `SAT` 或 `NOSAT` 生效，必须将 `CTRLValue.saturation_strategy` 设为
`OverrideStrategy.USE_API`。这是向量核的全局设置。在 `tla.vector()` 内、首次进入使用该设置的
`tla.vec.func(mode="simd")` 前调用一次 `tla.arch.set_ctrl()` 即可。同一内核在该向量核上的
后续转换会沿用此设置，直到 `saturation_strategy` 被修改。

```python
with tla.vector():
    tla.arch.set_ctrl(
        tla.params.CTRLValue(
            saturation_strategy=tla.params.OverrideStrategy.USE_API
        )
    )
```

使用 `OverrideStrategy.USE_GLOBAL` 时，`CTRLValue` 中的 `conversion_mode` 和饱和设置也必须符合下表对该转换方向的限制。
`CastParams` 仍须满足表中的要求。全局设置不合适时，即使转换能通过编译，也可能得到错误结果。

### 转换模式总表

每条箭头左侧的任一源类型都可以转为右侧列出的任一目标类型。标有 (复合) 的方向需要多条指令完成转换。

| 转换方向 | 允许的 `round_mode` | 允许的 `sat_mode` |
| --- | --- | --- |
| `Float16`、`BFloat16` → `Float8E4M3FN`、`Float8E5M2` (复合)<br>`Float32` → `Float8E4M3FN`、`Float8E5M2` | `CAST_NONE`、`CAST_RINT` | `SAT`、`NOSAT` |
| `Float16`、`Float32` → `HiFloat8` | `CAST_NONE`、`CAST_ROUND`、`CAST_HYBRID` | `SAT`、`NOSAT` |
| `Float16` → `Int4x2`、`Int8`、`Int16`、`UInt8`<br>`BFloat16` → `Float16`、`Int32`<br>`Float32` → `BFloat16`、`Int16`、`Int32`、`Int64` | `CAST_NONE`、`CAST_RINT`、`CAST_ROUND`、`CAST_FLOOR`、`CAST_CEIL`、`CAST_TRUNC` | `SAT`、`NOSAT` |
| `Float32` → `Float16` | `CAST_NONE`、`CAST_RINT`、`CAST_ROUND`、`CAST_FLOOR`、`CAST_CEIL`、`CAST_TRUNC`、`CAST_ODD` | `SAT`、`NOSAT` |
| `Int16` → `Int4x2`、`UInt8`<br>`Int32` → `Int16`、`UInt8`、`UInt16`<br>`Int64` → `Int32`<br>`UInt16` → `UInt8`<br>`UInt32` → `Int16`、`UInt8`、`UInt16` | `CAST_NONE` | `SAT`、`NOSAT` |
| `Int16`、`Int32` → `Int8` | `CAST_NONE` | `NOSAT` |
| `Float16` → `Float16`、`BFloat16`、`Int32`<br>`BFloat16` → `BFloat16`、`Float4E2M1x2`、`Float4E1M2x2`<br>`Float32`、`Int32`、`Int64` → `Float32`<br>`Int16` → `Float16` | `CAST_NONE`、`CAST_RINT`、`CAST_ROUND`、`CAST_FLOOR`、`CAST_CEIL`、`CAST_TRUNC` | `UNKNOWN` |
| `Float16`、`BFloat16`、`Int16` → `Float32`<br>`Float8E4M3FN`、`Float8E5M2` → `Float16`、`BFloat16` (复合)<br>`Float8E4M3FN`、`Float8E5M2` → `Float32`<br>`HiFloat8` → `Float16`、`Float32`<br>`BFloat16` → `Float8E8M0` (复合)<br>`Float8E8M0` → `BFloat16` (复合)<br>`Float4E2M1x2`、`Float4E1M2x2` → `BFloat16`<br>`Int4x2` → `Int16`、`Float16`、`BFloat16`<br>`Int8` → `Int16`、`Int32`、`Float16`<br>`Int16` → `Int32`、`UInt32`<br>`Int32` → `Int64`<br>`UInt8` → `UInt16`、`UInt32`、`Float16`<br>`UInt8` → `Float32` (复合)<br>`UInt16` → `UInt32` | `CAST_NONE` | `UNKNOWN` |

`BFloat16` 转为 `Float8E8M0` 时，正负无穷大均编码为 `0xfe`，表示最大有限缩放值。NaN 编码为 `0xff`。

### 寄存器槽位

`reg_slot` 默认为 `RegSlot.ZERO`。

| 转换方向 | 允许的 `reg_slot` | 位置 |
| --- | --- | --- |
| 打包四位转换<br>位宽相差四倍的整数转换<br>`UInt8` → `Float32`<br>`Float32` ↔ `Float8E4M3FN`、`Float8E5M2`、`HiFloat8` | `ZERO`、`ONE`、`TWO`、`THREE` | 较窄类型一侧的位置 0 到 3。 |
| 其他受支持的转换 | `ZERO`、`ONE` | 转换使用两个槽位时，`ZERO` 选择较窄类型一侧的偶数位置，`ONE` 选择奇数位置。 |

函数原型：

```python
tla.VectorSSA.to(dst_type: Any, params: CastParams, mask: Any | None = None) -> Any
```

参数说明：

- `dst_type`（`type[Numeric]`）为支持列表中的目标类型。
- `params`（`CastParams`）为必填参数，用来设置 `round_mode`、`sat_mode` 和 `reg_slot`。
- `mask`（`MaskSSA | None`）按源位置选择参与转换的元素。默认值 `None` 启用掩码的所有位置。未选中的目标位置填零。
  打包四位值时，可传入对应 128 个源元素的掩码，由偶数位置选择元素对。也可以用 `dtype=tla.UInt32` 创建掩码，64 个位置各选择一对元素。

约束说明：

- 从支持的转换中选择源类型和目标类型，再按总表设置该方向允许的模式。无效组合会在编译时报错。
- 在 `@tla.kernel` 函数的 `tla.vec.func(mode="simd")` 中调用。SIMT 标量转换使用 `Numeric.to`。

示例在 `@tla.kernel` 中运行。`i16_ub` 包含 128 个 `Int16` 元素，`fp32_ub` 包含 64 个 `Float32` 元素。两个张量都在 UB 中。

调用示例：

```python
with tla.vector():
    tla.arch.set_ctrl(
        tla.params.CTRLValue(
            saturation_strategy=tla.params.OverrideStrategy.USE_API
        )
    )
    with tla.vec.func(mode="simd"):
        x_i16 = i16_ub.load()
        narrow = tla.params.CastParams(sat_mode=tla.params.SatMode.NOSAT)
        as_i8 = x_i16.to(tla.Int8, narrow)
        as_f32 = x_i16.to(tla.Float32, tla.params.CastParams())
        as_hif8 = fp32_ub.load().to(tla.HiFloat8, narrow)
```

---

## `tla.VectorSSA.bitcast`

**源码：** [`catlass.core_api.VectorSSA.bitcast`](../../../../../catlass/core_api.py)

功能说明：

将寄存器内 vector 的比特按另一种元素类型重新解释，不做数值转换。物理寄存器
保持全部 2048 bit，源和目标元素位宽可以不同；物理 lane 数和有效 lane 数分别
调整，保留各自的位数。例如，64 个 Float32 lane 可解释为 128 个 BFloat16 lane，
暴露每个 Float32 比特模式中的两个半字。

支持 `f32`/`i32`/`u32`、`f16`/`bf16`/`i16`/`u16`、
`i8`/`u8`/`f8e4m3fn`/`f8e5m2`/`hif8`/`f4e2m1x2`/`f4e1m2x2` 和 `i64`/`u64`。
`tla.Float8E8M0` 与 `tla.UInt8` 可以双向按位重解释，用于处理 E8M0 编码字节。
不支持 `tla.Bool` 或逻辑四位 Tensor 类型。该操作不改变字节顺序
或 register slot，也不压紧稀疏值。数值转换使用 `.to()`，布局变化使用显式重排。

`i64`/`u64` 寄存器可通过 UB load、`tla.extern` 或 bitcast 获得。
寄存器搬运支持对齐、满宽的 `DIST_NORM` load，以及对齐且不带显式 mask 的
`DIST_NORM` store。Tensor copy 支持范围取决于搬运通路，见 `tla.copy` 路由表。
bitcast 不会放宽这些搬运约束，也不会启用 64 位算术运算。

函数原型：

```python
tla.VectorSSA.bitcast(dst_type: Any) -> Any
```

参数说明：

- `dst_type`（`type[Numeric]`）：重新解释为的元素类型。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- 静态有效位数须能组成完整的目标元素。
- 动态有效范围保持动态，消费者须使用显式 mask。
- 重解释为源自身的元素类型是空操作，直接返回 `self`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    bits = x.bitcast(tla.Int32)
    exponent = tla.bitwise_and(
        tla.shift_right(bits, 23, mask=m), tla.full(0xFF, tla.Int32)
    )
```
