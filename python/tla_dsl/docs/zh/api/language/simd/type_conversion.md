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

将 `VectorSSA` 中的元素转换为 `dst_type`，返回以 `dst_type` 为元素类型的 `VectorSSA`。

| 源元素类型 | 目标元素类型 |
| --- | --- |
| `Float16` | `Float16`、`BFloat16`、`Float32`、`Float8E4M3FN`、`Float8E5M2`、`HiFloat8`、`Int4x2`、`Int8`、`Int16`、`Int32`、`UInt8` |
| `BFloat16` | `Float16`、`BFloat16`、`Float32`、`Float8E4M3FN`、`Float8E5M2`、`Float8E8M0`、`Float4E2M1x2`、`Float4E1M2x2`、`Int32` |
| `Float32` | `Float16`、`BFloat16`、`Float32`、`Float8E4M3FN`、`Float8E5M2`、`HiFloat8`、`Int16`、`Int32`、`Int64` |
| `Float8E4M3FN`、`Float8E5M2` | `Float16`、`BFloat16`、`Float32` |
| `HiFloat8` | `Float16`、`Float32` |
| `Float8E8M0` | `BFloat16` |
| `Float4E2M1x2`、`Float4E1M2x2` | `BFloat16` |
| `Int4x2` | `Int16`、`Float16`、`BFloat16` |
| `Int8` | `Int16`、`Int32`、`Float16` |
| `Int16` | `Int4x2`、`Int32`、`UInt8`、`UInt32`、`Float16`、`Float32` |
| `Int32` | `Int8`、`Int16`、`Int64`、`UInt8`、`UInt16`、`Float32` |
| `Int64` | `Int32`、`Float32` |
| `UInt8` | `UInt16`、`UInt32`、`Float16`、`Float32` |
| `UInt16` | `UInt8`、`UInt32` |
| `UInt32` | `UInt8`、`UInt16`、`Int16` |

`Float16` 或 `Float32` 转为 `HiFloat8` 时，须指定 `CAST_ROUND`（最近舍入，中点远离零）
或 `CAST_HYBRID`（混合舍入，结果可能偏离最近的可表示值）。

`BFloat16` 转为 `Float8E8M0` 时提取指数位。正负零和次正规数编码为 `0x00`。
正负无穷大固定饱和到最大有限缩放值 `0xfe`，NaN 编码为 `0xff`。反向转换时，`0x00` 解码为 `2**-127`
（BF16 位型为 `0x0040`），`0xff` 解码为静默 NaN。

打包四位类型每字节存两个元素，第一个元素在低半字节。打包将第 `j` 对写入字节
`4*j + reg_slot`，解包从同一位置读取。扩宽 `Int4x2` 时使用 `CAST_NONE`。
`Float32` 经 `BFloat16` 转为打包 FP4 时，可能发生两次舍入。

函数原型：

```python
tla.VectorSSA.to(dst_type: Any, params: CastParams, mask: Any | None = None) -> Any
```

参数说明：

- `dst_type`（`type[Numeric]`）为表中列出的目标元素类型。
- `params`（`CastParams`）为必填参数，用于设置舍入模式 `round_mode`、饱和模式 `sat_mode` 和寄存器槽位 `reg_slot`。
- `mask`（`MaskSSA | None`）按源元素下标选择参与转换的元素。默认值 `None` 表示掩码全开。
  未选中的目标元素置零。打包四位值时，使用 128 位源掩码的偶数位，或使用
  `dtype=tla.UInt32` 创建 64 位数对掩码，每位选择一对元素。

约束说明：

- 在 `@tla.kernel` 函数的 `tla.vec.func(mode="simd")` 中调用。SIMT 标量转换使用 `Numeric.to`。
- `reg_slot` 默认为 `RegSlot.ZERO`。位宽相差两倍且使用槽位选择时，`ZERO` 和 `ONE`
  分别选择较窄类型一侧的偶数和奇数位置。打包四位转换、位宽相差四倍的整数转换、
  `UInt8` 转 `Float32`，以及 `Float32` 与 `Float8E4M3FN`、`Float8E5M2` 或
  `HiFloat8` 的互转，均可使用 `ZERO` 至 `THREE` 选择四个位置之一。
- 对需要舍入且支持 `CAST_RINT` 的转换，默认的 `CAST_NONE` 按中点取偶舍入。
  `BFloat16` 转 `Float16`、`Float16` 转 `Int32` 或 `Int4x2`、`Float32` 与 `Int64` 互转、
  `BFloat16` 转打包 FP4 时均不可使用 `CAST_ODD`。
- `Float16` 转 `BFloat16` 时，`sat_mode` 须设为 `SatMode.UNKNOWN`。
  `BFloat16` 转打包 FP4 时固定饱和，须设为 `SatMode.UNKNOWN` 或 `SatMode.SAT`。
  `Int16` 转 `UInt32` 时固定饱和。
- `Float16`、`BFloat16`、`Float32` 转为自身类型时，对选中的元素舍入，保留原位置。
  槽位用 `ZERO` 或 `ONE`，饱和保持关闭。

调用示例：

```python
with tla.vec.func(mode="simd"):
    as_i32 = x_f32.to(tla.Int32, tla.params.CastParams())
    to_hif8 = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_ROUND)
    as_hif8 = x_f32.to(tla.HiFloat8, to_hif8)
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
