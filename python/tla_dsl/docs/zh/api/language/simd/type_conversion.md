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

寄存器 vector 上的元素类型转换（`VectorSSA.to`）与同位宽按位重解释（`VectorSSA.bitcast`）。

## `tla.VectorSSA.to`

**源码：** [`catlass.core_api.VectorSSA.to`](../../../../../catlass/core_api.py)

功能说明：

将寄存器内的 vector 转换为另一种元素类型。

FP32→HiF8 支持 `CAST_ROUND`（最近舍入，中点远离零）和 `CAST_HYBRID`、
`SAT`/`NOSAT` 和四个 register slot；结果 `j` 位于字节位置 `4*j + reg_slot`。
HiF8→FP32 读取相同 slot，无需舍入。
算术运算先转换为 Float32，查看编码使用 UInt8 bitcast。
`CAST_HYBRID` 仅支持 FP32→HiF8，采用 CANN 原生混合舍入，
结果不保证是最近的可表示值。默认使用 `CAST_ROUND`。

lowering 支持有符号整数 `tla.Int8` ~ `tla.Int64`、浮点 `tla.Float16` /
`tla.BFloat16` / `tla.Float32`，以及两种 OCP fp8 格式 `tla.Float8E4M3FN` /
`tla.Float8E5M2`。无符号整数、`tla.Bool`（i1）、`tla.Float64` 与 `tla.Float8E8M0`
会被拒绝。

`tla.Float16` 到 `tla.BFloat16` 使用一条直接转换指令，保留全部 lane，支持
通用舍入模式（不含 HiF8 专用的 `CAST_HYBRID`）。`sat_mode` 必须为 `UNKNOWN`；
由于指令没有饱和模式参数，`NOSAT` 和 `SAT` 均会被拒绝。对于这一等宽转换，`reg_slot` 的 `ZERO` 与 `ONE` 效果相同。

fp8 操作数可与 `tla.Float32`、`tla.Float16`、`tla.BFloat16` 互相转换。其中只有
f32 这一对是单条指令：f16 与 bf16 两对由 lowering 展开为经过 f32 的两次 cast，
扩宽一侧精确、收窄一侧只舍入一次；`f8e5m2 -> f16` 则展开为一次整数扩宽加一次移位，
因为这两种格式共享同一符号与指数字段——与经过 f32 的组合路径选出相同元素，但只需
两条指令而非五条。fp8 之间没有重编码路径——两者指数范围不同，属于重量化而非
cast——也没有 fp8 与整数之间的路径。

同类型浮点 cast（`tla.Float16` -> `tla.Float16`、`tla.BFloat16` -> `tla.BFloat16`、
`tla.Float32` -> `tla.Float32`）保留元素类型，逐 lane 重新舍入，降级为 AVE `vtrc`
指令：只有 `round_mode` 到达指令，`sat_mode` 与 `reg_slot` 不产生任何效果。整数与
fp8 格式没有同类型 cast。

`CastParams.round_mode` 支持 `CAST_ROUND` / `CAST_FLOOR` / `CAST_CEIL` /
`CAST_TRUNC` / `CAST_RINT` / `CAST_ODD` / `CAST_NONE`；`CAST_HYBRID` 仅支持
FP32→HiF8。`CAST_NONE` 须显式传入：无损转换（如 `f16→f32`）仍走无
`rnd` 的指令；有损转换按 `CAST_RINT`（ties to even）处理。

函数原型：

```python
tla.VectorSSA.to(dst_type: Any, params: CastParams, mask: Any | None = None) -> Any
```

参数说明：

- `dst_type`（`type[Numeric]`）：目标元素类型。必填。
- `params`（`CastParams`）：舍入模式、饱和模式与寄存器 slot。必填。
- `mask`（`MaskSSA | None`）：可选执行掩码；`None` 表示所有 lane 使能。选填，默认 `None`。
  mask 按**源**元素域索引：lane `i` 谓词源元素 `i`，与位宽变化无关；被 mask 关掉的
  元素对结果贡献为零。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- 同类型浮点 cast 的 `sat_mode` 不允许 `SAT`（同类型 cast 从不饱和，
  `UNKNOWN` 与 `NOSAT` 均可），`reg_slot` 必须为 `ZERO` 或 `ONE`：该转换没有
  饱和参数，也没有 pack-quarter 选择子。
- `reg_slot` 的 `TWO` 与 `THREE` 指向 pack 的四分之一，仅在位宽变化为 4 倍的整数
  cast 以及直接的 fp8/HiF8<->f32 指令上合法。
- 线程级标量 cast 在 `mode="simt"` 下使用 `Numeric.to`；本方法是 vector（AVE）
  形态，始终需要 `CastParams`。

调用示例：

```python
trait = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
with tla.vec.func(mode="simd"):
    wide = narrow.to(tla.Float32, trait)
    # f32 -> f32 配合 CAST_TRUNC：小数位向零舍入。
    rounded = as_f32.to(tla.Float32, tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.UNKNOWN,
        round_mode=tla.params.RoundMode.CAST_TRUNC,
    ))
    # CAST_NONE 须显式传入；有损路径行为等同 CAST_RINT。
    as_i32 = as_f32.to(tla.Int32, tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_NONE,
    ))
```

---

## `tla.VectorSSA.bitcast`

**源码：** [`catlass.core_api.VectorSSA.bitcast`](../../../../../catlass/core_api.py)

功能说明：

将寄存器内 vector 的比特按另一种**同位宽**元素类型重新解释。该操作不做任何数值
转换，也不产生硬件指令：寄存器的比特保持原样，只有元素类型改变，因此开销为零。

这使得浮点数的比特模式在 Python 前端变得可读——取出指数、通过移入第 23 位构造
`2**k`、清除符号位——而不必经由 UB 以另一种元素类型存取一轮。

同位宽的组合为 `f32`/`i32`/`u32`、`f16`/`bf16`/`i16`/`u16`、
`i8`/`u8`/`f8e4m3fn`/`f8e5m2`/`hif8` 以及 `i64`/`u64`，lane 数保持不变。位宽变化属于
`VectorSSA.to` 的职责，在此会被拒绝。`tla.Bool`（i1）、`tla.Float8E8M0` 与打包的
fp4 格式不是合法的 `!tla.vector` 元素类型，同样会被拒绝。

`i64`/`u64` 寄存器可由 extern 返回，并在两种类型间进行 bitcast。
现有 tensor API 不支持直接 load 这些寄存器类型。

函数原型：

```python
tla.VectorSSA.bitcast(dst_type: Any) -> Any
```

参数说明：

- `dst_type`（`type[Numeric]`）：重新解释为的同位宽元素类型。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- `dst_type` 的位宽须与源元素类型相同。
- 重解释为源自身的元素类型是空操作，直接返回 `self`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    bits = x.bitcast(tla.Int32)
    exponent = tla.bitwise_and(
        tla.shift_right(bits, 23, mask=m), tla.full(0xFF, tla.Int32)
    )
```
