---
nav_order: 57
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/rearrange.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 数据重排

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

提供 vector 寄存器或 predicate mask 的 lane 交插与解交插操作。

## `tla.pack`

**源码：** [`catlass.core_api.pack`](../../../../../catlass/core_api.py)

功能说明：

将谓词值打包到所选目标寄存器半区。对整数向量，截取每个源元素的低 8、16 或 32 位并写入所选半区，另一半区的值未定义。结果元素为无符号整数，位宽为源元素的一半。

函数原型：

```python
tla.pack(src: VectorSSA | MaskSSA, *, part: RegPart) -> VectorSSA | MaskSSA
```

参数说明：

- `src`（`VectorSSA | MaskSSA`）：源寄存器值。必填。
- `part`（`tla.RegPart`）：目标寄存器半区，支持 `LOW`（低半区）和 `HIGH`（高半区）。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func(mode="simd")` 内调用。
- Mask 输入支持 `mask<64> -> mask<128>` 和 `mask<128> -> mask<256>`。
- 向量类型对（目标 <- 源）包括 `UInt8 <- Int16/UInt16`、`UInt16 <- Int32/UInt32` 和 `UInt32 <- Int64/UInt64`。
- 向量输入须填满一个 256 字节寄存器（`RegTraitNumOne`），结果的 lane 数为源的两倍。对于 64 位源，AscendC `Pack` 组合使用零填充和 `DeInterleave`。
- Vector 形式仅支持 Ascend 950（`--npu-arch 3510`）。`LOW/HIGH` 对应 CANN 的 `HighLowPart::LOWEST/HIGHEST`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    packed = tla.pack(src, part=tla.RegPart.LOW)
    narrow = tla.pack(src_reg, part=tla.RegPart.HIGH)
```

---

## `tla.unpack`

**源码：** [`catlass.core_api.unpack`](../../../../../catlass/core_api.py)

功能说明：

读取源寄存器所选半区。对谓词输入，将所选半区扩展为逻辑 lane 数减半的谓词。对整数向量，有符号类型执行符号扩展，无符号类型执行零扩展。结果保留源元素的符号性，位宽为源元素的两倍。

函数原型：

```python
tla.unpack(src: VectorSSA | MaskSSA, *, part: RegPart) -> VectorSSA | MaskSSA
```

参数说明：

- `src`（`VectorSSA | MaskSSA`）：源寄存器值。必填。
- `part`（`tla.RegPart`）：读取的源半区，支持 `LOW` 和 `HIGH`。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体及 `tla.vec.func(mode="simd")` 内调用。
- 谓词输入支持 `mask<256> -> mask<128>` 和 `mask<128> -> mask<64>`。
- 向量类型对（目标 <- 源）包括 `Int16 <- Int8`、`UInt16 <- UInt8`、`Int32 <- Int16`、`UInt32 <- UInt16`、`Int64 <- Int32` 和 `UInt64 <- UInt32`。
- 向量输入须填满一个 256 字节寄存器（`RegTraitNumOne`）。结果的 lane 数为源的一半，且保留源元素的符号性。对于 64 位结果，AscendC `UnPack` 组合使用符号位或零填充及 `Interleave`。
- 仅支持 Ascend 950（`--npu-arch 3510`）。`LOW/HIGH` 对应 CANN 的 `HighLowPart::LOWEST/HIGHEST`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    mask_wide = tla.unpack(src_mask, part=tla.RegPart.LOW)
    wide = tla.unpack(src_reg, part=tla.RegPart.HIGH)
    wide64 = tla.unpack(src32_reg, part=tla.RegPart.LOW)  # Int32 -> Int64，符号扩展
```

---

## `tla.interleave`

**源码：** [`catlass.core_api.interleave`](../../../../../catlass/core_api.py)

功能说明：

按 lane 交替排列两个 vector 寄存器或两个 predicate mask，并返回两个寄存器。

函数原型：

```python
tla.interleave(src0: VectorSSA | MaskSSA, src1: VectorSSA | MaskSSA) -> tuple[VectorSSA, VectorSSA] | tuple[MaskSSA, MaskSSA]
```

参数说明：

- `src0`（`VectorSSA` 或 `MaskSSA`）：偶数 lane 输入。必填。
- `src1`（`VectorSSA` 或 `MaskSSA`）：与 `src0` 类别和类型相同的奇数 lane 输入。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；输入类型须匹配。MaskSSA 仅支持 `mask<64>`、`mask<128>` 和 `mask<256>`。
- VectorSSA 支持 i8/i16/i32、u8/u16/u32 和 f16/bf16/f32。
- 谓词重排使用 MaskSSA。暂不支持 b64 的 RegTraitNumTwo 形式。

调用示例：

```python
with tla.vec.func(mode="simd"):
    lo, hi = tla.interleave(a, b)
    mask_lo, mask_hi = tla.interleave(mask0, mask1)
```

---

## `tla.deinterleave`

**源码：** [`catlass.core_api.deinterleave`](../../../../../catlass/core_api.py)

功能说明：

按顺序拼接两个 vector 寄存器或两个 predicate mask，再分别返回偶数位置和奇数位置的 lane。

函数原型：

```python
tla.deinterleave(src0: VectorSSA | MaskSSA, src1: VectorSSA | MaskSSA) -> tuple[VectorSSA, VectorSSA] | tuple[MaskSSA, MaskSSA]
```

参数说明：

- `src0`（`VectorSSA` 或 `MaskSSA`）：拼接输入的前半部分。必填。
- `src1`（`VectorSSA` 或 `MaskSSA`）：与 `src0` 类别和类型相同的后半部分。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；两路 vector 的元素类型与 lane 数须匹配。
- MaskSSA 的输入和输出类型必须完全一致，仅支持 `mask<64>`、`mask<128>` 和 `mask<256>`；不支持 `mask<32>`/b64。
- VectorSSA 支持 i8/i16/i32、u8/u16/u32 和 f16/bf16/f32。
- 谓词重排使用 MaskSSA。暂不支持 b64 的 RegTraitNumTwo 形式。

调用示例：

```python
with tla.vec.func(mode="simd"):
    even, odd = tla.deinterleave(a, b)
    mask_even, mask_odd = tla.deinterleave(mask0, mask1)
```
