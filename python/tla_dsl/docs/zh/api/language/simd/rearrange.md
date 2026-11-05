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

将 predicate 值打包到目标寄存器的低半区或高半区；对于整数 vector，截取每个源元素的低 8 或 16 位，写入所选目标半区。vector 另一半区的值未定义。目标元素类型由源类型推导：结果始终为无符号类型，位宽为源的一半。

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
- Vector 类型对（目标 <- 源）：`UInt8 <- Int16/UInt16`、`UInt16 <- Int32/UInt32`。不支持 64 位源（`Int64`/`UInt64`）。
- Vector 输入须填满一个 256Byte 寄存器（RegTraitNumOne），结果的 lane 数为源的两倍。`VectorSSA` 无法表示 RegTraitNumTwo；CANN 对该 trait 仅允许 LOW。
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

读取源寄存器所选半区的整数元素并展宽。有符号类型执行符号扩展，无符号类型执行零扩展。目标元素类型由源类型推导：结果保留源的符号性，位宽为源的两倍。

函数原型：

```python
tla.unpack(src: VectorSSA, *, part: RegPart) -> VectorSSA
```

参数说明：

- `src`（`VectorSSA`）：窄整数源寄存器。必填。
- `part`（`tla.RegPart`）：读取的源半区，支持 `LOW` 和 `HIGH`。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体及 `tla.vec.func(mode="simd")` 内调用。
- 类型对（目标 <- 源）：`Int16 <- Int8`、`UInt16 <- UInt8`、`Int32 <- Int16`、`UInt32 <- UInt16`。不支持 64 位目标（`Int64`/`UInt64`）。
- 源和目标的符号性须一致，结果的 lane 数为源的一半。
- 输入须填满一个 256Byte 寄存器（RegTraitNumOne）。`VectorSSA` 无法表示 RegTraitNumTwo；CANN 对该 trait 仅允许 LOW。
- 仅支持 Ascend 950（`--npu-arch 3510`）。`LOW/HIGH` 对应 CANN 的 `HighLowPart::LOWEST/HIGHEST`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    wide = tla.unpack(src_reg, part=tla.RegPart.HIGH)
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

调用示例：

```python
with tla.vec.func(mode="simd"):
    even, odd = tla.deinterleave(a, b)
    mask_even, mask_odd = tla.deinterleave(mask0, mask1)
```
