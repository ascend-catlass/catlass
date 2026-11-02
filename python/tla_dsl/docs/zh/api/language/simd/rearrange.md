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

**源码：** [`catlass.core_api.pack`](../../../../../catlass/core_api.py#L7890)

功能说明：

将 predicate 值打包到更宽目标 predicate 寄存器的低半区或高半区。

函数原型：

```python
tla.pack(src: MaskSSA, *, part: RegPart) -> MaskSSA
```

参数说明：

- `src`（`MaskSSA`）：待打包的 predicate 寄存器值。必填。
- `part`（`tla.RegPart`）：目标寄存器半区，支持 `LOW`（低半区）和 `HIGH`（高半区）。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func(mode="simd")` 内调用。
- Mask 输入支持 `mask<64> -> mask<128>` 和 `mask<128> -> mask<256>`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    packed = tla.pack(src, part=tla.RegPart.LOW)
```

---

## `tla.interleave`

**源码：** [`catlass.core_api.interleave`](../../../../../catlass/core_api.py#L7889)

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

**源码：** [`catlass.core_api.deinterleave`](../../../../../catlass/core_api.py#L7967)

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
