---
nav_order: 58
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/compress.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 数据压缩

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

## `tla.squeeze`

**源码：** [`catlass.core_api.squeeze`](../../../../../catlass/core_api.py)

功能说明：

按 mask 将选中元素压缩到结果低位。

函数原型：

```python
tla.squeeze(src: VectorSSA, mask: MaskSSA, *, record_nbytes: bool = False) -> VectorSSA
```

参数说明：

- `src`（`VectorSSA`）：待压缩的源 vector。必填。
- `mask`（`MaskSSA`）：保留元素的掩码。必填。
- `record_nbytes`（`bool`）：设为 `True` 时，硬件计算 `mask` 选中数据的字节数，
  并将字节数记录到内部专有寄存器。`store` 的 `POST_MODE_UPDATE_NBYTES` 模式使用
  这个字节数。默认 `False`，仅接受关键字传参。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func(mode="simd")` 内调用。
- 支持的元素类型：`f32`、`f16`、`i8`、`i16`、`i32`、`u8`、`u16`、`u32`。

`record_nbytes=True` 时，记录的字节数等于保留元素个数乘以单个元素的字节数，这个记录在内部专有寄存器中。
用户需要保证在计数被覆盖前，以 `UnalignStoreParams(post_mode=POST_MODE_UPDATE_NBYTES)` 调用 `store` 将预期的数据写到目的区。
具体使用示例参考 [`Tensor.store`](../data_movement.md#tlatensorstore)。

调用示例：

```python
# src_ub 是包含 128 个元素的连续 Int16 UB 窗口。
with tla.vec.func(mode="simd"):
    keep, _ = tla.update_mask(3, dtype=tla.Int16)
    packed = tla.squeeze(src_ub.load(), keep)
```
