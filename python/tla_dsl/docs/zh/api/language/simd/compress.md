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
具体使用示例参考 `Tensor.store`。

调用示例：

```python
# src_ub 是包含 128 个元素的连续 Int16 UB 窗口。
with tla.vec.func(mode="simd"):
    keep, _ = tla.update_mask(3, dtype=tla.Int16)
    packed = tla.squeeze(src_ub.load(), keep)
```

---

## `tla.unsqueeze`

**源码：** [`catlass.core_api.unsqueeze`](../../../../../catlass/core_api.py)

功能说明：

按 mask 展开每个 lane 的压缩前缀索引。

结果的 lane `i` 保存 mask 中 lane `< i` 的有效位个数。该操作是 `tla.squeeze` 的逆映射：将索引传给 `tla.gather`，可将压缩后的寄存器按索引还原到 `mask` 选中的 lane。

函数原型：

```python
tla.unsqueeze(mask: MaskSSA, dtype: type[Numeric]) -> VectorSSA
```

参数说明：

- `mask`（`MaskSSA`）：待展开的掩码。必填。
- `dtype`（`type[Numeric]`）：结果元素类型。必填。元素位宽决定 mask lane 数：`i8`/`u8` 对应 `mask<256>`，`i16`/`u16` 对应 `mask<128>`，`i32`/`u32` 对应 `mask<64>`。结果始终覆盖整个向量寄存器（8/16/32 位元素分别为 256/128/64 个有效 lane），不随 mask 的有效 lane 数变化。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func(mode="simd")` 内调用。
- 支持的结果元素类型：`i8`、`u8`、`i16`、`u16`、`i32`、`u32`。
- `dtype` 位宽须与 mask lane 数匹配。
- 部分有效的 mask（如 64 lane 寄存器中的 32 lane 尾块）仍会得到满寄存器的索引；请配合带 mask 的 `store`，只写回还原的 lane。
- 计数按 `dtype` 的位宽写入。`i8` 结果在计数超过 127 时回绕（如 128 对应 -128）；需要完整非负计数时使用 `u8`、`u16` 或 `u32`，回绕后的位模式仍可作为合法的 gather 索引。

调用示例：

```python
# 通过 squeeze -> unsqueeze -> gather 还原 32 lane 尾块。
with tla.vec.func(mode="simd"):
    keep, _ = tla.update_mask(32, dtype=tla.Int32)
    packed = tla.squeeze(src_tile.load(), keep)
    idx = tla.unsqueeze(keep, tla.UInt32)
    dst_tile.store(tla.gather(packed, idx), mask=keep)
```
