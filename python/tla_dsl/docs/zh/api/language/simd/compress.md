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
tla.squeeze(src: VectorSSA, mask: MaskSSA) -> VectorSSA
```

参数说明：

- `src`（`VectorSSA`）：待压缩的源 vector。必填。
- `mask`（`MaskSSA`）：保留元素的掩码。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    packed = tla.squeeze(src, m)
```
