---
nav_order: 510
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/histogram.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 直方图计算

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

寄存器向量上的直方图累加。

## `tla.histograms`

**源码：** [`catlass.core_api.histograms`](../../../../../catlass/core_api.py)

功能说明：

寄存器驻留的 FREQUENCY 或 ACCUMULATE 直方图。

函数原型：

```python
tla.histograms(src: VectorSSA, dst: VectorSSA, params: HistogramsParams, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `src`（`VectorSSA`）：`i8` / `ui8` 源寄存器（256 lane）。必填。
- `dst`（`VectorSSA`）：`i16` / `ui16` 累加器（128 lane）。必填。在已有
  `dst` 值上累加；全新直方图须先传入 `tla.full(0, dtype=...)`。
- `params`（`HistogramsParams`）：`mode`（BIN0/BIN1）与 `type`
  （FREQUENCY/ACCUMULATE）。必填。
- `mask`（`MaskSSA`）：可选 lane mask；默认全真。

约束说明：

- 须在 `@tla.kernel` 与 `tla.vec.func()` 内调用。
- FREQUENCY：半区间逐值计数。ACCUMULATE：对 u8 做 CDF 再切分；ACCUMULATE 下
  `BIN1` 并非仅表示 `[128,255]`。

调用示例：

```python
hist_params = tla.params.HistogramsParams(
    mode=tla.params.HistogramsBinType.BIN0,
    type=tla.params.HistogramsType.ACCUMULATE,
)
with tla.vec.func(mode="simd"):
    hist_lo = tla.full(0, dtype=tla.Int16)
    hist_lo = tla.histograms(keys_u8, hist_lo, hist_params, mask=m)
```
