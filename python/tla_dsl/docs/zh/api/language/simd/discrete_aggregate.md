---
nav_order: 56
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/discrete_aggregate.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 离散与聚合

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

按索引 gather，以及带 mask 的归约（`VectorSSA.reduce`）。

## `tla.VectorSSA.reduce`

**源码：** [`catlass.core_api.VectorSSA.reduce`](../../../../../catlass/core_api.py#L576)

功能说明：

在 mask 下对本寄存器驻留 vector 做归约，得到单 lane 片段
（`tla.ReductionOp.ADD` / `MAX` / `MIN`）。可用
`tla.full(fragment, dtype=...)` 再广播回全宽。

函数原型：

```python
tla.VectorSSA.reduce(kind: ReductionOp, *, mask: Any, init_value: Any | None = None, reduction_profile: Any | None = None) -> Any
```

参数说明：

- `kind`（`ReductionOp`）：归约种类（`ADD`、`MAX` 或 `MIN`）。必填。
- `mask`（`MaskSSA`）：谓词哪些 lane 参与。必填。
- `init_value`：预留；当前仅支持 `None`。可选，默认 `None`。
- `reduction_profile`：预留；当前仅支持 `None`。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- 元素类型须为 f16/f32/i16/i32/u16/u32 之一。
- `init_value` 与 `reduction_profile` 须省略或为 `None`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    total = src_reg.reduce(tla.ReductionOp.ADD, mask=m)
    broadcast = tla.full(total, dtype=tla.Float32)
```

---

## `tla.gather`

**源码：** [`catlass.core_api.gather`](../../../../../catlass/core_api.py#L9201)

功能说明：

按索引从 UB tile 或寄存器 gather 元素。

两种形态：

- **UB→Reg**：`x` 为 UB 上的 `Tensor`；`y` 为 `i32` 索引；可选 `mask`。
- **Reg→Reg**：`x` 为 `VectorSSA`；`y` 为与 `x` 元素位宽相同的整数索引（`i8`/`u8`、`i16`/`u16` 或 `i32`/`u32`）；`result[i] = x[y[i] % VL]`。无 `mask`。

函数原型：

```python
tla.gather(x: Tensor | VectorSSA, y: VectorSSA, *, mask: MaskSSA | None = None) -> VectorSSA
```

参数说明：

- `x`（`Tensor | VectorSSA`）：源 tile（UB）或源寄存器。必填。
- `y`（`VectorSSA`）：索引 vector 寄存器。必填。
- `mask`（`MaskSSA | None`）：仅 UB→Reg 可选执行掩码；`None` 表示全有效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。
- Tensor 源须位于 UB。

调用示例：

```python
with tla.vec.func(mode="simd"):
    vals = tla.gather(ub_tile, idx_i32)
    w_brc = tla.gather(w, idx_u16)  # Reg→Reg
```
