---
nav_order: 55
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/fill.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 数据填充

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

## `tla.full`

**源码：** [`catlass.core_api.full`](../../../../../catlass/core_api.py)

功能说明：

用标量填充 vector SSA，或将单 lane 片段广播到全寄存器。

函数原型：

```python
tla.full(value: bool | int | float | Numeric | VectorSSA, dtype: type[Numeric], *, mask: Any | None = None) -> VectorSSA
```

参数说明：

- `value`（`bool | int | float | Numeric | VectorSSA`）：填充常量，或待广播的
  单 lane vector 片段。必填。
- `dtype`（`type[Numeric]`）：vector 元素类型。必填。
- `mask`（`MaskSSA | None`）：谓词哪些 lane 被写入。可选；默认写全部 lane。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`value` 须为 Python 标量字面量、Host
  `Numeric`，或单 lane vector 片段。
- 当 `value` 为 vector 时，必须恰好 1 个有效 lane，且元素类型与 `dtype` 一致。
- 若提供 `mask`，须与结果元素类型匹配。

调用示例：

```python
with tla.vec.func(mode="simd"):
    zeros = tla.full(0.0, dtype=tla.Float32)
    all_lanes, _ = tla.update_mask(64, tla.Float32)
    total = src.reduce(tla.ReductionOp.ADD, mask=all_lanes)
    spread = tla.full(total, dtype=tla.Float32)
```

---

## `tla.arange`

**源码：** [`catlass.core_api.arange`](../../../../../catlass/core_api.py)

功能说明：

创建单调递增或递减的一维 vector SSA（`base` + `order`）。

函数原型：

```python
tla.arange(base: bool | int | float | Numeric = 0, *, order: str = 'increase', dtype: type[Numeric]) -> VectorSSA
```

参数说明：

- `base`（`bool | int | float | Numeric`）：起始基数。可选，默认 `0`。
- `order`（`str`）：`'increase'` 递增或 `'decrease'` 递减。可选，默认 `'increase'`。
- `dtype`（`type[Numeric]`）：vector 元素类型。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`order` 仅支持 `increase` / `decrease`。

调用示例：

```python
with tla.vec.func(mode="simd"):
    lane_idx = tla.arange(base=0, order="increase", dtype=tla.Int32)
```
