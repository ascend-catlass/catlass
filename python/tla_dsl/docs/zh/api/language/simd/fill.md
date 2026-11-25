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

- `value`（`bool | int | float | Numeric | VectorSSA`）：填充常量、运行期
  标量 SSA，或待广播的单 lane vector 片段。必填。
- `dtype`（`type[Numeric]`）：vector 元素类型。必填。支持 Bool、`i8`/`u8`、
  `i16`/`u16`、`i32`/`u32`、`i64`/`u64`、`f16`、`bf16`、`f32`、`f8e4m3fn`、
  `f8e5m2`、`f8e8m0`、`hif8`、`f4e2m1x2`、`f4e1m2x2`；complex 无对应 TLA
  vector 类型，会被拒绝。
- `mask`（`MaskSSA | None`）：谓词哪些 lane 被写入。可选；默认写全部 lane。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func(mode="simd")` 内调用；`value` 须为 Python 标量字面量、
  Host `Numeric`、运行期标量 SSA（如 kernel 标量实参或 tensor `[]` 取值），
  或单 lane vector 片段。运行期标量须为目标元素类型或其广播载体：
  Bool、`hif8`、`f8e8m0`、`f4e2m1x2`/`f4e1m2x2` 经 `i8` 字节载体广播，
  无符号整型经同宽 signless 载体广播，其余类型直接广播自身。16 位浮点
  vector 的标量经 tensor `[]` 取值传入，勿使用 16 位浮点 kernel 标量 ABI
  形参——与 SIMD helper 标量调用组合会触发设备崩溃。
- `f4e2m1x2` / `f4e1m2x2` 的标量是一个已编码的 pair 字节（低半字节在前），
  不是两个独立的 FP4 值；`hif8` 须显式给出编码字节（暂无 Host 编码器）；
  `f8e8m0` 须为可表示范围内（2⁻¹²⁷ ~ 2¹²⁷）的 2 的整数次幂，例如 0.5、
  1.0 和 2.0。
- `f8e4m3fn` / `f8e5m2` 的 trace-time 标量必须能由目标格式精确表示；前端
  遇到不可精确表示的值会报错，不做舍入。请传入已舍入的值（例如 0.125），
  或传入显式转换后的运行期标量。
- 64 位广播由链接的 bitcode helper 实现；64 位的 fragment 源需要 64 位
  `tla.reduce`，而 `tla.reduce` 暂不支持 64 位。
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
