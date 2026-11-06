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

**源码：** [`catlass.core_api.VectorSSA.reduce`](../../../../../catlass/core_api.py)

功能说明：

在 mask 下对本寄存器驻留 vector 做归约。`mode` 选择归约范围：

- `ReduceMode.INTER_BLOCK`（默认）：跨全部 lane 归约
  （`tla.ReductionOp.ADD` / `MAX` / `MIN`），得到单 lane 片段。
  可用 `tla.full(fragment, dtype=...)` 再广播回全宽。
- `ReduceMode.INTRA_BLOCK`：对每个 32 字节数据块独立归约
  （`tla.ReductionOp.ADD` / `MAX` / `MIN`）；每个块的结果连续写入
  返回 vector 的最低有效 lane（packed）；未被块结果占用的 lane 置 0，
  全屏蔽块的结果 lane 为恒等值（见约束说明）。结果除 i16/u16 的
  `ADD` 返回加宽的 32 位结果（见约束说明）外，与操作数的 shape 和
  元素类型一致。

函数原型：

```python
tla.VectorSSA.reduce(kind: ReductionOp, *, mode: ReduceMode = ReduceMode.INTER_BLOCK, mask: Any, init_value: Any | None = None, reduction_profile: Any | None = None) -> Any
```

参数说明：

- `kind`（`ReductionOp`）：归约种类（`ADD`、`MAX` 或 `MIN`）。必填。
- `mode`（`ReduceMode`）：归约范围。可选，默认 `ReduceMode.INTER_BLOCK`。
- `mask`（`MaskSSA`）：谓词哪些 lane 参与。必填。
- `init_value`：预留；当前仅支持 `None`（`INTER_BLOCK`）。可选，默认 `None`。
- `reduction_profile`：预留；当前仅支持 `None`（`INTER_BLOCK`）。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`INTRA_BLOCK` 须在
  `tla.vec.func(mode="simd")` 内调用。
- 元素类型须为 f16/f32/i16/i32/u16/u32 之一。
- `INTER_BLOCK`：`init_value` 与 `reduction_profile` 须省略或为 `None`。
- `INTRA_BLOCK`：`init_value` 与 `reduction_profile` 须省略或为 `None`。
  完全 mask 掉的块得到硬件恒等值：`ADD` 为 0；`MAX` 为元素类型
  最小值（浮点为负无穷），`MIN` 为元素类型最大值（浮点为正无穷）。
- `INTRA_BLOCK` 的 i16/u16 源 `ADD` 返回底层指令加宽后的 32 位结果
  （与 vcadd 相同）：结果 lane 数按 32 位结果 lane 减半并向上取整
  （尾块最后一个有效元素仍保留其加宽 lane）、元素类型为 i32/u32，
  且不做转回 16 位的 cast。超出源类型范围的块内累加和在 i32/u32
  结果中保持精确。

调用示例：

```python
with tla.vec.func(mode="simd"):
    m = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    total = src_reg.reduce(tla.ReductionOp.ADD, mask=m)
    broadcast = tla.full(total, dtype=tla.Float32)
    packed = src_reg.reduce(
        tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=m
    )
```

---

## `tla.gather`

**源码：** [`catlass.core_api.gather`](../../../../../catlass/core_api.py)

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
