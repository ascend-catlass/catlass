---
nav_order: 4
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/matrix.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 矩阵运算

[语言参考](index.md) | [API 速查](../quick_reference.md)

Cube 侧矩阵乘加（`tla.mmad` / `tla.mmad_mx`）。

## `tla.mmad`

**源码：** [`catlass.core_api.mmad`](../../../../catlass/core_api.py#L6843)

功能说明：

在 TLA tile 上执行矩阵乘累加。

函数原型：

```python
tla.mmad(acc: Tensor, lhs: Tensor, rhs: Tensor, init_c: bool | Bool | None = None, unit_flag: IndexLike | None = None, compute_order: ComputeOrder = ComputeOrder.M_FIRST, hf32_mode: HF32Mode = HF32Mode.HF32_DISABLE, bias: Tensor | None = None, **extra_kwargs: object) -> None
```

参数说明：

- `acc`（`Tensor`）：累加器 / 输出 tile。必填。
- `lhs`（`Tensor`）：左矩阵 tile。必填。
- `rhs`（`Tensor`）：右矩阵 tile。必填。
- `init_c`（`bool | Bool | None`）：是否先清零累加器；省略时默认为 `False`。
  若提供 `bias`，则由 bias 决定累加起点，`init_c` 对该条指令无效。可选，默认 `None`。
- `unit_flag`（`IndexLike | None`）：unit flag 控制位；省略时默认为 `0`。可选，默认 `None`。
- `compute_order`（`ComputeOrder`）：M/N 计算方向优先级；默认 `M_FIRST`。
- `hf32_mode`（`HF32Mode`）：FP32 操作数在 L0A/L0B 上、矩阵乘之前的 HF32 舍入模式。可选，默认 `HF32_DISABLE`。
- `bias`（`Tensor | None`）：位于 bias 表缓冲（C2 / `biasbuf`）上的一维 `VectorLayout`
  tile，长度为 `N`，元素类型与累加器一致；块内首次 matmul 从该 bias 起累加。
  非 `None` 时累加器从 bias 起步而非 L0C 现值，故 `init_c` 对该指令无效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 内调用；`acc`/`lhs`/`rhs` 须为匹配的 L0 tile。
- 支持的元素类型通路包括 `f16`/`bf16`/`f32`，以及任意 `f8e4m3fn` / `f8e5m2` 操作数配对，累加到 L0C 上的 fp32。
- `init_c` 仅接受 Python `bool` 或 `i1` SSA 值。
- 若设置 `bias`：tile 须在 `biasbuf`、rank-1 `VectorLayout`、元素类型与 `acc` 一致、
  长度 `N` 与 `acc` 的 N 匹配；经 `GM`→`L1`→`biasbuf` 的 `tla.copy` 搬入（见 `copy`）。
  此时忽略 `init_c`。
- 不接受未知关键字参数；传入则报错。

调用示例：

```python
# 前：l0a / l0b 为当前 K 切片；l0c 为 L0C 上的累加器。
with tla.cube():
    tla.mmad(l0c, l0a, l0b, init_c=True, unit_flag=0b11)
    # 带 bias（l0_bias 已由 copy 搬到 biasbuf）：
    # tla.mmad(l0c, l0a, l0b, bias=l0_bias)
    # 后：l0c 累加 lhs@rhs（init_c=True 时先清零）。
```

---

## `tla.mmad_mx`

**源码：** [`catlass.core_api.mmad_mx`](../../../../catlass/core_api.py#L6973)

功能说明：

在 L0 tile 上做 microscaling（MX）矩阵乘累加。

函数原型：

```python
tla.mmad_mx(acc: Tensor, lhs: Tensor, rhs: Tensor, init_c: bool | Bool | None = None, unit_flag: IndexLike | None = None, compute_order: ComputeOrder = ComputeOrder.M_FIRST, bias: Tensor | None = None, **extra_kwargs: object) -> None
```

参数说明：

- `acc`（`Tensor`）：累加器 / 输出 tile，位于 L0C，始终为 fp32。必填。
- `lhs`（`Tensor`）：左矩阵 tile，位于 L0A。必填。
- `rhs`（`Tensor`）：右矩阵 tile，位于 L0B。必填。
- `init_c`（`bool | Bool | None`）：是否先清零累加器；省略时默认为 `False`。
  若提供 `bias`，则由 bias 决定累加起点，`init_c` 对该条指令无效。可选，默认 `None`。
- `unit_flag`（`IndexLike | None`）：unit flag 控制位；省略时默认为 `0`。可选，默认 `None`。
- `compute_order`（`ComputeOrder`）：M/N 计算方向优先级；默认 `M_FIRST`。
- `bias`（`Tensor | None`）：位于 bias 表缓冲（C2 / `biasbuf`）上的一维 `VectorLayout`
  tile，长度为 `N`，始终为 f32（与累加器一致）；块内首次 matmul 从该 bias 起累加。
  非 `None` 时累加器从 bias 起步而非 L0C 现值，故 `init_c` 对该指令无效。可选，默认 `None`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 内调用；`acc`/`lhs`/`rhs` 须为匹配的 L0 tile。
- 操作数须为 `f8e4m3fn`/`f8e5m2` 配对，或 `f4e2m1`/`f4e1m2` 配对；
  同一对内两种格式可混用，累加器为 fp32。
- 两个操作数都必须经 `tla.copy(..., scale=...)` 加载。
- 若设置 `bias`：tile 须在 `biasbuf`、rank-1 `VectorLayout`、元素类型 f32、
  长度 `N` 与 `acc` 的 N 匹配；经 `GM`→`L1`→`biasbuf` 的 `tla.copy` 搬入（见 `copy`）。
  此时忽略 `init_c`。

调用示例：

```python
# 前：l1a/l1b 为 fp8 操作数，l1sa/l1sb 为其 e8m0 scale block。
with tla.cube():
    tla.copy(l0a, l1a, scale=l1sa)   # scale 随 load，不在 matmul 参数里
    tla.copy(l0b, l1b, scale=l1sb)
    tla.mmad_mx(l0c, l0a, l0b, init_c=True)
    # 带 bias：tla.mmad_mx(l0c, l0a, l0b, bias=l0_bias)
    # 后：l0c 累加 microscaled 的 lhs@rhs。
```
