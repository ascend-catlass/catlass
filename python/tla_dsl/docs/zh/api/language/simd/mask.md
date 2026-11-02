---
nav_order: 51
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/mask.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# Mask 计算

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

Mask 创建与尾块更新。

## `tla.create_mask`

**源码：** [`catlass.core_api.create_mask`](../../../../../catlass/core_api.py#L10064)

功能说明：

按固定 pattern（`tla.mask.*` token）创建 vector mask。

函数原型：

```python
tla.create_mask(*, pattern: _MaskPattern | str | None = None, dtype: DTypeLike = Float32) -> MaskSSA
```

参数说明：

- `pattern`（`_MaskPattern | str | None`）：掩码 pattern token 或其名字字符串。
  运行时必填（传 `None` 会报错）。token 挂在 `tla.mask` 下：

  | Pattern | 含义 |
  | --- | --- |
  | `ALL` | 全部元素有效 |
  | `ALLF` | 全部元素无效 |
  | `VL1` / `VL2` / `VL3` / `VL4` | 最低 1 / 2 / 3 / 4 个元素有效 |
  | `VL8` / `VL16` / `VL32` / `VL64` / `VL128` | 最低 8 / 16 / 32 / 64 / 128 个元素有效 |
  | `M3` | 下标为 3 的倍数的元素有效 |
  | `M4` | 下标为 4 的倍数的元素有效 |
  | `H` | 最低一半元素有效 |
  | `Q` | 最低四分之一元素有效 |

- `dtype`（`DTypeLike`）：与掩码关联的元素类型（也决定一条 vector 能放多少元素：
  256 字节 / 元素大小）。可选，默认 `Float32`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用；`pattern` 必填。
- 接受 `mask=` 的接口需要 `create_mask` / `update_mask` 得到的 `MaskSSA`，
  不能直接传 `tla.mask.*` token。

调用示例：

```python
with tla.vec.func(mode="simd"):
    # 用 pattern token 构建 mask：
    m_all = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
    m_tail = tla.create_mask(pattern=tla.mask.VL8, dtype=tla.Float16)
    # 前：x_reg / y_reg 为整 vector；只让最低 8 个元素做加法。
    z = tla.add(x_reg, y_reg, mask=m_tail)
    # 后：有效元素为 x+y；被 mask 掉的元素不参与。
```

---

## `tla.update_mask`

**源码：** [`catlass.core_api.update_mask`](../../../../../catlass/core_api.py#L10128)

功能说明：

创建尾部 mask，并返回剩余元素计数。

函数原型：

```python
tla.update_mask(true_shape: IndexLike, dtype: DTypeLike = Float32) -> tuple[MaskSSA, Numeric]
```

参数说明：

- `true_shape`（`IndexLike`）：当前有效（true）区域形状。必填。
- `dtype`（`DTypeLike`）：与掩码关联的元素类型。可选，默认 `Float32`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.vec.func()` 内调用。

调用示例：

```python
with tla.vec.func(mode="simd"):
    tail_mask, remain = tla.update_mask(true_shape, dtype=tla.Float32)
```
