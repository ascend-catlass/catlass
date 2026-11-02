---
nav_order: 9
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/resources.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 资源管理

[语言参考](index.md) | [API 速查](../quick_reference.md)

片上缓冲分配。

## `tla.allocate`

**源码：** [`catlass.core_api.allocate`](../../../../catlass/core_api.py#L9602)

功能说明：

分配本地内存并返回类型化指针。

函数原型：

```python
tla.allocate(shape: ShapeLike, dtype: type[Numeric], mem_scope: AddressSpace, byte_alignment: int) -> Pointer
```

参数说明：

- `shape`（`ShapeLike`）：分配块的形状。必填。
- `dtype`（`type[Numeric]`）：元素数值类型。必填。
- `mem_scope`（`AddressSpace`）：片上地址空间，见
  [片上 AddressSpace](system.md#片上-addressspace)。必填。
- `byte_alignment`（`int`）：字节对齐要求。必填。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `mem_scope` 须为片上地址空间（l1/l0a/l0b/l0c/ub），不可为 gm/generic；`shape` 须全静态。

调用示例：

```python
ptr = tla.allocate(
    shape=(256, 128),
    dtype=tla.Float16,
    mem_scope=tla.AddressSpace.ub,
    byte_alignment=32,
)
```
