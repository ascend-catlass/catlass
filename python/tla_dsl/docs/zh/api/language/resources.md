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

**源码：** [`catlass.core_api.allocate`](../../../../catlass/core_api.py)

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
- `byte_alignment`（`int`）：字节对齐要求。必填。须为该空间本地内存对齐的整数倍；更小或非倍数的值会被拒绝。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- `mem_scope` 须为片上地址空间（l1/l0a/l0b/l0c/ub/biasbuf），不可为 gm/generic；`shape` 须全静态。
- `byte_alignment` 须为该空间本地内存对齐的整数倍，因为缓冲区不能从块中间开始：`l1` / `ub` 为 32 字节，`l0a` / `l0b` 为 512，`l0c` 为 1024，`biasbuf` 为 64。L0C 比 L0 传输块更严格，因为 `tla.mmad` 要求累加器 1024 字节对齐。倍数按请求保留，不要求 2 的幂（1024 可以，L0A 的 1536 也可以）；其余值——包括低于该对齐的值——会被拒绝而非向上取整，因为抬高对齐会把缓冲区放到调用者未请求的对齐上。
- 分配大小同样向上取整到 `byte_alignment`，因此首地址与 `size_bytes` 遵循同一对齐：在 UB 中以 `byte_alignment=32` 分配 16 字节会占用 32，以 256 则会占用 256。
- 一个 kernel 中的全部分配（含对齐填充）求和后必须容纳在所选的地址空间中。上限是缓冲区容量：`l1` 512 KiB，`l0a` / `l0b` 各 64 KiB，`l0c` 256 KiB。`ub` 在此按完整的 256 KiB 检查；更紧的上限（248 KiB，SIMT kernel 则更小）在 lowering 阶段应用，因为核类型与编译器保留开关只有那时才可知。

调用示例：

```python
ptr = tla.allocate(
    shape=(256, 128),
    dtype=tla.Float16,
    mem_scope=tla.AddressSpace.ub,
    byte_alignment=32,
)
```
