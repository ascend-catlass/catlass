---
nav_order: 10
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/debug.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 调试与打印

[语言参考](index.md) | [API 速查](../quick_reference.md)

kernel 内标量 / tensor 调试打印。

## `tla.print`

**源码：** [`catlass.core_api.print`](../../../../catlass/core_api.py)

功能说明：

在 `cube` / `vector` 区域内打印标量、格式化标量字符串，或打印 GM、UB、L1、L0C tensor 的物理前缀。

函数原型：

```python
tla.print(*args: object, **kwargs: object) -> None
```

参数说明：

- *`args`*（`object`）：要打印的值（可变位置参数）。必填。
- **`kwargs`**（`object`）：不接受关键字参数；传入则报错。可选。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 或 `tla.vector()` 内调用。
- GM/UB 支持 rank-1/rank-2 物理前缀；L1/L0C 仅支持 C310 AIC 和经验证的 CANN 9.1 DebugTunnel ABI。
- L1 须为 32 字节对齐的 zN/nZ 稠密前缀，C0 维度占 32 字节，最多打印 8 个元素，并要求驱动提供 DebugBus HAL 映射接口。
- L0C 须为 32 字节对齐的 L0Clayout 16x16 完整分形（256 个元素）；L0A/L0B 不支持。
- Tensor dtype 支持 `f16`、`f32`、`i8`、`i16`、`i32`、`u8`、`u16`、`u32`。

调用示例：

```python
with tla.vector():
    tla.print(x_scalar)
    tla.print(x_ub, 64)  # tensor + prefix length
```
