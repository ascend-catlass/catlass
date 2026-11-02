---
nav_order: 40
---

# CATLASS Python API

CATLASS Python 的 API 参考与文档生成维护说明。

| 文档 | 范围 |
| ------ | ------ |
| [API 速查](quick_reference.md) | 按任务与执行上下文选择常用 API：Host、Kernel、Cube、Vector、SIMD、SIMT。 |
| [语言参考](language/index.md) | 按语言概念组织：值与数据类型、Layout 与 Tensor、数据搬运、计算、同步、资源与调试。 |
| [程序生命周期参考](runtime/index.md) | Kernel 入口与语言边界声明、编译启动、Host Tensor 接入。 |
| [API 文档生成](generate_api_docs.md) | 从英文 docstring 重新生成语言参考 / 程序生命周期参考。 |

调用上下文（`Host`、`Kernel`、`Cube`、`Vector`、`SIMD`、`SIMT`）记录在各 API 条目中，不作为一级文档分类。
