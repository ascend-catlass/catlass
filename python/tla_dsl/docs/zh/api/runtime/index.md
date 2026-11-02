<!--
手工维护的中文程序生命周期参考。
英文源：docs/en/api/runtime/index.md
（由 python/tla_dsl/tools/generate_host_api_reference.py + 源码英文 docstring 生成）。
英文稿变更后请同步翻译更新本文件。
不要用术语表 / glossary 自动生成本文件。
-->

# CATLASS Python 程序生命周期参考

先通过 [API 速查](../quick_reference.md) 按任务与执行上下文选择接口。
本文档覆盖 **CATLASS Python 程序生命周期**：Kernel 入口与语言边界声明，
`tla.compile` / `JitCompiledFunction` 编译启动，以及 Host tensor 绑定。
使用流程见 [编译与启动](../../kernel_development/core_concepts/compile_and_launch.md)，环境变量见
[环境变量](../../kernel_development/core_concepts/env_vars.md)。语言接口见
[语言参考](../language/index.md)。

接口说明与调用示例来自各 API 源码 docstring；这些接口均在 Python Host 脚本中、
`@tla.kernel` 函数体**外**调用。

DLPack 接入教程见 [Host Tensor 接入](../../kernel_development/core_concepts/tensor_binding.md)。
动态 layout 编程见 [静态与动态 Layout](../../kernel_development/core_concepts/layout.md)。

## 分类页面

| 章节 | 页面 |
| --- | --- |
| 1 | [Kernel 入口与语言边界](kernel_entry.md) |
| 2 | [编译与启动](compile_launch.md) |
| 3 | [Host Tensor 接入](host_tensor.md) |

---
