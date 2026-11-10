---
nav_order: 0
---

# CATLASS Python 文档

## 文档索引

| 文档 | 范围 |
|-----|--------|
| [快速开始](quick_start.md) | 兼容性要求、安装方式与首个示例的最短路径。 |
| [环境准备](dsl_development/build_guide/environment.md) | 环境要求、安装方式（Conda / Docker）与最短上手路径。 |
| [构建 CATLASS Python](dsl_development/build_guide/build.md) | `./build.sh` 构建（Development / Release 模式）、产物检查。 |
| [运行测试用例](dsl_development/build_guide/testing.md) | pytest、lit test 与端到端验证。 |
| [API 速查](api/quick_reference.md) | 按任务与执行上下文选择常用 API。 |
| [语言参考](api/language/index.md) | CATLASS Python 语言表面：值与数据类型、Layout 与 Tensor、计算、同步与资源。 |
| [程序生命周期参考](api/runtime/index.md) | Kernel 入口、编译启动与 Host Tensor 接入。 |
| [Host Tensor 接入](kernel_development/core_concepts/tensor_binding.md) | Host 侧 `from_dlpack` / `make_fake_tensor`，供 `tla.compile` / 启动使用。 |
| [编译与启动](kernel_development/core_concepts/compile_and_launch.md) | 编译样本、缓存、重复启动与 stream。 |
| [结构化 Kernel 参数](kernel_development/core_concepts/structured_arguments.md) | 支持类型、动静结合、自定义 adapter 和编译/启动用法。 |
| [结构化参数设计](dsl_development/feature_development/structured_arguments.md) | 参数树、叶子绑定、设备侧重建和 host ABI。 |
| [静态与动态 Layout](kernel_development/core_concepts/layout.md) | 静态 / 动态 layout；在 Host tensor 上标记动态；在 kernel 中编程。 |
| [语法约束](kernel_development/core_concepts/syntax_guide.md) | `@tla.kernel` 内允许的 Python 写法。 |
| [同步使用指南](kernel_development/core_concepts/sync.md) | 何时需要同步，flag / cross_flag / mutex / barrier 的选择与排查。 |
| [环境变量](kernel_development/core_concepts/env_vars.md) | `CATLASS_DSL_*` 及相关环境变量。 |
| [Kernel 入口 SoC 状态初始化](kernel_development/core_concepts/soc_state_init.md) | 入口自动初始化的状态清单及其效果。 |
| [构建 API 文档](api/generate_api_docs.md) | 如何从英文 docstring 重新生成 Kernel / 程序生命周期参考。 |

英文语言参考 / 程序生命周期参考由脚本生成（`python tools/generate_kernel_api_reference.py`、
`python tools/generate_host_api_reference.py`）；中文稿为手工维护，英文稿变更后请同步更新。
