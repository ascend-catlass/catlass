---
nav_order: 10
---

# 核心概念

CATLASS Python 的核心概念与编程模型，包括 语法约束、控制流、Layout、Tensor 接入，
以及 Host 侧编译与启动。

| 文档 | 范围 |
|------|------|
| [语法约束](syntax_guide.md) | `@tla.kernel` 内允许的 Python 写法与限制。 |
| [控制流](control_flow.md) | Python staging 与运行时控制流的边界。 |
| [同步使用指南](sync.md) | 何时需要同步、flag / cross_flag / mutex / barrier 的选择与常见错误。 |
| [Layout](layout.md) | 静态 / 动态 layout 的含义与 kernel 侧编程。 |
| [Tensor 接入](tensor_binding.md) | Host 侧 `from_dlpack` / `make_fake_tensor`。 |
| [编译与启动](compile_and_launch.md) | 编译样本、缓存、重复启动与 stream。 |
| [结构化 Kernel 参数](structured_arguments.md) | tuple/list、结构化对象、静态字段与自定义 adapter 的传参方式。 |
| [程序生命周期参考](../../api/runtime/index.md) | Host 侧 `@tla.kernel`、`tla.compile` / 启动、Host tensor。 |
| [环境变量](env_vars.md) | `CATLASS_DSL_*` 及相关环境变量。 |
| [Kernel 入口 SoC 状态初始化](soc_state_init.md) | 入口自动初始化的状态清单、效果与出口恢复。 |
