---
nav_order: 10
---

# Core Concepts

The core concepts and programming model of CATLASS Python, including Python syntax constraints, control flow, Layout, Tensor integration,
and Host-side compile and launch.

| Document | Scope |
|------|------|
| [Syntax Constraints](syntax_guide.md) | Python constructs and restrictions allowed inside `@tla.kernel`. |
| [Control Flow](control_flow.md) | The boundary between Python staging and runtime control flow. |
| [Layout](layout.md) | The meaning of static / dynamic layout and kernel-side programming. |
| [Python Tensor Integration](tensor_binding.md) | Host-side `from_dlpack` / `make_fake_tensor`. |
| [Compile and Launch](compile_and_launch.md) | Compile samples, caching, repeated launch, and stream. |
| [Structured Kernel Arguments](structured_arguments.md) | How to pass tuple/list, structured objects, static fields, and custom adapters. |
| [Program Lifecycle Reference](../../api/runtime/index.md) | Host-side `@tla.kernel`, `tla.compile` / launch, Host tensor. |
| [Python Environment Variables](env_vars.md) | `CATLASS_DSL_*` and related environment variables. |
