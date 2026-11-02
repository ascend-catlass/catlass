---
nav_order: 40
---

# CATLASS Python API

API references and documentation generation/maintenance instructions for CATLASS Python.

| Document | Scope |
| ------ | ------ |
| [API Quick Reference](quick_reference.md) | Choose common APIs by task and execution context: Host, kernel, Cube, Vector, SIMD, and SIMT. |
| [Language Reference](language/index.md) | Values and types, layout and tensor, data movement, compute, synchronization, resources, and debug, organized by language concept. |
| [Program Lifecycle Reference](runtime/index.md) | Kernel entry, language-boundary declarations, compile and launch, and Host tensor binding. |
| [API Documentation Generation](generate_api_docs.md) | Regenerate the Language / Program Lifecycle reference from English docstrings. |

Execution context (`Host`, `Kernel`, `Cube`, `Vector`, `SIMD`, `SIMT`) is a property recorded in each API entry, not a top-level document category.
