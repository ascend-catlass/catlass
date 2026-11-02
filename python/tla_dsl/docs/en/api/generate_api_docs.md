---
nav_order: 20
---

# Generating API Documentation

CATLASS Python API documentation is generated as Markdown and then built into a static site by the MkDocs configuration at the repository root. Host generation uses AST analysis only. Kernel generation combines AST analysis with runtime docstring hooks that append supported copy routes and matrix data types.

- Language reference: `core_api.py`, `tla/tensor.py` → `docs/en/api/language/`
- Program lifecycle reference: Host source files → `docs/en/api/runtime/`

Each reference is written as a landing page plus one page per top-level topic. SIMD subsections are separate pages under `language/simd/`. Host and kernel are execution-context labels inside API entries, not top-level document sections.

The shared parsing / Markdown rendering lives in `tools/common.py`.

## Prerequisites

Host generation requires only Python source parsing. Kernel generation imports `catlass.core_api` to run its docstring hooks, so it also requires the MLIR Python bindings. Complete a [Development build](../dsl_development/build_guide/build.md) or install a package that provides those bindings before generating or checking the Kernel reference. The same requirement applies to `--check`.

Install the documentation dependencies before building the MkDocs site:

```bash
# Replace /path/to/catlass with the actual path of the CATLASS repository you cloned
cd /path/to/catlass/python/tla_dsl
python -m pip install -r requirements-docs.txt
```

## Generating the API Reference Markdown

Source docstrings use English paragraph headings with directory tags:

- `Directory:`: the path of the API in the reference document TOC (nesting can be expressed with `/`)
- `Description:` / `Parameters:` / `Constraints:` / `Example:`

The generator builds the directory tree from `Directory:`; the section order and section introductions are written in the script's `DIRECTORY_SECTIONS` (Kernel) and `HOST_DIRECTORY_SECTIONS` (Host). Within a section, the API order follows the source definition order.

The generators produce the English reference pages (auto-generated, do not edit manually):

- `docs/en/api/language/index.md` plus topic pages under `docs/en/api/language/` (`tools/generate_kernel_api_reference.py`)
- `docs/en/api/runtime/index.md` plus topic pages under `docs/en/api/runtime/` (`tools/generate_host_api_reference.py`)

The Chinese pages under `docs/zh/api/language/` and `docs/zh/api/runtime/` are **hand-maintained** and use the same page structure; after the English pages change, translate and update the matching Chinese pages in sync.

Environment variables are described in [Environment Variables](../kernel_development/core_concepts/env_vars.md) and are **not** scanned by the program lifecycle generator.

```bash
cd /path/to/catlass/python/tla_dsl
python tools/generate_kernel_api_reference.py
python tools/generate_host_api_reference.py
```

Check whether the generated files match the current code without rewriting them:

```bash
python tools/generate_kernel_api_reference.py --check
python tools/generate_host_api_reference.py --check
```

When a file is out of date, `--check` prints a diff and returns a non-zero status, which is suitable for a pre-commit check.
