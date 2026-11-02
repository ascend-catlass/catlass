---
nav_order: 50
---

# Environment Variables

This document describes the environment variables that can be set when running and compiling CATLASS Python. Unless otherwise noted, all variable names use the `CATLASS_DSL_` prefix.

## Value Conventions for Boolean Switches

Some variables are switches. The following values (case-insensitive) are all treated as on:

`1`, `true`, `yes`, `on`

The following values are all treated as off:

`0`, `false`, `no`, `off`

The "default" in the tables below refers to the value used when the variable is not set.

---

## Compiling and Running

### `CATLASS_DSL_CACHE`

Controls whether the compilation cache is enabled. When enabled, existing compilation results can be reused under the same configuration.

| Value | Meaning |
| --- | --- |
| Default: `1` | Cache enabled |
| `0` | Cache disabled, recompile every time |

---

### `CATLASS_DSL_CACHE_DIR`

Specifies the compilation cache directory.

| Value | Meaning |
| --- | --- |
| Default: `$HOME/.cache/catlass` | If `XDG_CACHE_HOME` is set, this is `$XDG_CACHE_HOME/catlass` |
| Directory path | Write the cache into this directory |

When caching is disabled, this directory is not used persistently.

---

### `CATLASS_DSL_FORCE_RECOMPILE`

Controls whether to force recompilation (ignoring the existing cache).

| Value | Meaning |
| --- | --- |
| Default: `0` | Not forced; the cache may be hit |
| `1` | Force recompilation |

---

### `CATLASS_DSL_PRINT_IR`

Controls whether to print the intermediate representation (IR) during compilation, used to diagnose compilation problems.

| Value | Meaning |
| --- | --- |
| Default: `0` | Do not print |
| `1` | Print the intermediate IR from the compilation process |

---

### `CATLASS_DSL_KEEP`

After a successful compile, **additionally copies** the selected artifacts into the export directory (see `CATLASS_DSL_DUMP_DIR`).
Several values may be given, separated by commas, case-insensitively; unrecognized names are ignored.

Each value selects exactly one artifact and is named after the pipeline stage that produces it.

| Value | Meaning |
| --- | --- |
| Default: empty | Nothing is copied; the artifacts still remain in the cache directory |
| `tla-ir` | Frontend TLA IR (`<kernel>.tlair.mlir`) |
| `tla-passes` | Per-pass IR dump of the TLA pass pipeline (`<kernel>.tla-passes.mlir`); this value enables `CATLASS_DSL_PRINT_IR` on its own |
| `hivmc-input` | The MLIR handed to hivmc-a5 (`<kernel>.mlir`) |
| `kernel` | The device-side compile product (`.o`) |
| `all` | Export every artifact above |

For example: `tla-ir,kernel` or `all`.

Each value enables only the work its own artifact requires, so requesting one artifact never produces another as a side effect.
The former values `ir` and `ir-debug` have been **removed**: `ir-debug` stood for two artifacts at once, and keeping it as an alias would have enabled IR printing for callers that never asked for the pass dump.

---

### `CATLASS_DSL_DUMP_DIR`

The destination directory for files exported by `KEEP`.

| Value | Meaning |
| --- | --- |
| Default: empty | Export into the current working directory when `KEEP` is enabled |
| Directory path | Export into this directory, creating it if it does not exist |

Note: setting this variable alone, without enabling `KEEP`, exports nothing; what is exported is decided entirely by the value of `KEEP`.

---

### `CATLASS_DSL_NPU_DEVICE`

Specifies the NPU device number to use for execution when the current `torch_npu` device is not used.

| Value | Meaning |
| --- | --- |
| Default: `0` | Use device number `0` |
| Non-negative integer (such as `0`, `1`) | Use the specified device number |

If a usable `torch` / `torch_npu` is loaded in the environment, the current torch device takes precedence and this variable has no effect.

---

## Build Environment

### `CATLASS_DSL_PREBUILT_ASCENDNPU_IR`

The locally built AscendNPU-IR directory, used for building CATLASS Python and running tests.

| Value | Meaning |
| --- | --- |
| Default: empty | The build uses the in-repository `python/tla_dsl/3rdparty/AscendNPU-IR` |
| Directory path | Point to a built AscendNPU-IR directory, for example `/opt/AscendNPU-IR` |

---

### `MLIR_TBLGEN_INCLUDE_DIR`

The MLIR header directory used by TableGen generation and the compiler toolchain.

| Value | Meaning |
| --- | --- |
| Default: empty | Set automatically by the build script from the resolved AscendNPU-IR install directory |
| Directory path | Specify the directory manually |

Normally no modification is needed.

---

### `CATLASS_DSL_CATLASS_INCLUDE_DIR`

The directory used when looking for Catlass headers (build-related).

| Value | Meaning |
| --- | --- |
| Default: empty | Derived automatically by CMake from `CATLASS_INCLUDE_DIR` or the repository layout |
| Directory path | Specify the header directory manually |

Normally no modification is needed.

---

## Other Common Environment Variables

### `ASCEND_HOME_PATH`

The installation path of the Ascend CANN toolkit (usually set by `set_env.sh`).

| Value | Meaning |
| --- | --- |
| Default: empty | Rely on the system `PATH` to find tools such as the compiler |
| Directory path | Look for build tools under this installation tree |

---

### `XDG_CACHE_HOME`

Affects the default cache directory when `CATLASS_DSL_CACHE_DIR` is not specified.

| Value | Meaning |
| --- | --- |
| Default: empty | The cache directory is `$HOME/.cache/catlass` |
| Directory path | The default cache is the `catlass` subdirectory under this path |

---

### `PYTHONPATH` / `LD_LIBRARY_PATH`

The Python module search path and the dynamic library search path, respectively. By system convention, multiple directories are separated by `:`. The specific paths to include are described in the environment preparation document.
