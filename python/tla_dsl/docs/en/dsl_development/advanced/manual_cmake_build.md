---
nav_order: 50
---

# CMake Target Debugging

CATLASS Python uses [`build.sh`](../build_guide/build.md) to prepare the AscendNPU-IR paths and configure CMake uniformly. This document is for developers who need to compile CMake targets individually or run compiler regression tests.

## Configuration and Compilation

First load CANN, then run a Development build in the CATLASS Python subproject root:

```bash
source /path/to/cann/set_env.sh
cd /path/to/catlass/python/tla_dsl
./build.sh
```

After `build.sh` generates `csrc/mlir/build`, you can rebuild targets individually:

```bash
cmake --build csrc/mlir/build --target tla-compiler CatlassPythonModules
```

The main artifacts are located at:

```text
csrc/mlir/build/python/catlass/_tla_type_bridge_native*.so
catlass/_mlir/
csrc/mlir/build/tools/tla-compile/TlaCompile
csrc/mlir/build/tests/lit/
```

Check the artifacts:

```bash
test -x csrc/mlir/build/tools/tla-compile/TlaCompile
test -n "$(find csrc/mlir/build/python/catlass -name '_tla_type_bridge_native*.so' -print -quit)"
```

## Running lit

```bash
cmake --build csrc/mlir/build --target check-tla-lit
```

You can also run it with the lit in the current environment:

```bash
lit -sv csrc/mlir/build/tests/lit
```

## CMake Configuration Options

Pass configuration options to the build system through `CMAKE_ARGS`:

| Option | Default | Meaning |
| --- | --- | --- |
| `ENABLE_CPU_TRACE_INTRINSIC` | `OFF` | Enable the CPU trace intrinsic |
| `CATLASS_INCLUDE_DIR` | `/path/to/catlass/include` | CATLASS common header directory |

For example, to disable the HIVM template bitcode:

```bash
CMAKE_ARGS="-DBISHENGIR_BUILD_TEMPLATE=OFF" ./build.sh
```

The Release wheel uses the same path preparation logic:

```bash
./build.sh --release
```

## `pybind11` Not Found

```bash
python -c "import pybind11; print(pybind11.get_include())"
```

`pybind11` must be installed in the Python environment used by `build.sh`.
