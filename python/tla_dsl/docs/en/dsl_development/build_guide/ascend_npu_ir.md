---
nav_order: 40
---

# Building AscendNPU-IR

CATLASS Python builds the TLA Dialect on top of the AscendNPU-IR Dialect. The source headers, generated headers, static libraries, MLIR CMake packages, and Python bindings required for the build all come from AscendNPU-IR.

The AscendNPU-IR version is the revision locked by CATLASS (that is, the commit `feature/regbase@a07821269…` pointed to by `python/tla_dsl/3rdparty/AscendNPU-IR` in `.gitmodules`). For specific environment version requirements, see [Environment Preparation](environment.md#build-environment-requirements).

## Getting the Source Code

AscendNPU-IR is a Git submodule of CATLASS and contains its own submodules. Recursively initialize it in the CATLASS repository root:

```bash
# Replace /path/to/catlass with the actual path of the CATLASS repository you cloned
cd /path/to/catlass
git submodule sync --recursive
git submodule update --init --recursive python/tla_dsl/3rdparty/AscendNPU-IR
```

Confirm that the checked-out commit matches the submodule pointer recorded by CATLASS in the parent repository (the gitlink, that is, the `feature/regbase@a07821269…` locked in `.gitmodules`):

```bash
git submodule status python/tla_dsl/3rdparty/AscendNPU-IR
git -C python/tla_dsl/3rdparty/AscendNPU-IR rev-parse HEAD
```

The compatibility scope of CATLASS Python is determined by the fixed AscendNPU-IR revision recorded in the repository.

## Build

First confirm that Clang, Clang++, lld, CMake, and Ninja are available, for example that `command -v clang clang++ lld cmake ninja` produces no output (meaning they are all in `PATH`), or use `clang --version`, `clang++ --version`, `cmake --version`, and `ninja --version` to check the versions. The following commands are executed in the AscendNPU-IR source root directory:

```bash
cd /path/to/catlass/python/tla_dsl/3rdparty/AscendNPU-IR
./build-tools/build.sh \
  --c-compiler /path/to/clang \
  --cxx-compiler /path/to/clang++ \
  '--add-cmake-options=-DCMAKE_SYSROOT=/' \
  '--add-cmake-options=-DLLVM_ENABLE_ZSTD=OFF' \
  '--add-cmake-options=-DLLVM_ENABLE_RTTI=ON' \
  --build-type Release \
  -j 128 \
  --enable-assertion \
  --disable-werror \
  --disable-mlir-werror \
  --disable-bishengir-werror \
  --build-triton \
  --enable-lld \
  --build ./build \
  --apply-patches
```

Replace `/path/to/clang` and `/path/to/clang++` with the actual compiler paths, for example the results of `$(command -v clang-19)` and `$(command -v clang++-19)`. `-j 128` is only an example; adjust it based on available CPU, memory, and disk I/O, since too much concurrency exhausts resources.

## Verifying the Build Artifacts

Still in the AscendNPU-IR source root directory, run:

```bash
test -f build/install/include/bishengir/Dialect/HIVM/IR/HIVM.h
test -f build/install/include/bishengir/Interfaces/BiShengIREnums.h.inc
test -f third-party/llvm-project/mlir/cmake/modules/AddMLIRPython.cmake
test -f build/install/lib/cmake/mlir/MLIRConfig.cmake
test -f build/install/lib/cmake/llvm/LLVMConfig.cmake
test -f build/install/lib/libMLIRHIVMDialect.a
```

Export the current build tree as:

```bash
export CATLASS_DSL_PREBUILT_ASCENDNPU_IR="$PWD"
```

Then return to the CATLASS Python subproject and continue with [Building CATLASS Python](build.md) and [Running Test Cases](testing.md).
