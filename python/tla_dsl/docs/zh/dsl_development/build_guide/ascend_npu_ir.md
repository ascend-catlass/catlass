---
nav_order: 40
---

# 构建 AscendNPU-IR

CATLASS Python 基于 AscendNPU-IR 的 Dialect 构建 TLA Dialect。构建所需的源码头文件、生成头文件、静态库、MLIR CMake 包和 Python bindings 均来自 AscendNPU-IR。

AscendNPU-IR 的版本以 CATLASS 锁定的 revision 为准（即 `.gitmodules` 中 `python/tla_dsl/3rdparty/AscendNPU-IR` 指向的提交 `stable@693abf31…`）。具体的环境版本要求统一见[环境准备](environment.md#构建环境要求)。

## 获取源码

AscendNPU-IR 是 CATLASS 的 Git 子模块，并包含自身子模块。在 CATLASS 仓库根目录递归初始化：

```bash
# /path/to/catlass 需替换为你 clone 的 CATLASS 仓库实际路径
cd /path/to/catlass
git submodule sync --recursive
git submodule update --init --recursive python/tla_dsl/3rdparty/AscendNPU-IR
```

确认检出的提交与 CATLASS 在父仓库中记录的子模块指针（gitlink，即 `.gitmodules` 锁定的 `stable@693abf31…`）一致：

```bash
git submodule status python/tla_dsl/3rdparty/AscendNPU-IR
git -C python/tla_dsl/3rdparty/AscendNPU-IR rev-parse HEAD
```

CATLASS Python 的兼容范围以仓库记录的 AscendNPU-IR 固定版本（revision）为准。

## 构建

先确认 Clang、Clang++、lld、CMake 和 Ninja 可用，例如 `command -v clang clang++ lld cmake ninja` 无输出即全部在 `PATH` 中，或用 `clang --version`、`clang++ --version`、`cmake --version`、`ninja --version` 查看版本。以下命令在 AscendNPU-IR 源码根目录执行：

```bash
export CATLASS_DSL_ASCENDNPU_IR_ROOT="/path/to/catlass/python/tla_dsl/3rdparty/AscendNPU-IR"

bash /path/to/catlass/python/tla_dsl/tools/patches/apply_patch.sh

cd $CATLASS_DSL_ASCENDNPU_IR_ROOT
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
  --bishengir-publish OFF \
  --build ./build
```

将 `/path/to/clang` 和 `/path/to/clang++` 替换为实际编译器路径，例如 `$(command -v clang-19)` 和 `$(command -v clang++-19)` 对应的结果。`-j 128` 只是示例，请根据可用 CPU、内存和磁盘 I/O 调整；并发数过大会导致资源耗尽。

## 验证构建产物

仍在 AscendNPU-IR 源码根目录执行：

```bash
install_paths=(
  build/install/include/bishengir/Dialect/HIVM/IR/HIVM.h
  build/install/include/bishengir/Interfaces/BiShengIREnums.h.inc
  build/install/lib/cmake/mlir/MLIRConfig.cmake
  build/install/lib/cmake/llvm/LLVMConfig.cmake
  build/install/lib/libBiShengIRHIVMDialect.a
  build/install/lib/libMLIRHIVMAVEToStandard.a
)
for p in "${install_paths[@]}"; do
  [[ -e "$p" ]] && echo "OK   $p" || echo "MISS $p"
done
```

将当前构建树导出为：

```bash
export CATLASS_DSL_PREBUILT_ASCENDNPU_IR="$PWD"
```

然后返回 Python 子项目并按照[构建 CATLASS Python](build.md)和[运行测试用例](testing.md)继续。
