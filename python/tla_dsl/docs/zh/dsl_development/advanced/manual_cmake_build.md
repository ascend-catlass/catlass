---
nav_order: 50
---

# CMake 目标调试

CATLASS Python 通过 [`build.sh`](../build_guide/build.md) 统一准备 AscendNPU-IR 路径并配置 CMake。本文面向需要单独编译 CMake 目标或运行编译器回归测试的开发者。

## 配置和编译

先加载 CANN，然后在 Python 子项目根目录执行 Development 构建：

```bash
source /path/to/cann/set_env.sh
cd /path/to/catlass/python/tla_dsl
./build.sh
```

`build.sh` 生成 `csrc/mlir/build` 后，可以单独重编译目标：

```bash
cmake --build csrc/mlir/build --target tla-compiler CatlassPythonModules
```

主要产物位于：

```text
csrc/mlir/build/python/catlass/_tla_type_bridge_native*.so
catlass/_mlir/
csrc/mlir/build/tools/tla-compile/TlaCompile
csrc/mlir/build/tests/lit/
```

检查产物：

```bash
test -x csrc/mlir/build/tools/tla-compile/TlaCompile
test -n "$(find csrc/mlir/build/python/catlass -name '_tla_type_bridge_native*.so' -print -quit)"
```

## 运行 lit

```bash
cmake --build csrc/mlir/build --target check-tla-lit
```

也可以使用当前环境中的 lit 执行：

```bash
lit -sv csrc/mlir/build/tests/lit
```

## CMake 配置项

通过 `CMAKE_ARGS` 将配置项传给构建系统：

| 配置项 | 默认值 | 含义 |
| --- | --- | --- |
| `ENABLE_CPU_TRACE_INTRINSIC` | `OFF` | 启用 CPU trace intrinsic |
| `CATLASS_INCLUDE_DIR` | `/path/to/catlass/include` | CATLASS 公共头文件目录 |

例如关闭 HIVM template bitcode：

```bash
CMAKE_ARGS="-DBISHENGIR_BUILD_TEMPLATE=OFF" ./build.sh
```

Release wheel 使用相同的路径准备逻辑：

```bash
./build.sh --release
```

## 找不到 `pybind11`

```bash
python -c "import pybind11; print(pybind11.get_include())"
```

`pybind11` 必须安装在 `build.sh` 使用的 Python 环境中。
