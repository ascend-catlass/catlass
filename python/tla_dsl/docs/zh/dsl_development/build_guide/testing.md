---
nav_order: 30
---

# 运行测试用例

## pytest：前端降级到TLA IR

```bash
cd /path/to/catlass/python/tla_dsl
python -m pytest -q tests
```

默认复用 `tests/.cache` 中已编译的 BC 与 kernel 产物。需要清空缓存并重新编译 BC 时传入 `--clean`，重编完成后开始测试。

运行单个测试文件时使用相同入口，例如：

```bash
python -m pytest -q tests/test_frontend_lowering.py
```

按测试文件分发到多个 worker 并行执行：

```bash
python -m pytest -q tests -n 8 --dist loadfile
python -m pytest -q tests -n 8 --dist loadfile --clean
```

- `-n 8` 指定 worker 数，可按机器核数调整；`-n auto` 使用全部 CPU 核。
- `--dist loadfile` (可选)以测试文件为单位分发：同一文件内的用例仍在同一个 worker 中顺序执行，文件级共享状态不会跨 worker 干扰。
- 多个 worker 共享 `tests/.cache`，BC 编译缓存由文件锁串行化；冷缓存下的首次运行需按文件逐个完成编译。
- `--clean` 由 xdist 的 controller 进程在测试开始前清理一次，worker 复用重建后的缓存。

## lit test：TLA IR 降级到 NPUIR

```bash
lit -sv csrc/mlir/build/tests/lit
```

## e2e test：端到端验证

- 该测试额外依赖NPU环境，并且需要安装`torch` / `torch_npu`。

```bash
cd /path/to/catlass
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py --device 0
```

指定矩阵shape、layout和数据类型：

```bash
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py \
    --device 0 \
    --m 256 --n 512 --k 128 \
    --layout-a row --layout-b col \
    --dtype-a f16 --dtype-b f16 --dtype-c f32
```

成功时输出包含：

```text
passed=True cache_key=<CACHE_KEY>
kernel.o=<CACHE_DIR>/<CACHE_KEY>/kernel.o
```

可用参数以脚本帮助为准：

```bash
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py --help
```

## 完整的端到端测试

在 CATLASS 仓库根目录执行：

```bash
cd /path/to/catlass
bash tests/run_dsl_test.sh --device 0
```

该脚本会加载 CANN、检查 AscendNPU-IR、构建 Python，并运行 `tests/dsl_battery` 上板回归测试。
`tests/dsl_battery` 内的用例串行执行，不使用 `pytest-xdist`，也不使用 `-n` 参数。

常用变量如下：

| 变量                                | 含义                                                     |
| ----------------------------------- | -------------------------------------------------------- |
| `ASCEND_HOME_PATH`                  | CANN Toolkit 根目录；脚本也会尝试定位并加载 `set_env.sh` |
| `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` | 共享的 AscendNPU-IR 源码与构建树根目录，作为仓库内子模块之后的回退路径 |
| `CATLASS_DSL_DIR`                   | Python 子项目路径；默认从脚本位置推导                       |
| `DEVICE_ID`                         | NPU device id；默认 `1`，可由 `--device` 覆盖            |
| `CATLASS_DSL_FORCE_RECOMPILE`       | 是否强制重新编译运行时产物；脚本默认设为 `1`             |

查看脚本当前支持的用例和路径解析规则：

```bash
bash tests/run_dsl_test.sh --help
```
