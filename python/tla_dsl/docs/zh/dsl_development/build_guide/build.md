---
nav_order: 20
---

# 构建 CATLASS Python

除特别说明外，命令均在 `/path/to/catlass/python/tla_dsl` 执行，其中 `/path/to/catlass` 需替换为你 clone 的 CATLASS 仓库根目录。

## Development 模式构建（开发态）

```bash
cd /path/to/catlass/python/tla_dsl
./build.sh
# 强制清理并重新构建
./build.sh --clean
```

`build.sh` 默认执行 Development 模式构建，主要步骤为：

1. 检查 `ASCEND_HOME_PATH` 和 AscendNPU-IR 构建产物。
2. 在 `csrc/mlir/build` 中构建编译器和 Python 扩展。
3. 以 editable 模式安装 `ascend-catlass-dsl`，且不重复安装依赖。

- `--clean` 会删除 Python 子项目中的 `build/`、`csrc/mlir/build/`、`dist/`、egg-info、pytest 缓存和二进制库（`_tla_type_bridge_native*.so`）后，再执行 Development 模式构建。

构建成功后可检查关键产物：

```bash
test -x csrc/mlir/build/tools/tla-compile/TlaCompile
test -n "$(find csrc/mlir/build/python/catlass -name '_tla_type_bridge_native*.so' -print -quit)"
python -c "from catlass import tla"
```

以上命令均无任何输出、退出码为 `0`，即表示检查通过、产物就绪。

## Release 模式

```bash
./build.sh --release
ls dist/*.whl
```

Release 模式在 `dist/` 生成 wheel。
