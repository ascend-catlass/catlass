# Ascend 950 FP8 E4M3 Quant Matmul Example Readme

> **注意**：本样例位于 `experimental/` 目录下，如需编译运行，请先将样例目录拷贝至 `examples/` 下，并在 `examples/CMakeLists.txt` 中添加样例名称 `ascend950_fp8_e4m3_quant_matmul`。

## 功能介绍

- 演示 Ascend 950 上的 **FP8 矩阵乘 + epilogue per-token/per-channel 量化**。
- 计算：`out = (A @ B) * perTokenScale * perChannelScale`。
- A、B 元素类型为 `float8_e4m3_t`，scale 为 `float8_e4m3_t`，输出为 FP32。
- 默认布局为 A `RowMajor`、B `RowMajor`，与 `gen_data.py` 生成的数据一致。

## 代码组织

```text
experimental
└── matmul
    └── ascend950_fp8_e4m3_quant_matmul
        ├── CMakeLists.txt
        ├── README.md
        ├── gen_data.py
        ├── fp8_e4m3_quant_matmul.cpp
        └── test_86_ascend950_fp8_e4m3_quant_matmul.py
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考 [quickstart](../../../docs/zh/1_Practice/01_quick_start.md#编译执行)。本用例为 Ascend 950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。
- 执行算子

```text
# 编译指定用例
bash scripts/build.sh ascend950_fp8_e4m3_quant_matmul -DCATLASS_ARCH=3510
# 生成测试样例（在 examples/ascend950_fp8_e4m3_quant_matmul/data 下生成 input/ 与 golden/）
python3 examples/ascend950_fp8_e4m3_quant_matmul/gen_data.py 256 256 128
# 可选：--data-root <DIR> 指定在 DIR/data/ 下生成（默认在脚本所在目录下生成）
# 输入参数分别对应 m, n, k
# 在 output/bin 中执行，以匹配示例读取数据的相对路径
cd output/bin
./ascend950_fp8_e4m3_quant_matmul 256 256 128 0
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID 可选，默认为 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## optest 测试

先按 [optest 说明](../../../tests/optest/README.md) 构建并安装 `torch_catlass`，然后在仓库根目录执行迁移后的测试件：

```bash
PYTHONPATH="$PWD/tests/optest/tests${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest experimental/matmul/ascend950_fp8_e4m3_quant_matmul/test_86_ascend950_fp8_e4m3_quant_matmul.py -v
```
