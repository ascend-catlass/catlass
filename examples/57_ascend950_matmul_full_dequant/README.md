# MatmulFullDequant Example Readme

## 代码组织

```text
├── 57_ascend950_matmul_full_dequant
│   ├── CMakeLists.txt  # CMake编译文件
│   ├── README.md
│   └── matmul_full_dequant.cpp # 主文件
```

- 支持的量化模式

| X1的量化模式 | X2的量化模式 | 带有Bias |
| ------------ | ------------ | -------- |
| per_token    | per_tensor   | False    |
| per_token    | per_channel  | False    |
| per_tensor   | per_channel  | False    |
| default      | per_channel  | False    |
| per_token    | per_tensor   | True     |
| per_token    | per_channel  | True     |
| default      | per_tensor   | True     |
| default      | per_channel  | True     |

default 模式即 `None`（`QuantMode::NONE`），表示该输入不采用任何量化模式。

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)，本用例为Ascend 950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 57_ascend950_matmul_full_dequant -DCATLASS_ARCH=3510
# 生成测试样例
cd examples/57_ascend950_matmul_full_dequant
python3 scripts/gen_data.py --shape "64 64 64" --x1_quant_mode per_token --x2_quant_mode per_channel
# 带有bias使用下面的命令
# python3 scripts/gen_data.py --shape "64 64 64" --x1_quant_mode per_token --x2_quant_mode per_channel --has_bias
cd ../../output/bin
cp -r ../../examples/57_ascend950_matmul_full_dequant/input/ .
cp -r ../../examples/57_ascend950_matmul_full_dequant/output/ .
# 可执行文件名 |矩阵m轴|n轴|k轴|x1QuantMode|x2QuantMode|has_bias
# has_bias可选，默认不带有bias；量化模式取值与 gen_data.py 保持一致
# default 也可写作 none
./57_ascend950_matmul_full_dequant 64 64 64 per_token per_channel
# 带有bias使用下面命令执行，需要配合gen_data.py生成bias
# ./57_ascend950_matmul_full_dequant 64 64 64 per_token per_channel has_bias
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```
