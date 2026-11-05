# FP8MxGroupedMatmulSliceMSwigluMxQuant Example Readme

## 功能介绍

- 演示 Ascend 950 上的 **Grouped MXFP8矩阵乘 + SwiGLU + MXFP8量化**：
  - 多组（grouped）矩阵乘按 M 轴切分（slice-M）：A、B 为 MX FP8，经 `float8_e8m0` 缩放后做矩阵乘，结果按N轴均分成Act和Gate，Act和Gate进行SwiGlu激活后量化为MXFP8
  - 本示例中 A、B 为 `float8_e4m3_t`，MX 量化scale为 `float8_e8m0_t`，量化输出结果为 `float8_e4m3_t`，量化结果scale为`float8_e8m0_t`。
- 默认布局：A `RowMajor`、B `ColumnMajor`、Q `RowMajor`、 QScale `RowMajor`。B及其scale支持转置。

## 代码组织

```text
experimental
├── gmm
│   ├── 65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant
│   │   ├── CMakeLists.txt                          # CMake 编译配置
│   │   ├── README.md
│   │   ├── gen_data.py                             # 生成 data/input/
│   │   ├── compare.py                              # 计算golden，对比结果
│   │   └── fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant.cpp      # 主程序
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)，本用例为 Ascend 950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant -DCATLASS_ARCH=3510

# 生成测试样例（在 examples/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant/data 下生成 input/ 与 golden/）
python3 examples/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant/gen_data.py 4 256 512 1024
# 输入参数分别对应 problem_count， m， n， k，其中n需要128对齐，problem_count不超过1024

# 执行测试样例
./output/bin/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant 4 256 512 1024 0
# 可执行文件名 | problem_count | M | n | K | Device ID
# Device ID 可选，默认为 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 使用说明

1、本 example 完成 grouped MX 量化矩阵乘 + SwiGLU + 在线 MX 量化输出：
C = (MxScaleA \* A) @ (MxScaleB \* B)
act,gate = C
S = Swish(act) * gate
Q, QScale = MXQuant(S)

2、输入B及其scale支持转置且两者转置状态须保持一致。`gen_data.py`会同时生成B转置的和不转置的文件，当前样例为B转置的示例。若要改为B不转置，请修改example示例中B的layout为layout::RowMajor和读取二进制文件的名称。因为layout隐式表征转置状态，即layout::RowMajor表示不转置，layout::ColumnMajor表示转置。

3、A、B支持数据类型为float8_e4m3或float8_e5m2
MxScaleA、MxScaleB支持数据类型为float8_e8m0
A的shape为(m, k), MxScaleA的shape为(m, ceil(k/64), 2)
B不转置的shape为(problem_count, k, n), MxScaleB的shape为(problem_count, ceil(k/64), n, 2)
B转置的shape为(problem_count, n, k), MxScaleB的shape为(problem_count, n, ceil(k/64), 2)
输出Q的数据类型为float8_e4m3或float8_e5m2，与输入A、B保持一致，shape为(m, n/2)
输出QScale的数据类型为float8_e8m0，shape为(problem_count, ceil((n/2)/64), 2)

4、关于Mx量化矩阵乘的详细特征详见[53_ascend950_fp8_mx_matmul](../53_ascend950_fp8_mx_matmul/README.md)和[54_ascend950_fp4_mx_matmul](../54_ascend950_fp4_mx_matmul/README.md)中说明文档的相关内容。

## 特殊说明

- 当前example中的L1TileShape(128,256,256)和L0TileShape(128,256,128)仅为样例[53_ascend950_fp8_mx_matmul](../53_ascend950_fp8_mx_matmul/README.md)的一半，原因在于在一次循环中需要进行两个block的matmul运算，这两次matmul的结果需要同时存在于UB上用来进行后续计算，同时后续的计算中也需要UB存放中间结果，当前场景下UB存放数据共使用约180K。因此，受限于UB大小，当前example的L1TileShape和L0TileShape仅为样例[53_ascend950_fp8_mx_matmul](../53_ascend950_fp8_mx_matmul/README.md)的一半。
- groupList中未指定的部分将不会参与更新。如groupList为(3,4,5)，m为20，则Q[12:]和QScale[12:]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。即Q和QScale的有效数据为Q[:12]和QScale[:12]。
