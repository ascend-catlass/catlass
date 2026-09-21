# WeightQuantA8W4GroupedMxMatmul Example Readme
> **注意**：社区包暂不支持 950 能力，后续支持的版本敬请期待。
## 功能介绍
- 演示 Ascend 950 上的伪量化场景下的 Group Mx Matmul 矩阵乘法：左矩阵 A 与伪量化后的右矩阵 B 经 MX 缩放（`float8_e8m0`）后在 Cube 上完成乘加，输出为 FP16。
- 本示例中 A 元素类型为 `float8_e4m3_t`，B 元素类型为 `float4_e2m1x2_t`；缩放因子为 `float8_e8m0_t`。未启用 Bias（`ElementBias` 为 `void`）。
- 默认布局为 A `RowMajor`、B `Weight4BitnZ`、C `RowMajor`。
- B矩阵布局仅支持`Weight4BitnZ`, `ColumnMajor`。通常的fp4内部分形为16x64，但这里的Weight4BitnZ内部分形为16x32，和fp8分形保持一致，目的是为了方便vector将fp4 cast到fp8，省去的分形转换，提高算子性能，Weight4BitnZ分形结构如下图：

![image-20260721093458714](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/image-20260721093458714.png)

## 代码组织
```text
├── 74_ascend950_weight_quant_a8w4_grouped_mx_matmul
│   ├── CMakeLists.txt                                  # CMake 编译文件
│   ├── README.md
│   ├── gen_data.py
│   └── weight_quant_a8w4_grouped_mx_matmul.cpp         # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)。本用例为 Ascend 950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。
- 执行算子。
**编译指定用例：**
```bash
bash scripts/build.sh 74_ascend950_weight_quant_a8w4_grouped_mx_matmul -DCATLASS_ARCH=3510
```
**生成测试样例**（在 `examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/data` 下生成 `input/` 与 `golden/`）：
输入支持两种模式，同时支持生成两种布局的 B 矩阵。
**模式一：group_list mode** —— 通过命令行参数显式输入 `group_m_list`。
```bash
python3 examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/gen_data.py group_list 128,128 256 256 256 1
```
**模式二：expect_m_per_group mode** —— 按专家数和期望平均值随机生成 `group_m_list`。
```bash
python3 examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/gen_data.py expect_m_per_group 2 128 256 256 256 1
```
**gen_data.py脚本参数说明：**
| 参数 | 含义 |
| :--- | :--- |
| `group_m_list` | 每个专家对应的分组大小 |
| `expect_m_per_group` | 随机分组模式，按每组期望分组大小随机生成分组 |
| `group_num` | 专家数 / 分组数 |
| `expect_m_per_group` | 每组期望平均分组大小 |
| `m` | 总的 M 上限，要求 `m >= sum(group_m_list)` |
| `k` | 矩阵乘的 k 维 |
| `n` | 矩阵乘的 n 维 |
| `isNz` | 0 为 nd 布局，1 为 nZ 布局 |

需要注意的是，脚本可以生成任意大小n、k数据。
对于本例matmul，nd支持任意shape，Weight4BitnZ需要按分形对齐

**执行测试样例：**
```bash
./output/bin/74_ascend950_weight_quant_a8w4_grouped_mx_matmul 2 256 256 256 0
# 可执行文件名 | 矩阵 m 轴（总m） | n 轴 | k 轴 | Device ID
# Device ID 可选，默认为 0
```
执行结果如下，说明精度比对成功：
```text
Compare success.
```
## 使用说明
**B 矩阵布局切换**
本案例支持 B 矩阵切换`ColumnMajor`和`Weight4BitnZ`格式的输入，默认为 `nZ` 格式。如需切换，请在 `weight_quant_a8w4_grouped_mx_matmul.cpp` 文件中将：
```cpp
using LayoutPrologueB = layout::Weight4BitnZ
```
修改为：
```cpp
using LayoutPrologueB = layout::ColumnMajor
```
生成 nd 布局的 B 矩阵时，请修改 `gen_data.py` 传入参数：
```bash
python3 examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/gen_data.py expect_m_per_group 2 128 256 256 256 0
```
**算子计算逻辑**
本 example 完成 Grouped mx 量化矩阵乘：
```
C = (MxScaleA x A) * (MxScaleB x B) + Bias
```
- A、B 支持数据类型为 `float8_e4m3` 和 `float4_e2m1`，B 矩阵Cast为 `float8_e4m3` 后参与 Cube 计算。
- `MxScaleA`、`MxScaleB` 支持数据类型为 `float8_e8m0`。
## 性能测试
本例性能对比如下表格所示
| group_num | m_per_expert | n | k | catlass(us) | 标杆(us) | 标杆/catlass |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 48 | 64 | 4096 | 1024 | 89.573 | 164.111 | 1.83214808 |
| 48 | 80 | 4096 | 1024 | 94.001 | 167.312 | 1.779895959 |
| 48 | 96 | 4096 | 1024 | 96.111 | 178.953 | 1.861940881 |
| 48 | 112 | 4096 | 1024 | 101.063 | 181.987 | 1.800728259 |
| 48 | 128 | 4096 | 1024 | 106.075 | 162.171 | 1.528833373 |

本例采用了动态 tiling、L2 Cache Hit、Double Buffer 等优化，实现性能提升 10%+。
