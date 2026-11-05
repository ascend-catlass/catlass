# FP8MxGroupedMatmulFinalizeRouting Example Readme

## 功能介绍

- 演示 Ascend 950 上的 **Grouped MXFP8矩阵乘 + Finalize Routing** 融合算子：
  - 多组（grouped）MX FP8 矩阵乘：A、B 为 MX FP8，经 `float8_e8m0` 缩放后做分组矩阵乘，结果写入 workspace 中间缓冲
  - AIV 侧完成后处理：输出清零、可选共享专家输出赋值、Logit 加权、Scatter Add 聚合
  - 支持可选 bias（BFLOAT16）、可选共享专家输入（SharedInput）、B 矩阵转置等特性
- 本示例中 A、B 为 `float8_e4m3_t` 或 `float8_e5m2_t`，MX 量化scale为 `float8_e8m0_t`，输出为 `float`。
- 默认布局：A `RowMajor`、B 支持 `RowMajor`（不转置）或 `ColumnMajor`（转置）、Out `RowMajor`。

## 代码组织

```text
experimental
├── gmm
│   ├── 71_ascend950_fp8_mx_grouped_matmul_finalize_routing
│   │   ├── CMakeLists.txt                          # CMake 编译配置
│   │   ├── README.md
│   │   ├── 71_ascend950_fp8_mx_grouped_matmul_finalize_routing.md  # 设计文档
│   │   ├── gen_data_compare.py                     # 生成 data/input/ 与 golden/，执行对比
│   │   ├── fp8_mx_grouped_matmul_finalize_routing.cpp          # 确定性版主程序
│   │   └── fp8_mx_grouped_matmul_finalize_routing_no_deter.cpp # 非确定性版主程序
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)，本用例为 Ascend 950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。
- 本目录提供两个版本：**确定性版**（`ColumnBlockSwizzle`调度）和**非确定性版**（`GemmGroupedAswtTailSplitSwizzle`调度，多核利用率更高），两者输入参数与精度一致，仅调度策略不同。
- 输入参数说明：`problem_count, m, n, k, trans_b, group_list_type, enable_bias, batch, data_parallel_size, enable_shared_input, shared_input_weight, shared_input_offset, quant_type, device_id`（Device ID 可选，默认为 0）

### 确定性版

```bash
# 编译
bash scripts/build.sh 71_ascend950_fp8_mx_grouped_matmul_finalize_routing -DCATLASS_ARCH=3510

# 生成测试样例并执行精度对比（在 data/ 下生成 input/ 与 golden/）
python3 examples/71_ascend950_fp8_mx_grouped_matmul_finalize_routing/gen_data_compare.py 4 128 128 128 0 0 0 16 2 0 0.0 0 float8_e5m2 0 --deter

# 单独执行测试样例
./output/bin/71_ascend950_fp8_mx_grouped_matmul_finalize_routing 4 128 128 128 0 0 0 16 2 0 0.0 0 float8_e5m2 0
```

### 非确定性版（no_deter）

```
# 编译
bash scripts/build.sh 71_ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter -DCATLASS_ARCH=3510

# 生成测试样例并执行精度对比（在 data/ 下生成 input/ 与 golden/）
python3 examples/71_ascend950_fp8_mx_grouped_matmul_finalize_routing/gen_data_compare.py 4 128 128 128 0 0 0 16 2 0 0.0 0 float8_e5m2 0

# 单独执行测试样例
./output/bin/71_ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter 4 128 128 128 0 0 0 16 2 0 0.0 0 float8_e5m2 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 使用说明

1、本 example 完成 grouped MX 量化矩阵乘 + Finalize Routing 融合操作：
C = (MxScaleA \* A) @ (MxScaleB \* B) + Bias
out[rowIndex[p], :] += logit[p] \* C[p, :]
若启用SharedInput：out[offset:offset+bsdp, :] += sharedInputWeight \* SharedInput

2、输入B及其scale支持转置且两者转置状态须保持一致。`gen_data_compare.py`会根据`trans_b`参数生成对应格式的数据。因为layout隐式表征转置状态，即layout::RowMajor表示不转置，layout::ColumnMajor表示转置。

3、A、B支持数据类型为float8_e4m3或float8_e5m2
MxScaleA、MxScaleB支持数据类型为float8_e8m0
A的shape为(m, k), MxScaleA的shape为(m, ceil(k/64), 2)
B不转置的shape为(problem_count, k, n), MxScaleB的shape为(problem_count, ceil(k/64), n, 2)
B转置的shape为(problem_count, n, k), MxScaleB的shape为(problem_count, n, ceil(k/64), 2)
输出Out的数据类型为float，shape为(batch, n)

4、groupListType支持两种模式：0为cumsum（前缀和）模式，1为count模式。

5、可选参数bias为BFLOAT16类型，shape为(problem_count, n)。

6、可选参数SharedInput为BFLOAT16类型，shape为(bsdp, n)，其中bsdp = batch / dataParallelSize。sharedInputWeight为FLOAT32标量系数，sharedInputOffset为输出中的行偏移量。

## 特殊说明

- 当前example中的L1TileShape(256,256,256)和L0TileShape(256,256,128)。AIC完成矩阵乘后结果写入GM workspace，AIV从workspace读取进行后处理。
- groupList中未指定的部分将不会参与更新。如groupList为(3,4,5)，m为20，则仅前12行参与计算，其余行不影响输出。
- 当前实现为 mix kernel，CMakeLists.txt 中通过 `catlass_example_add_executable(... mix ...)` 指定，编译时需开启 `L2_CACHE_HINT` 宏定义。
- AIC/AIV通过CrossCore Flag实现tile粒度的流水线化交替执行，避免全局同步。

## 确定性版与非确定性版（no_deter）差异

| 对比项         | 确定性版                                                | 非确定性版（no_deter）                                                        |
| -------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Kernel         | `GroupedMxMatmulFinalizeRoutingTla`                   | `GroupedMxMatmulFinalizeRoutingNoDeterTla`                                  |
| BlockScheduler | `ColumnBlockSwizzle`（按列分块调度）                  | `GemmGroupedAswtTailSplitSwizzle`（滚动核分配 + 窗口调度）                  |
| BlockEpilogue  | `BlockEpilogueFinalizeRouting`（AIV按N维切分）        | `BlockEpilogueFinalizeRoutingNoDeter`（AIV按M维切分）                       |
| 尾块处理       | 无尾块拆分                                              | 支持尾部tile多核拆分（`UpdateTailTile`），最后一个group在满足条件时自动启用 |
| 可执行文件     | `71_ascend950_fp8_mx_grouped_matmul_finalize_routing` | `71_ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter`              |

- 非确定性版采用 `GemmGroupedAswtTailSplitSwizzle` 调度器，`startBlockIdx_` 跨 group 滚动，提升多核利用率和尾块负载均衡，适用于对确定性无要求的场景。
- 两版的输入参数、数据格式、精度计算逻辑完全一致，仅调度与AIV后处理切分方向不同。
