# MxBatchedMatmul Example Readme

## 功能说明

- 算子功能：完成 Ascend 950 上的 MX 量化 Batch 矩阵乘计算，支持 FP8 与 FP4 两种量化精度。本样例包含两个可执行变体，两者共享同一 `MxBatchedMatmulTla` kernel，仅数据类型与 L1/L0 分块大小不同：
  - FP8 变体（`58_ascend950_fp8_mx_batch_matmul`）：A/B 为 `float8_e4m3_t`，累加后截断为 `bf16` 输出以节省带宽。
  - FP4 变体（`58_ascend950_fp4_mx_batch_matmul`）：A/B 为 `float4_e2m1x2_t`，量化误差较大，以 `fp32` 累加直接输出。

- 计算公式：

  $$
    \begin{aligned}
    C &= \tilde{A} \times \tilde{B} \\
    C_{b,i,j} &= \Sigma_{k}\left(2^{s^{A}_{b,i,\lfloor k/32 \rfloor}}\cdot a_{b,i,k}\right)\left(2^{s^{B}_{b,\lfloor k/32 \rfloor,j}}\cdot b_{b,k,j}\right)
    \end{aligned}
  $$

  其中 `A`、`B` 分别是形如 `(b,m,k)`、`(b,k,n)` 的输入矩阵，`C` 是形如 `(b,m,n)` 的输出矩阵；$a$、$b$ 为 FP8/FP4 量化值，$s^{A}$、$s^{B}$ 为沿 K 轴每 32 个元素共享的 MX 缩放因子。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(b, m, k)` | `float8_e4m3_t`（FP8 变体）/ `float4_e2m1x2_t`（FP4 变体） | 左矩阵，layout支持`RowMajor`（默认）和`ColumnMajor` |
| `MxScaleA` | Input | `(b, m, ⌈k/32⌉, 2)` | `float8_e8m0_t` | `A` 的 MX 缩放因子，K 方向每 32 个元素共享一组，最后维度 `2` 为对齐填充 |
| `B` | Input | `(b, k, n)` | `float8_e4m3_t`（FP8 变体）/ `float4_e2m1x2_t`（FP4 变体） | 右矩阵，layout支持`ColumnMajor`（默认）和`RowMajor`，数据类型与左矩阵一致 |
| `MxScaleB` | Input | `(b, n, ⌈k/32⌉, 2)` | `float8_e8m0_t` | `B` 的 MX 缩放因子，K 方向每 32 个元素共享一组，最后维度 `2` 为对齐填充 |
| `C` | Output | `(b, m, n)` | `bf16`（FP8 变体）/ `fp32`（FP4 变体） | Batch 矩阵乘结果，layout支持`RowMajor` |

## 使用范围说明

本样例在基础矩阵乘之上增加 MX 量化与 Batch 批次维度，分别提供 FP8 与 FP4 两种量化精度变体。

推荐范围：

- `m ≥ 256、n ≥ 256`
- `k` 为 32 的倍数（满足 MX 缩放因子沿 K 轴 32 元素分组的语义）
- `k <= 2048`

## 代码组织

```text
58_ascend950_mx_batch_matmul
├── README.md
├── CMakeLists.txt
├── fp8_mx_batch_matmul.cpp  # FP8 变体样例主程序
├── fp4_mx_batch_matmul.cpp  # FP4 变体样例主程序
├── gen_data.py  # 生成 FP8 变体的测试数据（input/ 与 golden/）
└── gen_data_fp4.py  # 生成 FP4 变体的测试数据（input/ 与 golden/）
```

## 使用示例

### 命令行参数

执行样例命令如下：

```bash
# FP8 变体
58_ascend950_fp8_mx_batch_matmul [batchSize] [m] [n] [k] [deviceId]
```

```bash
# FP4 变体
58_ascend950_fp4_mx_batch_matmul [batchSize] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batchSize` | 无 | batch 数 |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

生成测试数据命令行如下：

```bash
# 生成 FP8 变体测试数据
python3 gen_data.py [batch] [m] [n] [k] [trans_a] [trans_b]
```

```bash
# 生成 FP4 变体测试数据
python3 gen_data_fp4.py [batch] [m] [n] [k] [trans_a] [trans_b]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | batch 数 |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `trans_a` | 无 | A 矩阵是否转置（0 不转置 / 1 转置） |
| `trans_b` | 无 | B 矩阵是否转置（0 不转置 / 1 转置） |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。本用例为 Ascend 950 算子，编译时需加 `-DCATLASS_ARCH=3510`。

    ```bash
    # FP8 变体
    bash scripts/build.sh 58_ascend950_fp8_mx_batch_matmul -DCATLASS_ARCH=3510
    # FP4 变体
    bash scripts/build.sh 58_ascend950_fp4_mx_batch_matmul -DCATLASS_ARCH=3510
    ```

2. 在项目根目录下生成测试样例数据。测试样例数据随机生成，尺寸从命令行输入，`trans_a`/`trans_b` 支持 `0`（不转置）/ `1`（转置）。

    ```bash
    # 生成 FP8 变体测试数据（参数: batch m n k trans_a trans_b）
    python3 examples/58_ascend950_mx_batch_matmul/gen_data.py 5 256 512 1024 0 1
    # 生成 FP4 变体测试数据
    python3 examples/58_ascend950_mx_batch_matmul/gen_data_fp4.py 5 256 512 1024 0 1
    ```

3. 在项目根目录下执行算子样例程序。样例按相对路径 `./examples/...` 读取测试数据，请勿切换到 `output/bin` 目录下执行。

    ```bash
    # FP8 变体
    ./output/bin/58_ascend950_fp8_mx_batch_matmul 5 256 512 1024 0
    # FP4 变体
    ./output/bin/58_ascend950_fp4_mx_batch_matmul 5 256 512 1024 0
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
