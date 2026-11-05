# MX Batch Matmul (FP8 / FP4) Example Readme

## 功能说明

- 算子功能：完成 Ascend 950 上的 MX 量化 Batch 矩阵乘计算，支持 FP8 与 FP4 两种量化精度，两者共享同一 `MxBatchedMatmulTla` kernel，仅数据类型和分块大小不同
- 计算公式：

  $$
    \begin{aligned}
    C &= (MxScaleA \times A) \times (MxScaleB \times B) \\
    C_{b,i,j} &= \Sigma_{k} \left(2^{s^A_{b,i,\lfloor k/32 \rfloor}} \cdot A_{b,i,k}\right)\left(2^{s^B_{b,j,\lfloor k/32 \rfloor}} \cdot B_{b,k,j}\right)
    \end{aligned}
  $$

  其中$A$和$B$分别是形如`(b, m, k)`，`(b, k, n)`的输入矩阵，$C$是形如`(b, m, n)`的输出矩阵；缩放因子$s^A$、$s^B$为`float8_e8m0_t`类型（纯指数位，值为2的幂），沿K轴每32个元素共享一组（`MX_SCALE_GROUP_NUM=32`）。
- 支持产品型号：Ascend 950 系列产品

## 样例参数说明

| 参数        | 属性   | shape             | dtype                                  | 说明                                                             |
| ----------- | ------ | ----------------- | -------------------------------------- | ---------------------------------------------------------------- |
| `A`         | Input  | `(b, m, k)`       | `float8_e4m3/float8_e5m2`（FP8）<br>`float4_e2m1`（FP4） | 左矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`，本样例仅实现`RowMajor`（`trans_a=0`） |
| `MxScaleA`  | Input  | `(m, mxScaleK, 2)` | `float8_e8m0`                          | A的缩放因子，沿K轴每32个元素共享一组                              |
| `B`         | Input  | `(b, k, n)`       | 同`A`                                  | 右矩阵，layout支持`RowMajor`和`ColumnMajor`（默认），本样例仅实现`ColumnMajor`（`trans_b=1`） |
| `MxScaleB`  | Input  | `(n, mxScaleK, 2)` | `float8_e8m0`                          | B的缩放因子，沿K轴每32个元素共享一组                              |
| `C`         | Output | `(b, m, n)`       | `bfloat16`（FP8）<br>`float`（FP4）     | 矩阵乘结果，layout仅支持`RowMajor`；FP4量化误差较大，以FP32累加直接输出，FP8累加后截断为BF16输出以节省带宽 |

其中 `mxScaleK = RoundUp<2>(ceil(k/32))`，scale的最后一维`2`是对齐填充；FP4数据以`float4_e2m1x2_t`打包类型存储（2个4-bit值打包为1字节）。

## 使用范围说明

本样例为 Ascend 950 上 MX 量化 Batch 矩阵乘的基础实现，未开启 L1 常驻、L0C 双缓冲等优化。

推荐范围：
- `m ≥ 256、n ≥ 256`（与 L1 分块 256×256 对齐）
- `k` 为 32 的倍数（满足 MX 缩放因子沿 K 轴 32 元素分组语义）

### 命令行参数

```bash
58_ascend950_fp8_mx_batch_matmul [batchSize] [m] [n] [k] [deviceId]
58_ascend950_fp4_mx_batch_matmul [batchSize] [m] [n] [k] [deviceId]
```
上述命令行参数具体说明如下：

| 参数        | 默认值 | 参数说明             |
| ----------- | ------ | -------------------- |
| `batchSize` | 无     | batch 数             |
| `m`         | 无     | A 矩阵的 m 轴大小    |
| `n`         | 无     | B 矩阵的 n 轴大小    |
| `k`         | 无     | A/B 矩阵的 k 轴大小  |
| `deviceId`  | `0`    | 指定运行设备ID       |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。本用例为 Ascend 950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。

    ```bash
    bash scripts/build.sh 58_ascend950_fp8_mx_batch_matmul -DCATLASS_ARCH=3510
    bash scripts/build.sh 58_ascend950_fp4_mx_batch_matmul -DCATLASS_ARCH=3510
    ```

2. 在项目根目录下生成测试样例数据。测试样例数据随机生成，尺寸从命令行输入。

    ```bash
    # 生成 FP8 测试数据（参数: batch m n k trans_a trans_b，本样例仅支持 trans_a=0, trans_b=1）
    python3 examples/58_ascend950_mx_batch_matmul/gen_data.py 5 256 512 1024 0 1
    # 生成 FP4 测试数据
    python3 examples/58_ascend950_mx_batch_matmul/gen_data_fp4.py 5 256 512 1024 0 1
    ```

3. 在项目根目录下执行算子样例程序。样例按相对路径 `./examples/...` 读取测试数据，请勿切换到 `output/bin` 目录下执行。

    ```bash
    ./output/bin/58_ascend950_fp8_mx_batch_matmul 5 256 512 1024 0
    ./output/bin/58_ascend950_fp4_mx_batch_matmul 5 256 512 1024 0
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
