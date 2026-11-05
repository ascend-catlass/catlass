# QuantMatmulFullLoadATla Example Readme

## 功能说明

- 算子功能：完成量化矩阵乘（Quant Matmul）计算，并在 12_quant_matmul/42_quant_optimized_matmul_tla 基础上支持 A 矩阵全载（Full LoadA）。左矩阵 `A` 与右矩阵 `B` 完成 int8 矩阵乘后，使用 per-channel 量化系数 `scale` 与 per-token 量化系数 `per_token_scale` 进行反量化。
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \left(\Sigma_{k} A_{i,k}B_{k,j}\right) \cdot scale_j \cdot per\_token\_scale_i
    \end{aligned}
  $$

  其中 `A` 是形如 `(m,k)` 的左矩阵，`B` 是形如 `(k,n)` 的右矩阵，`scale` 为 per-channel 量化系数（形如 `(1,n)`），`per_token_scale` 为 per-token 量化系数（形如 `(1,m)`），`C` 为形如 `(m,n)` 的输出矩阵。

- A 矩阵全载说明：普通模板仅将 `L1TileShape::M * L1TileShape::K` 大小的 A 块载入 L1，而本样例将 `L1TileShape::M * K` 大小的 A 块一次性全量载入 L1 并常驻，计算期间 pingpong 搬入 B 矩阵，从而减少 A 矩阵的重复搬运、提升数据复用率。`N` 轴越大、单核越能多次复用 L1 中的 A 矩阵，性能收益越大；`N` 轴较小时无法复用 A 矩阵，性能较 00_basic_matmul 可能出现劣化。

- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

## 样例参数说明

| 参数             | 属性   | shape   | dtype  | 说明                                                         |
| ---------------- | ------ | ------- | ------ | ------------------------------------------------------------ |
| `A`              | Input  | `(m,k)` | `int8` | 左矩阵，layout支持`RowMajor` |
| `B`              | Input  | `(n,k)` | `int8` | 右矩阵，layout支持`RowMajor`（默认）和`ColumnMajor` |
| `scale`          | Input  | `(1,n)` | `fp32` / `fp16` / `bf16` | per-channel 量化系数 |
| `per_token_scale`| Input  | `(1,m)` | `fp32` / `fp16` / `bf16` | per-token 量化系数，数据类型与`scale`一致 |
| `C`              | Output | `(m,n)` | `fp16/bf16` | 矩阵乘反量化结果，`scale`与`C`数据类型映射关系为：`fp16→fp16`、`bf16→bf16`、`fp32→fp16/bf16` |

## 使用范围说明

支持范围：

- `k` <= `2048`，需满足约束 `L1TileShape::M * k * 1B + L1TileShape::K * L1TileShape::N * L1B_STAGES * 1B ≤ L1_SIZE`。

## 使用示例

### 命令行参数

```bash
44_quant_matmul_full_loadA_tla [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数       | 默认值 | 参数说明           |
| ---------- | ------ | ------------------ |
| `m`        | 无     | A 矩阵的 m 轴大小 |
| `n`        | 无     | B 矩阵的 n 轴大小 |
| `k`        | 无     | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0`    | 指定运行设备ID     |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 44_quant_matmul_full_loadA_tla
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./44_quant_matmul_full_loadA_tla 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
