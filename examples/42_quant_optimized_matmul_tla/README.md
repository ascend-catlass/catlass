# QuantOptimizedMatmulTla Example Readme

## 功能说明

- 算子功能：完成量化矩阵乘（Quant Matmul）计算。左矩阵 `A` 与右矩阵 `B` 完成 int8 矩阵乘后，使用 per-channel 量化系数 `scale` 与 per-token 量化系数 `perToken` 进行反量化。
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \left(\Sigma_{k} A_{i,k}B_{k,j}\right) \cdot scale_j \cdot perToken_i
    \end{aligned}
  $$

  其中 `A` 是形如 `(m,k)` 的左矩阵，`B` 是形如 `(k,n)` 的右矩阵，`scale` 为 per-channel 量化系数（形如 `(1,n)`），`perToken` 为 per-token 量化系数（形如 `(1,m)`），`C` 为形如 `(m,n)` 的输出矩阵。

- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

## 样例参数说明

| 参数       | 属性   | shape   | dtype  | 说明                                       |
| ---------- | ------ | ------- | ------ | ------------------------------------------ |
| `A`        | Input  | `(m,k)` | `int8` | 左矩阵，layout支持`RowMajor`（默认）和`ColumnMajor` |
| `B`        | Input  | `(k,n)` | `int8` | 右矩阵，layout支持`RowMajor`和`ColumnMajor`（默认） |
| `scale`    | Input  | `(1,n)` | `fp32` | per-channel 量化系数，layout为`RowMajor` |
| `perToken` | Input  | `(1,m)` | `fp32` | per-token 量化系数，layout为`RowMajor` |
| `C`        | Output | `(m,n)` | `bf16` | 矩阵乘反量化结果，layout为`RowMajor` |

## 使用范围说明

支持范围：
- 当 A 为 `RowMajor` 且 B 为 `RowMajor` 时，需满足 `N < 65536`。
- 当 A 为 `ColumnMajor` 且 B 为 `RowMajor` 时，需满足 `M < 65536`、`N < 65536`。
- 当 A 为 `ColumnMajor` 且 B 为 `ColumnMajor` 时，需满足 `M < 65536`。

推荐范围：

- `M ≥ 256、N ≥ 256、256 < K ≤ 3072`

## 使用示例

### 命令行参数

```bash
42_quant_optimized_matmul_tla [m] [n] [k] [deviceId]
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
    bash scripts/build.sh 42_quant_optimized_matmul_tla
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./42_quant_optimized_matmul_tla 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
