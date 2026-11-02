# SingleCoreSplitkMatmul Example Readme

## 功能说明

- 算子功能：完成单核切 K（SingleCoreSplitk）优化的矩阵乘计算；当 L0C 不足以容纳放大后的 L1 tile 时，单核内沿 K 分轮累加。优化策略详见[单核切K策略说明](./34_single_splitk_matmul.md)。
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  其中$A$和$B$分别是形如`(m,k)`，`(k,n)`的输入矩阵，$C$是形如`(m,n)`的输出矩阵。

- 支持产品型号：Atlas A2/A3系列产品

## 样例参数说明

| 参数       | 属性 |shape|dtype| 说明                                                             |
| ---------- | ------ | --------|---------|----------------------------------------------- |
| `A`        | Input  | `(m,k)` | `float16/bfloat16/float32` | 左矩阵，支持转置；layout 可配置为 `RowMajor` 或 `ColumnMajor`，也可配合 Padding 使用 `zN`/`nZ` 等格式 |
| `B`        | Input  | `(k,n)` | `float16/bfloat16/float32` | 右矩阵，支持转置；layout 可配置为 `RowMajor` 或 `ColumnMajor`，数据类型与左矩阵一致 |
| `C`        | Output | `(m,n)` | `float16/bfloat16/float32` | 矩阵乘结果，layout为`RowMajor`，数据类型与左矩阵一致 |

本样例固定使用 A 为 `RowMajor`、B 为 `ColumnMajor` 的配置，命令行不提供转置开关。上述固定配置不限制算子接口的转置能力。

PyTorch 接口 `single_core_splitk_matmul` 通过 `transA` 和 `transB` 指定转置：`transA=false` 时 A 的输入形状为 `(m,k)`，`transA=true` 时为 `(k,m)`；`transB=false` 时 B 的输入形状为 `(k,n)`，`transB=true` 时为 `(n,k)`。输出形状均为 `(m,n)`。

## 使用范围说明

推荐范围：

- `M >= 2048, N >= 2048, K >= 4000` 的大尺寸场景（M、N 过小或 K 偏低会出现分核负载不均衡）。
- 实测大尺寸场景较 `00_basic_matmul` 约有 1.3~1.75 倍收益，且 K 越大收益越高。

## 使用示例

### 命令行参数

```bash
34_single_core_splitk_matmul [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 34_single_core_splitk_matmul
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./34_single_core_splitk_matmul 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
