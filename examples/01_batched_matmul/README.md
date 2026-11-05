# BatchedMatmul Example Readme

## 功能说明

- 算子功能：完成批量矩阵乘（Batched Matmul）计算。
- 计算公式：

  $$
    \begin{aligned}
    C_b &= A_b \times B_b \\
    (C_b)_{i,j} &= \Sigma_{k} (A_b)_{i,k}(B_b)_{k,j}
    \end{aligned}
  $$

  其中 $b$ 为 batch 下标，$A_b$、$B_b$ 分别是形如 `(m,k)`、`(k,n)` 的输入矩阵，$C_b$ 是形如 `(m,n)` 的输出矩阵；各 batch 沿连续地址依次存放。

- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

## 样例参数说明

| 参数       | 属性 |shape|dtype| 说明                                                             |
| ---------- | ------ | --------|---------|----------------------------------------------- |
| `A`        | Input  | `(batch,m,k)` | `float16/bfloat16` | 左矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`  |
| `B`        | Input  | `(batch,k,n)` | `float16/bfloat16` | 右矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`，数据类型与左矩阵一致 |
| `C`        | Output | `(batch,m,n)` | `float16/bfloat16` | 批量矩阵乘结果，layout仅支持`RowMajor`，数据类型与左矩阵一致 |

## 使用范围说明

推荐范围（按单 batch 的 MNK 理解，与基础矩阵乘样例一致）：

- `m >= 256, n >= 256, 256 < k <= 3072`。
- K轴、N轴对齐512B。

## 使用示例

### 命令行参数

```bash
01_batched_matmul [batch] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | batch 轴大小，即批量矩阵乘的批次数 |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 01_batched_matmul
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./01_batched_matmul 5 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
