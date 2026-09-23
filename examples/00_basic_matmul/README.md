# BasicMatmul Example Readme

## 功能说明

- 算子功能：完成基础矩阵乘计算
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  其中$A$和$B$分别是形如`(m,k)`，`(k,n)`的输入矩阵，$C$是形如`(m,n)`的输出矩阵。
- 支持产品型号：Atlas A2/A3 训练/推理系列产品

## 样例参数说明

| 参数       | 属性 |shape|dtype| 说明                                                             |
| ---------- | ------ | --------|---------|----------------------------------------------- |
| `A`        | Input  | `(m,k)` | `float16/bfloat16` | 左矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`  |
| `B`        | Input  | `(k,n)` | `float16/bfloat16` | 右矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`，数据类型与左矩阵一致 |
| `C`        | Output | `(m,n)` | `float16/bfloat16` | 矩阵乘结果，layout仅支持`RowMajor`，数据类型与左矩阵一致 |

## 使用范围说明

本样例无 Padding/Preload/切K 等优化，为各优化样例的公共基线。

推荐范围：
- `m ≥ 256、n ≥ 256、256 < k ≤ 3072`
- K轴、N轴对齐512B

### 命令行参数

```bash
00_basic_matmul [m] [n] [k] [deviceId]
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
    bash scripts/build.sh 00_basic_matmul
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./00_basic_matmul 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
