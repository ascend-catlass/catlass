# W8A16Matmul Example Readme

## 功能说明

- 算子功能：完成 `float16` 激活与 `int8` 权重的反量化矩阵乘计算。
- 计算公式：

  $$
    \begin{aligned}
    \widehat{B} &= (B + \mathrm{deqZeroPoint}) \times \mathrm{deqScalar} \\
    C &= A \times \widehat{B}
    \end{aligned}
  $$

  其中 $A$ 和 $B$ 分别是形如 `(m,k)`、`(k,n)` 的输入矩阵，$C$ 是形如 `(m,n)` 的输出矩阵。

- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float16` | 左矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor` |
| `B` | Input | `(k,n)` | `int8` | 量化右矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor` |
| `deqScalar` | Input | 标量 | `float16` | 反量化缩放系数，默认 `1.5` |
| `deqZeroPoint` | Input | 标量 | `float16` | 反量化加法偏移，默认 `0.1` |
| `C` | Output | `(m,n)` | `float16` | 矩阵乘结果，当前 layout 为 `RowMajor` |

## 使用范围说明

适用于权重使用统一缩放系数和偏移量的反量化场景。

## 使用示例

### 命令行参数

```bash
30_w8a16_matmul [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 30_w8a16_matmul
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./30_w8a16_matmul 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
