# MxMatmul Example Readme

## 功能说明

- 算子功能：完成微缩放（MX Scale）矩阵乘计算，k 轴上每 32 个元素共享一个 E8M0 的 2 的幂缩放因子
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} (A_{i,k} s^{A}_{i,\lfloor k/32 \rfloor})(B_{k,j} s^{B}_{\lfloor k/32 \rfloor,j})
    \end{aligned}
  $$

  其中$A$和$B$分别是形如`(m,k)`，`(k,n)`的输入矩阵，$C$是形如`(m,n)`的输出矩阵，$s^{A}$和$s^{B}$分别是形如`(m,ceil(k/32))`，`(ceil(k/32),n)`的 E8M0 分块缩放因子矩阵。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数       | 属性 |shape|dtype| 说明                                                             |
| ---------- | ------ | --------|---------|----------------------------------------------- |
| `A`        | Input  | `(m,k)` | `f8e4m3fn/f8e5m2/f4e2m1/f4e1m2` | 左矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`，数据类型为 fp4 时固定为`RowMajor` |
| `scale_A`  | Input  | `(m,ceil(k/32))` | `f8e8m0` | 左矩阵的分块缩放因子，k 轴上每 32 个元素对应一个，layout与左矩阵一致 |
| `B`        | Input  | `(k,n)` | `f8e4m3fn/f8e5m2/f4e2m1/f4e1m2` | 右矩阵，layout支持`ColumnMajor`（默认）和`RowMajor`，数据类型为 fp4 时固定为`ColumnMajor`，元素位宽与左矩阵一致 |
| `scale_B`  | Input  | `(ceil(k/32),n)` | `f8e8m0` | 右矩阵的分块缩放因子，k 轴上每 32 个元素对应一个，layout与右矩阵一致 |
| `C`        | Output | `(m,n)` | `f32` | 矩阵乘结果，layout仅支持`RowMajor` |

## 使用范围说明

左右矩阵的元素位宽必须一致，fp8 矩阵不能与 fp4 矩阵相乘；位宽一致时编码可以不同，任意 fp8 编码之间、任意 fp4 编码之间均可组合。

推荐范围：

- `m ≥ 256、n ≥ 256、k ≥ 256`
- k 轴为 32 的整数倍（且fp4 的 `k` 会向上取整为偶数），保证缩放分块不被填充。
- 为避免误差累加和计算超时，建议矩阵的 M/N/K 轴不超过 122880，且 GM 总分配不超过 10 GiB：fp8 按 `M*K + K*N + 4*M*N` 字节估算，fp4 按 `(M*K + K*N)/2 + 4*M*N` 字节估算。

## 使用示例

### 命令行参数

```bash
mx_matmul.py [--m m] [--n n] [--k k] [--layout-a layoutA] [--layout-b layoutB] [--case case] [--device deviceId] [--block-num blockNum]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | `256` | A 矩阵的 m 轴大小 |
| `n` | `512` | B 矩阵的 n 轴大小 |
| `k` | `1024` | A/B 矩阵的 k 轴大小 |
| `layoutA` | `row` | A 矩阵的 layout，取值为`row`或`col` |
| `layoutB` | `col` | B 矩阵的 layout，取值为`row`或`col` |
| `case` | `f8e4m3fn,f8e4m3fn` | A/B 矩阵的数据类型，重复该参数可在一个进程内运行多个用例 |
| `deviceId` | `0` | 指定运行设备ID |
| `blockNum` | `-1` | 使用的 AIC 核数，`-1`表示自动取满 |

### 执行示例

1. 进入项目根目录，参考[快速上手](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)准备样例代码的运行环境。

2. 在项目根目录下执行算子样例程序。

    ```bash
    python examples/python/tla_dsl/tla/kernel/matmul/mx_mmad/mx_matmul.py --m 512 --n 512 --k 512 --device 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    SUMMARY passed=1/1
    ```
