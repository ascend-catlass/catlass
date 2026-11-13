# A8W4MxMatmul Example Readme

## 功能说明

- 算子功能：完成 Ascend 950 上的 A8W4 混合量化矩阵乘计算，左矩阵 A 为 FP8，右矩阵权重 B 为 FP4，两者经 MX 缩放因子反量化后在 Cube 上完成乘加，输出为 FP32，未启用 Bias。
- 计算公式：

  $$
    \begin{aligned}
    C &= \tilde{A} \times \tilde{B} \\
    C_{i,j} &= \Sigma_{k}\left(2^{s^{A}_{i,\lfloor k/32 \rfloor}}\cdot a_{i,k}\right)\left(2^{s^{B}_{\lfloor k/32 \rfloor,j}}\cdot b_{k,j}\right)
    \end{aligned}
  $$

  其中 `A`、`B` 分别是形如 `(m,k)`、`(k,n)` 的输入矩阵，`C` 是形如 `(m,n)` 的输出矩阵；$a$、$b$ 为 FP8/FP4 量化值，$s^{A}$、$s^{B}$ 为沿 K 轴每 32 个元素共享的 MX 缩放因子。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m, k)` | `float8_e4m3_t` | 左矩阵，layout为`RowMajor` |
| `MxScaleA` | Input | `(m, ⌈k/32⌉)` | `float8_e8m0_t` | `A` 的 MX 缩放因子，K 方向每 32 个元素共享一组，存储时按 2 对齐补齐 |
| `B` | Input | `(k, n)` | `float4_e2m1x2_t` | 右矩阵权重，layout为`ColumnMajor` |
| `MxScaleB` | Input | `(⌈k/32⌉, n)` | `float8_e8m0_t` | `B` 的 MX 缩放因子，K 方向每 32 个元素共享一组，存储时按 2 对齐补齐 |
| `C` | Output | `(m, n)` | `fp32` | 混合量化矩阵乘结果，layout支持`RowMajor` |

## 使用范围说明

本样例在基础矩阵乘之上引入 A8W4 混合量化，经 MX 缩放反量化后在 Cube 上完成乘加。

推荐范围：

- `m、n 为 128 的倍数`
- `k 为 64 的倍数`
- `k <= 2048`

## 代码组织

```text
59_ascend950_a8w4_mx_matmul
├── README.md
├── CMakeLists.txt
├── gen_data.py  # 生成测试数据（input/ 与 golden/）
└── a8w4_mx.cpp  # 样例主程序
```

## 使用示例

### 命令行参数

执行样例命令如下：

```bash
59_ascend950_a8w4_mx_matmul [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

生成测试数据命令行如下：

```bash
python3 examples/59_ascend950_a8w4_mx_matmul/gen_data.py [m] [n] [k] 0 1
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | 矩阵 m 轴大小 |
| `n` | 无 | 矩阵 n 轴大小 |
| `k` | 无 | 矩阵 k 轴大小 |

本样例仅支持 `trans_a=0`、`trans_b=1`（A 为 `RowMajor`、B 为 `ColumnMajor`），因此 `gen_data.py` 末尾两项固定为 `0 1`。

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。本用例为 Ascend 950 算子，编译时需加 `-DCATLASS_ARCH=3510`。

    ```bash
    bash scripts/build.sh 59_ascend950_a8w4_mx_matmul -DCATLASS_ARCH=3510
    ```

2. 在项目根目录下生成测试样例数据。测试样例数据随机生成，尺寸从命令行输入，本样例仅支持 `trans_a=0`、`trans_b=1`。

    ```bash
    python3 examples/59_ascend950_a8w4_mx_matmul/gen_data.py 128 128 128 0 1
    ```

3. 在项目根目录下执行算子样例程序。样例按相对路径 `./examples/...` 读取测试数据，请勿切换到 `output/bin` 目录下执行。

    ```bash
    ./output/bin/59_ascend950_a8w4_mx_matmul 128 128 128 0
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```

## 补充说明

本样例完成 MX 量化矩阵乘：$C = (2^{S_A} \odot A) \times (2^{S_B} \odot B)$，其中右矩阵权重 B 为 `float4_e2m1x2_t`，反量化为 `float8_e4m3_t` 后参与 Cube 计算，MX 缩放因子为 `float8_e8m0_t`。

本样例固定左矩阵 A 为 `RowMajor`、右矩阵权重 B 为 `ColumnMajor`，对应 `trans_a=0`、`trans_b=1`。

`MxScaleA`、`MxScaleB` 的逻辑 shape 分别为 `(m, ⌈k/32⌉)`、`(⌈k/32⌉, n)`，表示沿 K 轴每 32 个元素共用一个 scale；实际存储时把 scale 分组数补齐到偶数，相邻 2 个 scale 打包为一组，物理存储维度展开为三维（`⌈k/32⌉` 补偶后等于 `2 × ⌈k/64⌉`）。本样例的存储 shape 如下：

- A 为 `RowMajor`：`MxScaleA` 存储 shape 为 `(m, ⌈k/64⌉, 2)`
- B 为 `ColumnMajor`：`MxScaleB` 存储 shape 为 `(n, ⌈k/64⌉, 2)`
