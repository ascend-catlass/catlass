# SparseMatmulTla Example Readme

## 功能说明

- 算子功能：完成 4:2 结构化稀疏矩阵乘。左矩阵 `A` 为稠密 `int8` 矩阵；右矩阵 `B` 为每 4 个连续元素中保留 2 个非零元素的稀疏矩阵，以压缩矩阵和索引矩阵表示；输出为 `int32`。
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  其中 $A$ 是形如 `(m, k)` 的稠密矩阵，$B$ 是形如 `(k, n)` 的稀疏矩阵，由压缩后的 `B`（`(k/2, n)`）和索引矩阵还原，$C$ 是形如 `(m, n)` 的输出矩阵。

- 支持产品型号：Atlas A2系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m, k)` | `int8` | 左矩阵，layout 为 `RowMajor` |
| `B` | Input | `(k/2, n)` | `int8` | 右矩阵的 4:2 稀疏压缩存储，layout 为 `ColumnMajor`（transB），每个 4 元素块仅保存 2 个非零值 |
| `index` | Input | `(n, k/8)` | `uint8` | 4:2 稀疏索引矩阵，按 NZ 分形布局存储，标识 `B` 中非零元素的位置 |
| `C` | Output | `(m, n)` | `int32` | 矩阵乘结果，layout 为 `RowMajor` |

## 使用范围说明

本样例固定 batch=1，启动时调用 `CanImplement` 校验问题形状，不满足时直接报错退出。

约束：

- `m`、`n`、`k` 均须大于 0。
- `k % 64 == 0`，同时满足 4:2 稀疏对 `k` 的 8 元素对齐要求。
- `n % 16 == 0`。
- `k ≤ 65536`。
- 编译期 Tile 配置 `<128, 256, 128>`、`<128, 256, 64>` 须满足 L0/L1 容量，当前配置满足。

本样例用于功能验证，不提供性能推荐范围。

## 使用示例

### 命令行参数

```bash
41_sparse_matmul_tla [m] [n] [k] [deviceId]
```

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 在工程目录下生成测试数据。

    ```bash
    python3 examples/41_sparse_matmul_tla/sparse_gen_data.py 160 320 64
    ```

    `input/` 下生成 `x1_gm.bin`、`x2_gm.bin`、`index_gm.bin`，`output/` 下生成 `golden.bin`。

2. 编译样例。

    ```bash
    bash scripts/build.sh 41_sparse_matmul_tla
    ```

3. 进入 `output/bin` 执行样例。

    ```bash
    cd output/bin
    ./41_sparse_matmul_tla 160 320 64 0
    ```

4. 显示如下结果表示样例执行成功，精度比对通过。

    ```text
    Compare success.
    ```
