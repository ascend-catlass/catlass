# GroupedMatmulSliceMPerTokenDequantMoe Example Readme

## 功能说明

- 算子功能：完成分组矩阵乘（Grouped Matmul）与 per-token/per-channel 反量化（dequant）融合计算。左矩阵沿 m 轴切分为 `g` 组，每组完成矩阵乘后，使用 per-channel 量化系数 `scale` 与 per-token 量化系数 `perToken` 进行反量化。
- 计算公式：

  $$
    \begin{aligned}
    C &= \mathrm{GroupedGEMM}(A,B) \\
    C_{i,j} &= \left(\Sigma_{k} A_{i,k}B_{g,k,j}\right) \cdot scale_{g,j} \cdot perToken_i
    \end{aligned}
  $$

  其中 `A` 是形如 `(m,k)` 的左矩阵，`B` 是形如 `(g,k,n)` 的右矩阵，`g` 为分组数量，m 轴具体切分的值由 `groupList` 确定；`scale` 为 per-channel 量化系数（形如 `(g,n)`），`perToken` 为 per-token 量化系数（形如 `(m,)`），`C` 为形如 `(m,n)` 的输出矩阵。

- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

## 样例参数说明

| 参数       | 属性   | shape    | dtype  | 说明                                                             |
| ---------- | ------ | -------- | ------ | ---------------------------------------------------------------- |
| `A`        | Input  | `(m,k)`  | `int8` | 左矩阵，layout 固定 `RowMajor`          |
| `B`        | Input  | `(g,k,n)`| `int8` | 右矩阵，layout 固定 `RowMajor`          |
| `groupList`| Input  | `(g,)`   | `int64`| m 轴切分的前缀和列表，终止偏移为 `m`，允许空分组 |
| `scale`    | Input  | `(g,n)`  | `fp32` | per-channel 量化系数，layout为`VectorLayout` |
| `perToken` | Input  | `(m,)`   | `fp32` | per-token 量化系数，layout为`VectorLayout` |
| `C`        | Output | `(m,n)`  | `fp16` | 矩阵乘反量化结果，layout 固定 `RowMajor`   |

## 使用范围说明

推荐场景：

- 分组数量 `g` 满足 `1 ≤ g ≤ 128` 且 `g ≤ m`。

## 使用示例

### 命令行参数

```bash
07_grouped_matmul_slice_m_per_token_dequant_moe [g] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数       | 默认值 | 参数说明                    |
| ---------- | ------ | --------------------------- |
| `g`        | 无     | 分组数量                  |
| `m`        | 无     | A 矩阵的 m 轴大小           |
| `n`        | 无     | B 矩阵的 n 轴大小           |
| `k`        | 无     | A/B 矩阵的 k 轴大小         |
| `deviceId` | `0`    | 指定运行设备ID              |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 07_grouped_matmul_slice_m_per_token_dequant_moe
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./07_grouped_matmul_slice_m_per_token_dequant_moe 128 512 1024 2048 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
