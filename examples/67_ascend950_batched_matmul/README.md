# BatchedMatmul Example Readme

## 功能说明

- 算子功能：基于 TLA 的批量矩阵乘（BatchedMatmul）计算。
- 计算公式：

  $$
    \begin{aligned}
    C[b] &= A[b] \times B[b] \\
    C[b]_{i,j} &= \Sigma_{k} A[b]_{i,k}B[b]_{k,j}
    \end{aligned}
  $$

  其中 $A$ 和 $B$ 分别是形如 `(batch,m,k)`、`(batch,k,n)` 的批量输入矩阵，$C$ 是形如 `(batch,m,n)` 的批量输出矩阵，`batch` 为批量数。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(batch,m,k)` | `float32` | 左矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor` |
| `B` | Input | `(batch,k,n)` | `float32` | 右矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor` |
| `C` | Output | `(batch,m,n)` | `float32` | 矩阵乘结果，当前 layout 为 `RowMajor` |

样例默认编译为 `float32`；修改源码中的数据类型与 layout 模板参数后可切换（泛化测试已覆盖 `float16` 输入），调整数据类型和布局后需重新编译。

## 使用范围说明

本样例基于 `MmadPingpong` DispatchPolicy 实现，batch 维连续存储（相邻矩阵步长分别为 `m*k`、`k*n`、`m*n`，不支持自定义 batch 步长），无 Padding/Preload/切K 等优化。

支持范围：

- `batch`、`m`、`n`、`k` 均为正整数，`batch = 1` 时等价于单矩阵乘。
- `A`、`B` 单个张量的字节数不超过 `2^31 - 1`（`int32` 偏移上限），即 `batch*m*k*sizeof(dtype)` 与 `batch*k*n*sizeof(dtype)` 均满足该限制。

推荐范围：

- `batch` 取 `1~16384`，`m ≥ 1`，`n ≥ 16`，`k` 取 `16~4096`。
- `A`、`B`、`C` 均为 `float32` 时，`k` 建议不超过 4096，以避免累计绝对误差超过校验阈值。
- `m`、`n` 建议为 256 的倍数，`k` 建议为 128 的倍数，与 L1 分块 `256*256*128` 对齐；`m`、`n` 非 256 整倍数时尾块仍可正确计算，性能低于对齐场景。
- 并行块数为 `batch * ceil(m/256) * ceil(n/256)`，小于实际 AI Core 核数时存在空核。batch 维已提供主要并行度，无需为对齐核数压缩单个矩阵乘的 `m`、`n`。
- 超出上述数据量上限时，依次限制 `k`、按较大轴折半收缩 `m`/`n`、最后收缩 `batch`，以尽量保留 `m`、`n`、`k` 的形状语义。

以上形状范围与数据量上限用于控制单个用例的数据量，来自用例生成时的规模裁剪，不是硬件维度上限或 kernel 的硬性约束。

### DispatchPolicy 模板参数

本样例使用的 `MmadPingpong` 支持以下模板参数：

| 模板参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `ArchTag` | 无 | 指定架构型号，本样例为 `Arch::Ascend950` |
| `enableUnitFlag` | `false` | 是否开启 UnitFlag；开启 L0C 多缓冲（`l0CStages > 1`）时必须设置为 `false` |
| `useHF32` | `false` | 是否开启 HF32，仅 `float` 类型支持 |
| `l0CStages` | `1` | 指定 L0C 的缓冲区数量，设置为 2 即可开启 L0C 双缓冲 |
| `enableL1Resident` | `false` | 是否开启 L1 常驻 |
| `l1AStages` | `2` | 指定 L1A 的缓冲区数量 |
| `l1BStages` | `2` | 指定 L1B 的缓冲区数量 |
| `l0AStages` | `2` | 指定 L0A 的缓冲区数量 |
| `l0BStages` | `2` | 指定 L0B 的缓冲区数量 |

本样例编译配置为 `enableUnitFlag = true`，其余参数保持默认值。调整缓冲区数量时，需满足分块大小与缓冲区数量的乘积不超过对应 L0A/L0B/L0C 存储空间的约束。

## 使用示例

### 命令行参数

```bash
67_ascend950_batched_matmul [batch] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | 批量矩阵的 batch 轴大小 |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 67_ascend950_batched_matmul -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./67_ascend950_batched_matmul 5 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
