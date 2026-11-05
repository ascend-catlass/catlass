# BasicSyrk Example Readme

## 功能说明

- 算子功能：计算输入矩阵与其转置的乘积，得到对称矩阵。
- 计算公式：

  $$
    \begin{aligned}
    Y &= X \times X^T \\
    Y_{i,j} &= \sum_k X_{i,k}X_{j,k}
    \end{aligned}
  $$

  其中 $X$ 是形如 `(M, K)` 的输入矩阵，$Y$ 是形如 `(M, M)` 的输出矩阵，满足 $Y=Y^T$。

- 支持产品型号：Ascend 950PR&950DT系列产品（`CATLASS_ARCH=3510`）。

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `X` | Input | `(M, K)` | `bfloat16` | 连续矩阵，layout 为 `RowMajor` |
| `Y` | Output | `(M, M)` | `bfloat16` | 对称矩阵乘结果，layout 为 `RowMajor`，数据类型与输入一致 |

当前 `basic_syrk_tla.cpp` 中 `ElementX`、`ElementY` 固定为 `bfloat16_t`，主机输入、输出也使用 bf16；命令行不提供 dtype 或 layout 切换。设备计算使用 fp32 累加；CPU golden 基于同一份量化后的输入使用 fp64 累加，完成后转换为 fp32，沿用公共 `CompareData` 的精度阈值，避免长 K 下参考计算的逐项 fp32 累加误差。

## 使用范围说明

本样例按 `256*256` 输出块计算下三角，非对角块通过 nz2nd/nz2dn 双写补全对称位置，对角块只写回一次。L1/L0 分块分别为 `(256,256,128)`、`(256,256,64)`，使用双缓冲和 `SyrkBlockScheduler` 调度。L0C 双写通过 `M_FIX` / `FIX_M` 事件同步，关闭 unitFlag，无需 workspace。

使用范围：

- `1 ≤ M ≤ 8192`、`1 ≤ K ≤ 65536`，且 `MK ≤ 8192² = 67108864`。M、K 可以独立取值，支持尾块，不要求按分块尺寸对齐。
- 内部 problem shape 为 `(M,M,K)`。`BasicSyrkTla::CanImplement` 检查正尺寸、上述上限、`M=N` 以及输出块 M/N 相等；样例在初始化设备和分配内存前检查返回值，不满足条件时返回非零退出码。
- `X^T` 是同一份 X 存储的 ColumnMajor 视图，无需分配或传入转置矩阵；Y 使用连续 RowMajor 存储。
- 启动核数自动取设备 Cube Core 数量与输出块网格大小中的较小值，命令行不提供核数参数。
- 仅计算 `Y = X @ X.T`，不支持 batch、alpha/beta 或输入 C。命令行仅接受 `m k [deviceId]`，不接受公共 `SyrkOptions` 的扩展参数形式。
- `deviceId` 必须是当前环境可用的设备编号，设备初始化由 ACL 检查。

bf16 输入、输出分别占 `2MK`、`2M²` 字节。输出随 M 平方增长，因此限制 `M ≤ 8192`；另限制 `MK ≤ 8192²`，使单份输入、输出各不超过 128 MiB，为 10 GiB 主机内存预算下的 golden 计算和比较留出空间。例如允许 `(1024,65536)`，拒绝 `(8192,65536)`。这是交付范围，不是硬件理论极限或进程内存的硬保证。

推荐范围：

- 在上述范围内，推荐 M 为 256 的整数倍、K 为 128 的整数倍，以匹配当前分块并避免尾块处理。
- 性能测试推荐输出块数量足以覆盖设备 Cube Core；小尺寸可能无法充分利用多核并行。

## 使用示例

### 命令行参数

```bash
./82_ascend950_basic_syrk m k [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无，必填 | 输入 X 的行数及输出 Y 的边长 |
| `k` | 无，必填 | 输入 X 的列数，即归约维度 |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 加载 CANN 环境，进入项目根目录并编译样例。

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    bash scripts/build.sh -DCATLASS_ARCH=3510 82_ascend950_basic_syrk
    ```

2. 切换到可执行文件目录，执行样例。

    ```bash
    cd output/bin
    ./82_ascend950_basic_syrk 1024 1024 0
    ```

    尾块用例可使用 `769 129 0`，长 K 用例可使用 `1024 65536 0`。

3. 执行结果如下，说明与 CPU golden 的精度比较通过，退出码为 0；比较失败时返回非零退出码。

    ```text
    Compare success.
    ```
