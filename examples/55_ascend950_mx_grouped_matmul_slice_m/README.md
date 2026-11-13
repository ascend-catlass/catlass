# MxGroupedMatmulSliceM Example Readme

## 功能说明

- 算子功能：完成 MX 量化分组矩阵乘（Slice M），把左矩阵 `A` 沿 M 方向均分为 `G` 组，第 `g` 组 `A_g` 与第 `g` 个 `B` 矩阵做带 MX 缩放因子的矩阵乘，各组结果按行拼接为 `C`。
- 计算公式：

  $$
    \begin{aligned}
    C_g &= \tilde{A}_g \times \tilde{B}_g \\
    C_{i,j} &= \Sigma_{k}\left(a_{i,k}\cdot s^{A}_{i,\lfloor k/32 \rfloor}\right)\left(b_{k,j}\cdot s^{B}_{\lfloor k/32 \rfloor,j}\right)
    \end{aligned}
  $$

  其中 $\tilde{A}_g$、$\tilde{B}_g$ 分别是第 `g` 组 `A` 与对应 `B` 经 MX 缩放因子反量化后的矩阵，$a$、$b$ 为 FP8/FP4 量化值，$s^{A}$、$s^{B}$ 为每 32 个 K 元素共享的缩放因子；$C$ 是形如`(m,n)`的输出矩阵。

- 本样例包含 5 个可执行变体，分别由编译宏控制：
  - 默认变体：`GroupedMxMatmulSliceMTla` 配合恒等块 swizzle，并按每组 `M` 与 `N` 的相对大小动态选择 swizzle 方向（每组 `M > N` 时 M 优先，否则 N 优先）。
  - `_aswt` 变体：`GroupedMxMatmulSliceMAswtTla` 配合 `GemmGroupedAswtTailSplitSwizzle`，按 ASWT 窗口调度，并在末轮把尾部 tile 拆分到多核。
  - `_basem` 变体：在 ASWT 基础上按基础 M 切分，按每组实际 M 动态调整基础 tile，实现负载均衡。
  - `_preload` 变体：改用 `BlockMmadMxPreloadTla` 的延迟 MMAD 流水，当前 K 条带的载入与该条带之前的计算重叠。
  - `_all` 变体：ASWT、base-M 负载均衡与 preload 流水全部启用。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float8_e4m3_t` / `float8_e5m2_t` / `float4_e2m1x2_t` | 左矩阵，按 M 方向均分为 `G` 组，layout支持`RowMajor` |
| `B` | Input | `(G,k,n)` | `float8_e4m3_t` / `float8_e5m2_t` / `float4_e2m1x2_t` | 每组对应的右矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`，数据类型与左矩阵一致 |
| `MxScaleA` | Input | `(m,⌈k/32⌉)` | `float8_e8m0_t` | `A` 的 MX 缩放因子，K 方向每 32 个元素共享 |
| `MxScaleB` | Input | `(G,⌈k/32⌉,n)` | `float8_e8m0_t` | `B` 的 MX 缩放因子，K 方向每 32 个元素共享 |
| `GroupList` | Input | `(G,)` | `int64_t` | 每组在 M 方向的行数，样例按 `m / G` 均分生成 |
| `C` | Output | `(m,n)` | `bf16` | 各分组矩阵乘结果按行拼接后的结果，layout支持`RowMajor` |

## 使用范围说明

本样例在基础矩阵乘之上增加 MX 量化与 Slice M 分组能力，并提供 ASWT、base-M 负载均衡、preload 流水三种调度优化及组合变体。

支持范围：

- `m` 须能被 `problem_count` 整除，否则末尾余数行不参与分组计算。

## 代码组织

```text
55_ascend950_mx_grouped_matmul_slice_m
├── README.md
├── gen_data_compare.py  # 生成 MX 量化输入与 golden，调用可执行文件并输出精度比对结果
├── mx_grouped_matmul_slice_m.cpp  # 样例主程序，含默认/ASWT/base-M/preload/全开 5 个编译变体
└── CMakeLists.txt
```

## 使用示例

### 命令行参数

执行样例命令如下：

```bash
# 默认调度变体
55_ascend950_mx_grouped_matmul_slice_m [problem_count] [m] [n] [k] [trans_b] [quant_type] [deviceId]
# ASWT 调度变体
55_ascend950_mx_grouped_matmul_slice_m_aswt [problem_count] [m] [n] [k] [trans_b] [quant_type] [deviceId]
# ASWT + base-M 负载均衡变体
55_ascend950_mx_grouped_matmul_slice_m_basem [problem_count] [m] [n] [k] [trans_b] [quant_type] [deviceId]
# preload 流水变体
55_ascend950_mx_grouped_matmul_slice_m_preload [problem_count] [m] [n] [k] [trans_b] [quant_type] [deviceId]
# 全开变体
55_ascend950_mx_grouped_matmul_slice_m_all [problem_count] [m] [n] [k] [trans_b] [quant_type] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `problem_count` | 无 | 分组个数 `G` |
| `m` | 无 | 左矩阵的 m 轴大小 |
| `n` | 无 | 右矩阵的 n 轴大小 |
| `k` | 无 | 左右矩阵的 k 轴大小 |
| `trans_b` | 无 | `B` 是否转置，`0` 表示不转置（RowMajor），`1` 表示转置（ColumnMajor） |
| `quant_type` | 无 | 量化类型，可选 `float8_e4m3fn`、`float8_e5m2`、`float4_e2m1fn_x2` |
| `deviceId` | `0` | 指定运行设备ID |

生成测试数据命令行如下：

```bash
python3 gen_data_compare.py [problem_count] [m] [n] [k] [trans_b] [quant_type] [deviceId] [--bin BIN]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `problem_count` | 无 | 分组个数 `G` |
| `m` | 无 | 左矩阵的 m 轴大小 |
| `n` | 无 | 右矩阵的 n 轴大小 |
| `k` | 无 | 左右矩阵的 k 轴大小 |
| `trans_b` | 无 | `B` 是否转置，`0` 表示不转置（RowMajor），`1` 表示转置（ColumnMajor） |
| `quant_type` | `float8_e5m2` | 量化类型，可选 `float8_e4m3fn`、`float8_e5m2`、`float4_e2m1fn_x2` |
| `deviceId` | `0` | 指定运行设备ID |
| `--bin` | 主程序名 | 指定待测可执行文件，用于切换 `_aswt` / `_basem` / `_preload` / `_all` 等变体 |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件（Ascend 950 需指定 `-DCATLASS_ARCH=3510`）。

    ```bash
    # 默认调度变体
    bash scripts/build.sh 55_ascend950_mx_grouped_matmul_slice_m -DCATLASS_ARCH=3510
    # ASWT 调度变体
    bash scripts/build.sh 55_ascend950_mx_grouped_matmul_slice_m_aswt -DCATLASS_ARCH=3510
    # ASWT + base-M 负载均衡变体
    bash scripts/build.sh 55_ascend950_mx_grouped_matmul_slice_m_basem -DCATLASS_ARCH=3510
    # preload 流水变体
    bash scripts/build.sh 55_ascend950_mx_grouped_matmul_slice_m_preload -DCATLASS_ARCH=3510
    # 全开变体
    bash scripts/build.sh 55_ascend950_mx_grouped_matmul_slice_m_all -DCATLASS_ARCH=3510
    ```

2. 进入样例目录，生成测试数据、执行算子并完成精度比对（输入数据与结果写入样例目录下的 `data/`）。

    ```bash
    cd examples/55_ascend950_mx_grouped_matmul_slice_m
    # 默认调度变体
    python3 gen_data_compare.py 2 128 256 1024 0 float8_e5m2 0
    # ASWT 调度变体（--bin 指定待测可执行文件）
    python3 gen_data_compare.py 2 128 256 1024 0 float8_e5m2 0 \
        --bin 55_ascend950_mx_grouped_matmul_slice_m_aswt
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    ------ 开始比较 ------
    比较结果：Compare success
    ```

## 补充说明

- 样例未启用 Bias（`ElementBias` 为 `void`）；`C` 按 `bf16` 计算后转为 `fp32` 写入 `data/result.bin`，再做精度比对。
- 编译与执行环境准备可参考 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)。
- MX 量化矩阵乘的详细特征，可参考 [53_ascend950_fp8_mx_matmul](../53_ascend950_fp8_mx_matmul/README.md) 与 [54_ascend950_fp4_mx_matmul](../54_ascend950_fp4_mx_matmul/README.md)。
- 也可使用下方的快速测试脚本，一次覆盖 3 种量化类型与 2 种 `trans_b` 取值：

```bash
#!/bin/bash
# Usage: bash test_55_quick.sh [device_id]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_ID="${1:-0}"

G=1 M=588 K=1030 N=988
QUANT_TYPES=("float8_e4m3fn" "float8_e5m2" "float4_e2m1fn_x2")
TRANSB_VALS=(0 1)

PASS=0
FAIL=0
TOTAL=$(( ${#QUANT_TYPES[@]} * ${#TRANSB_VALS[@]} ))

echo "=============================================="
echo "  Quick test: G=$G M=$M K=$K N=$N device=$DEVICE_ID  ($TOTAL cases, transA fixed=0)"
echo "=============================================="

for QT in "${QUANT_TYPES[@]}"; do
  for TB in "${TRANSB_VALS[@]}"; do
    IDX=$((PASS+FAIL+1))
    LOG="/tmp/test_55_${QT}_ta0_tb${TB}.log"
    printf "[%2d/%2d] QT=%-18s TA=0 TB=%s  " "$IDX" "$TOTAL" "$QT" "$TB"

    python3 "${SCRIPT_DIR}/gen_data_compare.py" \
         "$G" "$M" "$N" "$K" "$TB" "$QT" "$DEVICE_ID" \
         > "$LOG" 2>&1; RC=$?

    if [ $RC -ne 0 ]; then
      echo "CRASH (exit=$RC)  log: $LOG"
      FAIL=$((FAIL+1))
      tail -5 "$LOG"
    elif grep -q "Compare success" "$LOG"; then
      echo "PASS"
      PASS=$((PASS+1))
    else
      echo "FAIL (精度不达标)  log: $LOG"
      FAIL=$((FAIL+1))
      grep "Compare\|result\|npu\|upgrade\|mare\|mere\|rmse" "$LOG" | tail -5
    fi
  done
done

echo "=============================================="
echo "  Result: $PASS PASS, $FAIL FAIL"
echo "=============================================="
[ $FAIL -eq 0 ]
```
