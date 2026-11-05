# MxGroupedMatmulSliceM Example Readme

## 功能介绍

- 演示 Ascend 950 上的 **MX 分组矩阵乘（Slice M）**：左矩阵 A 按 M 方向均匀分组，每组 A 与对应的 B 矩阵做 MX 缩放矩阵乘；当前样例 C 类型默认使用 BF16，精度比较时结果会转为 FP32 写入 `result.bin` 并按 FP32 读取。
- 本示例同时支持 **MX FP8**（`float8_e4m3_t` / `float8_e5m2_t`）与 **MX FP4**（`float4_e2m1x2_t`），缩放因子统一为 `float8_e8m0_t`。未启用 Bias（`ElementBias` 为 `void`）。
- 支持通过命令行参数指定 `transB`，编译期模板参数决定 B矩阵的 Layout（RowMajor = 不转置，ColumnMajor = 转置）。
- 默认布局为 A `RowMajor`、B `RowMajor`、C `RowMajor`（对应 `transA=0, transB=0`）。

## 代码组织

```text
├── 55_ascend950_mx_grouped_matmul_slice_m
│   ├── CMakeLists.txt              # CMake 编译配置
│   ├── README.md
│   ├── gen_data_compare.py         # 数据生成 + 精度比对脚本
│   └── mx_grouped_matmul_slice_m.cpp  # 主程序
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)，本用例为 Ascend 950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`

```bash
# 编译指定用例
bash scripts/build.sh 55_ascend950_mx_grouped_matmul_slice_m -DCATLASS_ARCH=3510

# 生成测试数据并运行（在 examples/55_ascend950_mx_grouped_matmul_slice_m/data 下生成输入与结果）

# python3 examples/55_ascend950_mx_grouped_matmul_slice_m/gen_data_compare.py \
#   <group_count> <m> <n> <k> <trans_b> <quant_type> <device_id>
# 输入参数说明：
#   group_count : 分组个数 G
#   m, n, k     : 矩阵维度
#   trans_b     : B 矩阵是否转置，0=不转置，1=转置
#   quant_type  : 量化类型（float8_e4m3fn / float8_e5m2 / float4_e2m1fn_x2）
#   device_id   : 设备 ID

# 示例：
python3 examples/55_ascend950_mx_grouped_matmul_slice_m/gen_data_compare.py \
    2 128 256 1024 0 float8_e5m2 0
```

执行结果如下，说明精度（使用双精度标准）比对成功：

```text
------ 开始比较 ------
比较结果：Compare success.
```

也可使用下方的快速测试脚本：

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

## 使用说明

关于Mx量化矩阵乘的详细特征详见[53_ascend950_fp8_mx_matmul](../53_ascend950_fp8_mx_matmul/README.md)和[54_ascend950_fp4_mx_matmul](../54_ascend950_fp4_mx_matmul/README.md)中说明文档的相关内容。
