# BasicMatmulPreloadZN Example Readme

## 代码组织

```text
├── 21_basic_matmul_preload_zN
│   ├── CMakeLists.txt   # CMake 编译文件
│   ├── README.md
│   └── basic_matmul_preload_zN.cpp # 主文件
```

## 功能介绍

本样例基于 `00_basic_matmul`，使用 `MmadAtlasA2Preload` 调度策略。当前配置中，A 和 C 使用 `RowMajor` 布局，B 使用 `layout::zN` 分形布局。转置布局可将样例的 `LayoutB` 从 `layout::zN` 改为 `layout::nZ`。修改样例源码后需重新编译。

## 参数说明

| 参数 | 形状 | 数据类型 | 布局 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| A | `(m, k)` | `float16` | `RowMajor` | 输入矩阵 |
| B | `(k, n)` | `float16` | `zN` 分形布局 | 输入矩阵；当前默认布局 |
| C | `(m, n)` | `float16` | `RowMajor` | 输出矩阵 |

以上参数描述当前样例配置；修改源码中的数据类型或布局后，需重新编译并同步文档。

## 使用范围与约束

- 当前 B 输入使用 `zN` 分形布局，逻辑维度 `k` 须为 16 的倍数，`n` 须为 `32 / sizeof(ElementB)` 的倍数（当前为 16）；不满足这些条件的输入不受当前样例支持。
- `M ≥ 256、N ≥ 256、256 < K ≤ 3072` 是性能推荐 MNK 范围。
- 当 A、B 均为 fp32 时，推荐 `K ≤ 6144`。这些推荐范围不是所有合法输入的尺寸边界。

## 使用示例

- 获取代码后编译对应算子的可执行文件，详见 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)。
- 执行算子：

```bash
# 编译指定用例
bash scripts/build.sh 21_basic_matmul_preload_zN
cd /output/bin
# 可执行文件名 | 矩阵 m 轴 | n 轴 | k 轴 | Device ID
# Device ID 可选，默认值为 0
./21_basic_matmul_preload_zN 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 模板推荐场景

本样例与 `06_optimized_matmul` 使用相同的 `MmadAtlasA2Preload` 调度策略和 tile。区别在于本样例的 B 输入使用 `zN` 分形布局（NZ 布局），GM 到 L1 读取 B 时无需执行 ND2NZ 随路转换。推荐 MNK 范围为 `M ≥ 256、N ≥ 256、256 < K ≤ 3072`；当 A、B 均为 fp32 时，推荐 `K ≤ 6144`。这些推荐范围用于性能选择，不代表硬尺寸边界。
