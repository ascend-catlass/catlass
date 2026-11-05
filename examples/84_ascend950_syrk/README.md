# Syrk Example Readme

## 代码组织

```text
├── 84_ascend950_syrk
│   ├── CMakeLists.txt   # CMake 编译文件
│   ├── README.md
│   └── syrk_tla.cpp     # 主文件（host 数据生成、kernel 调度、精度校验）
```

## 使用示例

1. 编译样例（Ascend 950 需指定 `CATLASS_ARCH=3510`），可参考[快速入门](../../docs/zh/1_Practice/01_quick_start.md#编译执行)：

    ```bash
    bash scripts/build.sh -DCATLASS_ARCH=3510 84_ascend950_syrk
    ```

2. 切换到可执行文件目录并运行。测试数据随机生成，尺寸由命令行传入：

    ```bash
    cd output/bin
    # 可执行文件名 | m | k | batch | alpha | beta | deviceId(可选,默认0)
    ./84_ascend950_syrk 1024 1024 1 1.0 1.0 0
    ```

    - `1024`：矩阵 m 轴（$X$ 行数 / $Y$ 边长）
    - `1024`：k 轴（$X$ 列数）
    - `1`：batch 数
    - `1.0`：alpha 系数
    - `1.0`：beta 系数
    - `0`：Device ID

3. 执行成功输出：

    ```text
    Compare success.
    ```

## 功能说明

- 算子功能：完整 BLAS 定义的 SYRK（含 $\alpha$ / $\beta$ 缩放累加），并在 82 样例的 Basic SYRK
  之上支持多 batch 输入。
- 计算公式：

  $$
  D = \alpha \cdot X \cdot X^{T} + \beta \cdot Y
  $$

  其中 $X$ 形如 `(batch, M, K)`，$Y$、$D$ 形如 `(batch, M, M)` 且满足 $D = D^{T}$。

- 实现路径：MIX 双核流水。AIC 侧按 swizzle 只计算下三角 Block，将 float 精度的 $P$ / $P^{T}$
  双写入 per-core ping-pong GM workspace；AIV 侧从 workspace 读回，经 AXPBY epilogue
  （`BlockEpilogueSyrkAxpby`）执行 $D = \alpha P + \beta Y$ 并写回 GM。相比 82 样例的 cube-only
  直写路径，$Y$ 的读取与 alpha/beta 缩放移出 cube 关键路径，且累加在 float workspace 上完成，精度更好。

## 参数说明

本样例使用 `SyrkOptions`（定义于 `examples/common/options.hpp`，与 82 样例共用），本样例使用其
完整形式，命令行参数为 `m k batch alpha beta [device_id]`：

| 参数名     | 描述                                      | 约束                         |
| ---------- | ----------------------------------------- | ---------------------------- |
| `m`        | 输入矩阵 $X$ 的行数（也即输出 $Y/D$ 的边长） | 正整数                       |
| `k`        | 输入矩阵 $X$ 的列数                       | 正整数                       |
| `batch`    | 输入 batch 数（各 batch 尺寸一致，连续排布） | 正整数                       |
| `alpha`    | $X X^{T}$ 缩放系数                        | 浮点数                       |
| `beta`     | $Y$ 缩放系数                              | 浮点数                       |
| `deviceId` | 使用的 NPU 卡 ID（默认 0）                | 在设备 NPU 有效范围内        |
