# BasicConv2dTla Example Readme

## 功能说明

### basic_conv2d_tla

- 算子功能：使用[`TLA`](../../docs/zh/2_Design/02_tla/01_layout.md)抽象完成基础二维卷积计算
- 计算公式：

  $$
    \begin{aligned}
    Output_{n, co, ho, wo} &= \sum_{ci=0}^{Cin-1} \sum_{kh=0}^{Kh-1} \sum_{kw=0}^{Kw-1}
      Fmap_{n, ci,\ h_s,\ w_s} \cdot Filter_{ci, kh, kw, co} \\
    h_s &= ho \cdot strideH + kh \cdot dilationH - padTop \\
    w_s &= wo \cdot strideW + kw \cdot dilationW - padLeft
    \end{aligned}
  $$

  其中$Fmap$、$Filter$和$Output$分别为输入特征图、卷积核和输出特征图，$h_s$、$w_s$为输入特征图上的采样坐标，越界位置按零处理。

- 输入输出采用昇腾亲和布局：特征图为`(N, C1, H, W, C0)`（`layout::NC1HWC0`），卷积核为`(Cin1, Kh, Kw, Cout, C0)`（`layout::CI1KHKWCOCI0`），其中`C0 = 16`。
- 总体设计与[`33_basic_conv2d`](../33_basic_conv2d/README.md)相同，区别为使用了TLA相关抽象。
- 支持产品型号：Ascend 950PR&950DT系列产品

### conv2d_bias_silu

- 算子功能：完成 **Conv2D + Bias + SiLU 激活融合计算**，实现取自原`conv2d_bias_silu`算子的 Ascend 950（3510）分支，本目录内仅保留 3510 平台实现
- 计算公式：

  $$
    \begin{aligned}
    Output &= \mathrm{SiLU}(\mathrm{Conv2d}(Fmap, Filter) + Bias) \\
    \mathrm{SiLU}(x) &= \frac{x}{1 + e^{-x}}
    \end{aligned}
  $$

- 采用 AIC（Cube Core）+ AIV（Vector Core）跨核协同架构，避免中间结果的 GM 显存读写，实现访存开销最小化
- 算子流水线：**AIC 核执行 Conv2D（Matrix Multiply）→ 跨核同步 → AIV 核执行 BiasAdd + SiLU 激活（Element-wise）**
- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

### basic_conv2d_tla

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `Fmap` | Input | `(N, C1, H, W, C0)` | `float16` | 输入特征图，layout为`NC1HWC0` |
| `Filter` | Input | `(Cin1, Kh, Kw, Cout, C0)` | `float16` | 卷积核，layout为`CI1KHKWCOCI0` |
| `Output` | Output | `(N, Cout1, Ho, Wo, C0)` | `float16` | 输出特征图，layout为`NC1HWC0` |

其中`C1 = CeilDiv(Cin, C0)`，`Cout1 = CeilDiv(Cout, C0)`，`C0 = 16`。

### conv2d_bias_silu

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `Fmap` | Input | `(N, C1, H, W, C0)` | `float16` | 输入特征图，layout为`NC1HWC0` |
| `Filter` | Input | `(Cin1, Kh, Kw, Cout, C0)` | `float16` | 卷积核，layout为`CI1KHKWCOCI0` |
| `Bias` | Input | `(CoutRound,)` | `float16` | 输出通道偏置，`CoutRound`为`Cout`向`C0`对齐后的值 |
| `Output` | Output | `(N, Cout1, Ho, Wo, C0)` | `float16` | 输出特征图，layout为`NC1HWC0` |

## 使用范围说明

### basic_conv2d_tla

- 输入输出数据类型为fp16，输入特征图与卷积核需按上述昇腾亲和布局排布。
- 基础约束：
  - 膨胀系数`dilations`和卷积核计算步幅`strides`均不能为零
  - 卷积核的有效感受野大小不能超过输入特征图的大小，需满足：
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`

    其中`hi`、`wi`为输入特征图的高度和宽度，`kh`、`kw`为卷积核的高度和宽度，`dilationH`、`dilationW`为上述两方向上的膨胀系数，`padTop`、`padBottom`、`padLeft`、`padRight`为上、下、左、右填充大小。

- 空间约束（Ascend 950 的`L1_SIZE`为512KB，`L0A_SIZE`、`L0B_SIZE`为64KB，`L0C_SIZE`为256KB）：
  - `L1A_STAGES * FmapSize + L1B_STAGES * FilterSize <= L1_SIZE`，本样例中`L1A_STAGES`与`L1B_STAGES`均为2，其中：
    - `FmapSize = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0`，`hiBlock = (Ho - 1) * strideH + (kh - 1) * dilationH + 1`，`wiBlock = (Wo - 1) * strideW + (kw - 1) * dilationW + 1`，`Ho`、`Wo`为`FmapL1TileShape`中的Tiling常量，`BYTE_PER_C0 = 32`字节
    - `FilterSize = FilterL1TileShape::Cin1 * kh * kw * coutRound * BYTE_PER_C0`，`coutRound`为`FilterL1TileShape::Cout`向`C0`对齐后的值
  - `L0B_TILE_SIZE * L0B_STAGES <= L0B_SIZE`，其中`L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementFilter)`
  - `kh * kw * C0 <= 512`（L0B 的 `LoadData3D` 单次 kPart 硬件上限，超出时`CanImplement`判定为不可实现）
- 样例 Tile Shape：`FmapL1TileShape<8, 12, 8>`、`FilterL1TileShape<96, 8>`、`L0TileShape<16, 96, 16>`。按上述公式推算，默认 3×3 卷积核在 stride=1 时 L1 占用约 98%，stride=2 时超出 L1 容量，会被`CanImplement`拒绝。

### conv2d_bias_silu

- 输入输出 Layout：
  - **输入特征图**：`(N, C1, H, W, C0)` 对应 `layout::NC1HWC0`
  - **卷积核**：`(Cin1, Kh, Kw, Cout, C0)` 对应 `layout::CI1KHKWCOCI0`
  - **偏置 Bias**：`(CoutRound,)`，按输出通道索引
  - **输出特征图**：`(N, Cout1, Ho, Wo, C0)` 对应 `layout::NC1HWC0`
- 数据类型：
  - 特征图、卷积核、偏置、输出均为 `fp16_t`（half 精度）
  - 卷积累加与 BiasAdd、SiLU 中间计算使用 float（FP32）精度，确保激活函数的数值稳定性
- 基础约束：
  - 膨胀系数 `dilations` 和卷积核步幅 `strides` 均不能为零
  - 卷积核的有效感受野大小不能超过输入特征图大小，需满足：
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`
  - 需满足 L1/L0 缓存空间约束（同[`basic_conv2d_tla` 约束](#basic_conv2d_tla)），Tile Shape：
    - `FmapL1TileShape<4, 8, 4>`、`FilterL1TileShape<48, 4>`、`L0TileShape<16, 16, 16>`；按上述公式推算，默认配置下 L1 占用约 29%
- 跨核同步机制（3510 bulk 同步模型）：
  - AIC 核完成全部 Conv2D 计算后，通过 `CrossCoreBarrier` 等待所有 AIC 就绪，再以 `CrossCoreSetFlag<0x2, PIPE_FIX>` 设置完成标志
  - AIV 核通过 `CrossCoreWaitFlag<0x2, PIPE_MTE2>` 等待 AIC 完成，随后执行 BiasAdd + SiLU 并写出结果
  - Workspace 用于 AIC→AIV 的中间结果传递，AIC 以 half 精度写入，大小为 `batch * ho * wo * coutRound * sizeof(fp16)`
  - 3510 平台无需 `hardwareSyncAddr`（区别于 Atlas A2）

## 使用示例

### basic_conv2d_tla

#### 命令行参数

```bash
56_ascend950_basic_conv2d_tla [batch] [hi] [wi] [cin] [cout] [kh] [kw] [padLeft] [padRight] [padTop] [padBottom] [strideH] [strideW] [dilationH] [dilationW] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | 批量 Batch 大小 |
| `hi` | 无 | 输入特征图高度 |
| `wi` | 无 | 输入特征图宽度 |
| `cin` | 无 | 输入通道数 |
| `cout` | 无 | 输出通道数 |
| `kh` | 无 | 卷积核高度 |
| `kw` | 无 | 卷积核宽度 |
| `padLeft` | 无 | 左填充大小 |
| `padRight` | 无 | 右填充大小 |
| `padTop` | 无 | 上填充大小 |
| `padBottom` | 无 | 下填充大小 |
| `strideH` | 无 | 高度方向卷积步幅 |
| `strideW` | 无 | 宽度方向卷积步幅 |
| `dilationH` | 无 | 高度方向膨胀系数 |
| `dilationW` | 无 | 宽度方向膨胀系数 |
| `deviceId` | `0` | 指定运行设备ID |

#### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件（本样例为Ascend 950算子，需指定`-DCATLASS_ARCH=3510`）。

    ```bash
    bash scripts/build.sh 56_ascend950_basic_conv2d_tla -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./56_ascend950_basic_conv2d_tla 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```

### conv2d_bias_silu

#### 命令行参数

```bash
56_ascend950_conv2d_bias_silu [batch] [hi] [wi] [cin] [cout] [kh] [kw] [padLeft] [padRight] [padTop] [padBottom] [strideH] [strideW] [dilationH] [dilationW] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | 批量 Batch 大小 |
| `hi` | 无 | 输入特征图高度 |
| `wi` | 无 | 输入特征图宽度 |
| `cin` | 无 | 输入通道数 |
| `cout` | 无 | 输出通道数 |
| `kh` | 无 | 卷积核高度 |
| `kw` | 无 | 卷积核宽度 |
| `padLeft` | 无 | 左填充大小 |
| `padRight` | 无 | 右填充大小 |
| `padTop` | 无 | 上填充大小 |
| `padBottom` | 无 | 下填充大小 |
| `strideH` | 无 | 高度方向卷积步幅 |
| `strideW` | 无 | 宽度方向卷积步幅 |
| `dilationH` | 无 | 高度方向膨胀系数 |
| `dilationW` | 无 | 宽度方向膨胀系数 |
| `deviceId` | `0` | 指定运行设备ID |

#### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件（本样例为Ascend 950算子，需指定`-DCATLASS_ARCH=3510`）。

    ```bash
    bash scripts/build.sh 56_ascend950_conv2d_bias_silu -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./56_ascend950_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 2 2 1 1 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过（强校验：分别对比 AIC workspace 的纯卷积结果与 AIV 最终输出）：

    ```text
    Workspace(AIC conv) vs golden conv comparison: success.
    Output(AIV) vs golden SiLU(conv+bias) comparison: success.
    ```
