# BasicConv2d Example Readme

## 功能说明

### basic_conv2d

- 算子功能：完成基础二维卷积计算
- 计算公式：

  $$
    \begin{aligned}
    Output_{n, co, ho, wo} &= \sum_{ci=0}^{Cin-1} \sum_{kh=0}^{KH-1} \sum_{kw=0}^{KW-1}
      Fmap_{n, ci,\ h_s,\ w_s} \cdot Filter_{ci, kh, kw, co} \\
    h_s &= ho \cdot strideH + kh \cdot dilationH - padTop \\
    w_s &= wo \cdot strideW + kw \cdot dilationW - padLeft
    \end{aligned}
  $$

  其中$Fmap$、$Filter$和$Output$分别为输入特征图、卷积核和输出特征图，$h_s$、$w_s$为输入特征图上的采样坐标，越界位置按零处理。

- 输入输出采用昇腾亲和布局：特征图为`(N, C1, H, W, C0)`（`layout::NC1HWC0`），卷积核为`(Cin1, Kh, Kw, Cout, C0)`（`layout::CI1KHKWCOCI0`），其中`C0 = 16`。
- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

### conv2d_bias_silu

- 算子功能：完成 **Conv2D + Bias + SiLU 激活融合计算**，实现取自原`conv2d_bias_silu`算子的 Atlas A2（2201）分支，本目录内仅保留 2201 平台实现
- 计算公式：

  $$
    \begin{aligned}
    Output &= \mathrm{SiLU}(\mathrm{Conv2d}(Fmap, Filter) + Bias) \\
    \mathrm{SiLU}(x) &= \frac{x}{1 + e^{-x}}
    \end{aligned}
  $$

- 采用 AIC（Cube Core）+ AIV（Vector Core）跨核协同架构，避免中间结果的 GM 显存读写，实现访存开销最小化
- 算子流水线：**AIC 核执行 Conv2D（Matrix Multiply）→ 跨核同步 → AIV 核执行 BiasAdd + SiLU 激活（Element-wise）**
- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

## 样例参数说明

### basic_conv2d

| 参数     | 属性   | shape                          | dtype    | 说明                                        |
| -------- | ------ | ------------------------------ | -------- | ---------------------------------------------- |
| `Fmap`   | Input  | `(N, C1, H, W, C0)`            | `float16` | 输入特征图，layout为`NC1HWC0`               |
| `Filter` | Input  | `(Cin1, Kh, Kw, Cout, C0)`     | `float16` | 卷积核，layout为`CI1KHKWCOCI0`              |
| `Output` | Output | `(N, Cout1, Ho, Wo, C0)`       | `float16` | 输出特征图，layout为`NC1HWC0`               |

其中`C1 = CeilDiv(Cin, C0)`，`Cout1 = CeilDiv(Cout, C0)`，`C0 = 16`。

### conv2d_bias_silu

| 参数     | 属性   | shape                          | dtype    | 说明                                              |
| -------- | ------ | ------------------------------ | -------- | ------------------------------------------------- |
| `Fmap`   | Input  | `(N, C1, H, W, C0)`            | `float16` | 输入特征图，layout为`NC1HWC0`                    |
| `Filter` | Input  | `(Cin1, Kh, Kw, Cout, C0)`     | `float16` | 卷积核，layout为`CI1KHKWCOCI0`                   |
| `Bias`   | Input  | `(CoutRound,)`                 | `float16` | 输出通道偏置，`CoutRound`为`Cout`向`C0`对齐后的值 |
| `Output` | Output | `(N, Cout1, Ho, Wo, C0)`       | `float16` | 输出特征图，layout为`NC1HWC0`                    |

## 使用范围说明

### basic_conv2d

- 输入输出数据类型为fp16，输入特征图与卷积核需按上述昇腾亲和布局排布。
- 基础约束：
  - 膨胀系数`dilations`和卷积核计算步幅`strides`均不能为零
  - 卷积核的有效感受野大小不能超过输入特征图的大小，需满足：
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`

    其中`hi`、`wi`为输入特征图的高度和宽度，`kh`、`kw`为卷积核的高度和宽度，`dilationH`、`dilationW`为上述两方向上的膨胀系数，`padTop`、`padBottom`、`padLeft`、`padRight`为上、下、左、右填充大小。

- 考虑到空间分配，需满足下述条件（为做区分下述公式中小写的符号为运行时常量，反之是编译期常量）：
  - `L1_STAGES * FmapSize + L1_STAGES * FilterSize <= L1_SIZE`
    其中`L1_STAGES`在开double-buffer的情形下为2，不启用为1，L1_SIZE是512KB（Atlas A2系列产品、Atlas A3系列产品），FmapSize和FilterSize的具体计算公式为：
    - `FmapSize = Cin1 * hi * wi * C0 * sizeof(ElementFmap)`，`Cin1`为`FmapL1TileShape`下的Tiling常量（输入通道数），hi和wi由`FmapL1TileShape`下的`Ho`和`Wo`（输出特征尺寸）可反推得到，逆运算为：
      1. `hi = (Ho - 1) * strideH + dilationH * (kh - 1) + 1`
      2. `wi = (Wo - 1) * strideW + dilationW * (kw - 1) + 1`

    - `FilterSize = Cin1 * kh * kw * Cout * C0`，`Cin1`为`FilterL1TileShape`下的Tiling常量（输入通道数），`Cout`是`FilterL1TileShape`下`Cout`（输出通道数）对齐到`C0`后的值

  - `L0A_STAGES * FmapL0ASize <= L0A_SIZE`，其中`FmapL0ASize`具体为`Ho * Wo * max(L0K, kh * kw * C0) * sizeof(ElementFmap)`，`L0K`为`L0TileShape`下的Tiling常量，`L0A_SIZE`等于64KB（Atlas A2系列产品、Atlas A3系列产品）
  - `L0B_STAGES * FilterL0BSize <= L0B_SIZE`，其中`FilterL0BSize`具体为`max(L0K, kh * kw * C0) * CoutL0 * sizeof(ElementFilter)`，`L0K`为`L0TileShape`下的Tiling常量，`CoutL0`是`L0N`对齐到`C0`的大小，`L0B_SIZE`也等于64KB（Atlas A2系列产品、Atlas A3系列产品）
  - `Ho * Wo * Cout * sizeof(ElementOutput) <= L0C_SIZE`，其中`L0C_SIZE`等于128KB（Atlas A2系列产品、Atlas A3系列产品），`Ho`、`Wo`、`Cout`均为`FilterL1TileShape`中的Tiling常量。

  样例中ElementFmap，ElementFilter和ElementOutput为fp16类型。

### conv2d_bias_silu

- 输入输出 Layout：
  - **输入特征图**：`(N, C1, H, W, C0)` 对应 `layout::NC1HWC0`
  - **卷积核**：`(Cin1, Kh, Kw, Cout, C0)` 对应 `layout::CI1KHKWCOCI0`
  - **偏置 Bias**：`(CoutRound,)`，按输出通道索引
  - **输出特征图**：`(N, Cout1, Ho, Wo, C0)` 对应 `layout::NC1HWC0`
- 数据类型：
  - 特征图、卷积核、偏置、输出均为 `fp16_t`（half 精度）
  - 卷积累加与 BiasAdd、SiLU 中间计算默认使用 float（FP32）精度，确保数值稳定性
  - 可选特化宏（默认均关闭）：
    - `ENABLE_FP16_ACCUMULATOR=1`：卷积累加按 fp16 精度
    - `ENABLE_FP16_SILU=1`：SiLU 按 fp16 精度计算（对应附加编译目标`33_conv2d_bias_silu_fp16_silu`）
- 基础约束：
  - 膨胀系数 `dilations` 和卷积核步幅 `strides` 均不能为零
  - 卷积核的有效感受野大小不能超过输入特征图大小，需满足：
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`
  - 需满足 L1/L0 缓存空间约束（同[`basic_conv2d` 约束](#basic_conv2d)），Tile Shape：
    - `FmapL1TileShape<8, 12, 8>`，`FilterL1TileShape<96, 8>`，`L0TileShape<16, 96, 16>`
    - 注意：L1 需求随 `strideH/strideW` 增大而增长（`hi`/`wi` 随 stride 放大）。按上述 Tile Shape 推算，默认配置下 3×3 核仅支持 stride=1（L1 占用约 98%）；stride=2 时需使用更小的卷积核（如 2×2，L1 占用约 75%），否则会被 `CanImplement` 拒绝
- 跨核同步机制（Atlas A2 管线同步模型）：
  - AIC 核完成单个 block 的 Conv2D 计算后，通过 CANN 原生 API `CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>` 设置完成标志
  - AIV 核通过 `CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>` 等待 AIC 完成，随后执行 BiasAdd + SiLU 并写出结果
  - Workspace 用于 AIC→AIV 的中间结果传递，AIC 以 half 精度写入，大小为 `batch * ho * wo * coutRound * sizeof(fp16)`
  - Atlas A2 平台需要 `hardwareSyncAddr`（通过 `aclrtGetHardwareSyncAddr` 获取；获取失败时算子无法启用跨核融合，将直接退出）

## 使用示例

### basic_conv2d

#### 命令行参数

```bash
33_basic_conv2d [batch] [hi] [wi] [cin] [cout] [kh] [kw] [padLeft] [padRight] [padTop] [padBottom] [strideH] [strideW] [dilationH] [dilationW] [deviceId]
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

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 33_basic_conv2d
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```

### conv2d_bias_silu

#### 命令行参数

```bash
33_conv2d_bias_silu [batch] [hi] [wi] [cin] [cout] [kh] [kw] [padLeft] [padRight] [padTop] [padBottom] [strideH] [strideW] [dilationH] [dilationW] [deviceId]
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

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件（默认 float 累加与 SiLU 精度）。

    ```bash
    bash scripts/build.sh 33_conv2d_bias_silu
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./33_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过（强校验：分别对比 AIC workspace 的纯卷积结果与 AIV 最终输出）：

    ```text
    Workspace(AIC conv) vs golden conv comparison: success.
    Output(AIV) vs golden SiLU(conv+bias) comparison: success.
    ```

4. 如需验证 SiLU 按 fp16 计算的特化路径，编译附加目标`33_conv2d_bias_silu_fp16_silu`并执行：

    ```bash
    bash scripts/build.sh 33_conv2d_bias_silu_fp16_silu
    cd output/bin
    ./33_conv2d_bias_silu_fp16_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
    ```
