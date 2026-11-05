# BasicConv2d Example Readme

## Description

### basic_conv2d

- Function: performs basic 2D convolution computation.
- Formula:

  $$
    \begin{aligned}
    Output_{n, co, ho, wo} &= \sum_{ci=0}^{Cin-1} \sum_{kh=0}^{KH-1} \sum_{kw=0}^{KW-1}
      Fmap_{n, ci,\ h_s,\ w_s} \cdot Filter_{ci, kh, kw, co} \\
    h_s &= ho \cdot strideH + kh \cdot dilationH - padTop \\
    w_s &= wo \cdot strideW + kw \cdot dilationW - padLeft
    \end{aligned}
  $$

  where $Fmap$, $Filter$, and $Output$ are the input feature map, convolution kernel, and output feature map, respectively. $h_s$ and $w_s$ are the sampling coordinates on the input feature map, and out-of-bounds positions are treated as zero.

- Inputs and outputs use Ascend-friendly layouts: the feature map is `(N, C1, H, W, C0)` (`layout::NC1HWC0`), and the convolution kernel is `(Cin1, Kh, Kw, Cout, C0)` (`layout::CI1KHKWCOCI0`), where `C0 = 16`.
- Supported products: Atlas A2 series products and Atlas A3 series products

### conv2d_bias_silu

- Function: performs **Conv2D + Bias + SiLU activation fusion computation**. The implementation is taken from the Atlas A2 (2201) branch of the original `conv2d_bias_silu` example; only the 2201 platform implementation is kept in this directory.
- Formula:

  $$
    \begin{aligned}
    Output &= \mathrm{SiLU}(\mathrm{Conv2d}(Fmap, Filter) + Bias) \\
    \mathrm{SiLU}(x) &= \frac{x}{1 + e^{-x}}
    \end{aligned}
  $$

- Adopts an AIC (Cube Core) + AIV (Vector Core) cross-core collaborative architecture to avoid GM memory read/write of intermediate results and minimize memory access overhead.
- Operator pipeline: **AIC core executes Conv2D (Matrix Multiply) → Cross-core synchronization → AIV core executes BiasAdd + SiLU activation (Element-wise)**.
- Supported products: Atlas A2 series products and Atlas A3 series products

## Parameters

### basic_conv2d

| Parameter | Attribute | Shape                      | dtype     | Description                                              |
| --------- | --------- | -------------------------- | --------- | -------------------------------------------------------- |
| `Fmap`    | Input     | `(N, C1, H, W, C0)`        | `float16` | Input feature map; the layout is `NC1HWC0`               |
| `Filter`  | Input     | `(Cin1, Kh, Kw, Cout, C0)` | `float16` | Convolution kernel; the layout is `CI1KHKWCOCI0`         |
| `Output`  | Output    | `(N, Cout1, Ho, Wo, C0)`   | `float16` | Output feature map; the layout is `NC1HWC0`              |

where `C1 = CeilDiv(Cin, C0)`, `Cout1 = CeilDiv(Cout, C0)`, and `C0 = 16`.

### conv2d_bias_silu

| Parameter | Attribute | Shape                      | dtype     | Description                                                          |
| --------- | --------- | -------------------------- | --------- | -------------------------------------------------------------------- |
| `Fmap`    | Input     | `(N, C1, H, W, C0)`        | `float16` | Input feature map; the layout is `NC1HWC0`                           |
| `Filter`  | Input     | `(Cin1, Kh, Kw, Cout, C0)` | `float16` | Convolution kernel; the layout is `CI1KHKWCOCI0`                     |
| `Bias`    | Input     | `(CoutRound,)`             | `float16` | Per-output-channel bias; `CoutRound` is `Cout` aligned to `C0`       |
| `Output`  | Output    | `(N, Cout1, Ho, Wo, C0)`   | `float16` | Output feature map; the layout is `NC1HWC0`                          |

## Usage Scope

### basic_conv2d

- The input and output data type is fp16, and the input feature map and convolution kernel must be arranged in the Ascend-friendly layouts described above.
- Basic constraints:
  - Dilation coefficients `dilations` and convolution kernel strides `strides` cannot be zero.
  - The effective receptive field size of the convolution kernel cannot exceed the input feature map size:
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`

    where `hi` and `wi` are the height and width of the input feature map, `kh` and `kw` are the height and width of the convolution kernel, `dilationH` and `dilationW` are the dilation coefficients in the two directions, and `padTop`, `padBottom`, `padLeft`, and `padRight` are the top, bottom, left, and right padding sizes.

- On-chip buffer constraints (lowercase symbols in the following formulas are runtime constants; others are compile-time constants):
  - `L1_STAGES * FmapSize + L1_STAGES * FilterSize <= L1_SIZE`
    where `L1_STAGES` is 2 when double-buffering is enabled and 1 otherwise, and L1_SIZE is 512KB (Atlas A2 series products and Atlas A3 series products). FmapSize and FilterSize are calculated as follows:
    - `FmapSize = Cin1 * hi * wi * C0 * sizeof(ElementFmap)`, where `Cin1` is the tiling constant (number of input channels) under `FmapL1TileShape`; hi and wi can be derived from `Ho` and `Wo` (output feature map sizes) under `FmapL1TileShape`:
      1. `hi = (Ho - 1) * strideH + dilationH * (kh - 1) + 1`
      2. `wi = (Wo - 1) * strideW + dilationW * (kw - 1) + 1`

    - `FilterSize = Cin1 * kh * kw * Cout * C0`, where `Cin1` is the tiling constant (number of input channels) under `FilterL1TileShape`, and `Cout` is the value of `Cout` (number of output channels) under `FilterL1TileShape` aligned to `C0`

  - `L0A_STAGES * FmapL0ASize <= L0A_SIZE`, where `FmapL0ASize` is `Ho * Wo * max(L0K, kh * kw * C0) * sizeof(ElementFmap)`, `L0K` is the tiling constant under `L0TileShape`, and `L0A_SIZE` is 64KB (Atlas A2 series products and Atlas A3 series products)
  - `L0B_STAGES * FilterL0BSize <= L0B_SIZE`, where `FilterL0BSize` is `max(L0K, kh * kw * C0) * CoutL0 * sizeof(ElementFilter)`, `L0K` is the tiling constant under `L0TileShape`, `CoutL0` is the size of `L0N` aligned to `C0`, and `L0B_SIZE` is also 64KB (Atlas A2 series products and Atlas A3 series products)
  - `Ho * Wo * Cout * sizeof(ElementOutput) <= L0C_SIZE`, where `L0C_SIZE` is 128KB (Atlas A2 series products and Atlas A3 series products), and `Ho`, `Wo`, and `Cout` are all tiling constants under `FilterL1TileShape`.

  In this example, ElementFmap, ElementFilter, and ElementOutput are of the fp16 type.

### conv2d_bias_silu

- Input/Output Layout:
  - **Input feature map**: `(N, C1, H, W, C0)`, corresponding to `layout::NC1HWC0`
  - **Convolution kernel**: `(Cin1, Kh, Kw, Cout, C0)`, corresponding to `layout::CI1KHKWCOCI0`
  - **Bias**: `(CoutRound,)`, indexed by output channel
  - **Output feature map**: `(N, Cout1, Ho, Wo, C0)`, corresponding to `layout::NC1HWC0`
- Data types:
  - The feature map, convolution kernel, bias, and output are all `fp16_t` (half precision).
  - Convolution accumulation and BiasAdd/SiLU intermediate computation use float (FP32) precision by default to ensure numerical stability.
  - Optional specialization macros (both disabled by default):
    - `ENABLE_FP16_ACCUMULATOR=1`: convolution accumulation in fp16 precision
    - `ENABLE_FP16_SILU=1`: SiLU computed in fp16 precision (corresponding to the extra build target `33_conv2d_bias_silu_fp16_silu`)
- Basic constraints:
  - Dilation coefficients `dilations` and convolution kernel strides `strides` cannot be zero.
  - The effective receptive field size of the convolution kernel cannot exceed the input feature map size:
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`
  - L1/L0 cache space constraints must be satisfied (same as the [basic_conv2d constraints](#basic_conv2d)). Tile Shape:
    - `FmapL1TileShape<8, 12, 8>`, `FilterL1TileShape<96, 8>`, `L0TileShape<16, 96, 16>`
    - Note: the L1 requirement grows with `strideH/strideW` (`hi`/`wi` scale with the stride). Estimated from the Tile Shape above, with the default config a 3x3 kernel only supports stride=1 (about 98% L1 usage); for stride=2 a smaller kernel is required (for example 2x2, about 75% L1 usage), otherwise the case is rejected by `CanImplement`.
- Cross-Core Synchronization Mechanism (Atlas A2 pipeline synchronization model):
  - After an AIC core completes the Conv2D computation of a block, it sets the completion flag via the CANN native API `CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>`.
  - AIV cores wait for AIC completion via `CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>`, then execute BiasAdd + SiLU and write out the results.
  - Workspace is used for intermediate result transfer from AIC to AIV. AIC writes in half precision, with size `batch * ho * wo * coutRound * sizeof(fp16)`.
  - The Atlas A2 platform requires `hardwareSyncAddr` (obtained via `aclrtGetHardwareSyncAddr`; if the call fails, cross-core fusion cannot be enabled and the operator exits directly).

## Usage Example

### basic_conv2d

#### Command Line Parameters

```bash
33_basic_conv2d [batch] [hi] [wi] [cin] [cout] [kh] [kw] [padLeft] [padRight] [padTop] [padBottom] [strideH] [strideW] [dilationH] [dilationW] [deviceId]
```

The command line parameters are described as follows:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `batch` | None | Batch size |
| `hi` | None | Height of the input feature map |
| `wi` | None | Width of the input feature map |
| `cin` | None | Number of input channels |
| `cout` | None | Number of output channels |
| `kh` | None | Height of the convolution kernel |
| `kw` | None | Width of the convolution kernel |
| `padLeft` | None | Left padding size |
| `padRight` | None | Right padding size |
| `padTop` | None | Top padding size |
| `padBottom` | None | Bottom padding size |
| `strideH` | None | Convolution stride in the height direction |
| `strideW` | None | Convolution stride in the width direction |
| `dilationH` | None | Dilation coefficient in the height direction |
| `dilationW` | None | Dilation coefficient in the width direction |
| `deviceId` | `0` | ID of the device on which the program runs |

#### Example

1. Go to the project root directory and compile the sample code to generate the corresponding operator executable file.

    ```bash
    bash scripts/build.sh 33_basic_conv2d
    ```

2. Go to the compilation directory `output/bin` of the executable file and run the operator sample program.

    ```bash
    cd output/bin
    ./33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes:

    ```text
    Compare success.
    ```

### conv2d_bias_silu

#### Command Line Parameters

```bash
33_conv2d_bias_silu [batch] [hi] [wi] [cin] [cout] [kh] [kw] [padLeft] [padRight] [padTop] [padBottom] [strideH] [strideW] [dilationH] [dilationW] [deviceId]
```

The command line parameters are described as follows:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `batch` | None | Batch size |
| `hi` | None | Height of the input feature map |
| `wi` | None | Width of the input feature map |
| `cin` | None | Number of input channels |
| `cout` | None | Number of output channels |
| `kh` | None | Height of the convolution kernel |
| `kw` | None | Width of the convolution kernel |
| `padLeft` | None | Left padding size |
| `padRight` | None | Right padding size |
| `padTop` | None | Top padding size |
| `padBottom` | None | Bottom padding size |
| `strideH` | None | Convolution stride in the height direction |
| `strideW` | None | Convolution stride in the width direction |
| `dilationH` | None | Dilation coefficient in the height direction |
| `dilationW` | None | Dilation coefficient in the width direction |
| `deviceId` | `0` | ID of the device on which the program runs |

#### Example

1. Go to the project root directory and compile the sample code to generate the corresponding operator executable file (default float accumulation and SiLU precision).

    ```bash
    bash scripts/build.sh 33_conv2d_bias_silu
    ```

2. Go to the compilation directory `output/bin` of the executable file and run the operator sample program.

    ```bash
    cd output/bin
    ./33_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes (strong validation: the AIC workspace conv result and the AIV final output are compared separately):

    ```text
    Workspace(AIC conv) vs golden conv comparison: success.
    Output(AIV) vs golden SiLU(conv+bias) comparison: success.
    ```

4. To verify the SiLU fp16 specialization path, build the extra target `33_conv2d_bias_silu_fp16_silu` and execute it:

    ```bash
    bash scripts/build.sh 33_conv2d_bias_silu_fp16_silu
    cd output/bin
    ./33_conv2d_bias_silu_fp16_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
    ```
