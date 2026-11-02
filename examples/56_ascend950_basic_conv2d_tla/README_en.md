# BasicConv2dTla Example Readme

## Description

### basic_conv2d_tla

- Function: performs basic 2D convolution computation using [`TLA`](../../docs/en/2_Design/02_tla/01_layout.md) abstractions.
- Formula:

  $$
    \begin{aligned}
    Output_{n, co, ho, wo} &= \sum_{ci=0}^{Cin-1} \sum_{kh=0}^{Kh-1} \sum_{kw=0}^{Kw-1}
      Fmap_{n, ci,\ h_s,\ w_s} \cdot Filter_{ci, kh, kw, co} \\
    h_s &= ho \cdot strideH + kh \cdot dilationH - padTop \\
    w_s &= wo \cdot strideW + kw \cdot dilationW - padLeft
    \end{aligned}
  $$

  where $Fmap$, $Filter$, and $Output$ are the input feature map, convolution kernel, and output feature map, respectively. $h_s$ and $w_s$ are the sampling coordinates on the input feature map, and out-of-bounds positions are treated as zero.

- Inputs and outputs use Ascend-friendly layouts: the feature map is `(N, C1, H, W, C0)` (`layout::NC1HWC0`), and the convolution kernel is `(Cin1, Kh, Kw, Cout, C0)` (`layout::CI1KHKWCOCI0`), where `C0 = 16`.
- The overall design is the same as that of [`33_basic_conv2d`](../33_basic_conv2d/README_en.md), except that TLA-related abstractions are used.
- Supported products: Ascend 950PR&950DT series products

### conv2d_bias_silu

- Function: performs **Conv2D + Bias + SiLU activation fusion computation**. The implementation is taken from the Ascend 950 (3510) branch of the original `conv2d_bias_silu` example; only the 3510 platform implementation is kept in this directory.
- Formula:

  $$
    \begin{aligned}
    Output &= \mathrm{SiLU}(\mathrm{Conv2d}(Fmap, Filter) + Bias) \\
    \mathrm{SiLU}(x) &= \frac{x}{1 + e^{-x}}
    \end{aligned}
  $$

- Adopts an AIC (Cube Core) + AIV (Vector Core) cross-core collaborative architecture to avoid GM memory read/write of intermediate results and minimize memory access overhead.
- Operator pipeline: **AIC core executes Conv2D (Matrix Multiply) → Cross-core synchronization → AIV core executes BiasAdd + SiLU activation (Element-wise)**.
- Supported products: Ascend 950PR&950DT series products

## Parameters

### basic_conv2d_tla

| Parameter | Attribute | Shape | dtype | Description |
| --- | --- | --- | --- | --- |
| `Fmap` | Input | `(N, C1, H, W, C0)` | `float16` | Input feature map, layout is `NC1HWC0` |
| `Filter` | Input | `(Cin1, Kh, Kw, Cout, C0)` | `float16` | Convolution kernel, layout is `CI1KHKWCOCI0` |
| `Output` | Output | `(N, Cout1, Ho, Wo, C0)` | `float16` | Output feature map, layout is `NC1HWC0` |

where `C1 = CeilDiv(Cin, C0)`, `Cout1 = CeilDiv(Cout, C0)`, and `C0 = 16`.

### conv2d_bias_silu

| Parameter | Attribute | Shape | dtype | Description |
| --- | --- | --- | --- | --- |
| `Fmap` | Input | `(N, C1, H, W, C0)` | `float16` | Input feature map, layout is `NC1HWC0` |
| `Filter` | Input | `(Cin1, Kh, Kw, Cout, C0)` | `float16` | Convolution kernel, layout is `CI1KHKWCOCI0` |
| `Bias` | Input | `(CoutRound,)` | `float16` | Output channel bias; `CoutRound` is `Cout` aligned up to `C0` |
| `Output` | Output | `(N, Cout1, Ho, Wo, C0)` | `float16` | Output feature map, layout is `NC1HWC0` |

## Usage Scope

### basic_conv2d_tla

- The input and output data type is fp16, and the input feature map and convolution kernel must be arranged in the Ascend-friendly layouts described above.
- Basic constraints:
  - The dilation coefficients `dilations` and the convolution strides `strides` cannot be zero.
  - The effective receptive field of the convolution kernel cannot exceed the input feature map size:
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`

    where `hi` and `wi` are the height and width of the input feature map, `kh` and `kw` are the height and width of the convolution kernel, `dilationH` and `dilationW` are the dilation coefficients in the two directions, and `padTop`, `padBottom`, `padLeft`, and `padRight` are the top, bottom, left, and right padding sizes.

- Space constraints (on Ascend 950, `L1_SIZE` is 512KB, `L0A_SIZE` and `L0B_SIZE` are 64KB, and `L0C_SIZE` is 256KB):
  - `L1A_STAGES * FmapSize + L1B_STAGES * FilterSize <= L1_SIZE`, where `L1A_STAGES` and `L1B_STAGES` are both 2 in this example, and:
    - `FmapSize = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0`, where `hiBlock = (Ho - 1) * strideH + (kh - 1) * dilationH + 1`, `wiBlock = (Wo - 1) * strideW + (kw - 1) * dilationW + 1`, `Ho` and `Wo` are tiling constants in `FmapL1TileShape`, and `BYTE_PER_C0 = 32` bytes.
    - `FilterSize = FilterL1TileShape::Cin1 * kh * kw * coutRound * BYTE_PER_C0`, where `coutRound` is `FilterL1TileShape::Cout` aligned up to `C0`.
  - `L0B_TILE_SIZE * L0B_STAGES <= L0B_SIZE`, where `L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementFilter)`.
  - `kh * kw * C0 <= 512` (the hardware limit of a single kPart in the L0B `LoadData3D`; exceeding it makes `CanImplement` report the operator as not implementable).
- Tile Shape of this example: `FmapL1TileShape<8, 12, 8>`, `FilterL1TileShape<96, 8>`, and `L0TileShape<16, 96, 16>`. Estimated from the formulas above, with the default 3x3 kernel and stride=1 the L1 usage is about 98%; with stride=2 the L1 capacity is exceeded and `CanImplement` rejects the configuration.

### conv2d_bias_silu

- Input/output layouts:
  - **Input feature map**: `(N, C1, H, W, C0)` corresponding to `layout::NC1HWC0`
  - **Convolution kernel**: `(Cin1, Kh, Kw, Cout, C0)` corresponding to `layout::CI1KHKWCOCI0`
  - **Bias**: `(CoutRound,)`, indexed by output channel
  - **Output feature map**: `(N, Cout1, Ho, Wo, C0)` corresponding to `layout::NC1HWC0`
- Data types:
  - The feature map, convolution kernel, bias, and output are all `fp16_t` (half precision).
  - Convolution accumulation, BiasAdd, and SiLU intermediate computation use float (FP32) precision to ensure numerical stability of the activation function.
- Basic constraints:
  - The dilation coefficients `dilations` and the convolution strides `strides` cannot be zero.
  - The effective receptive field of the convolution kernel cannot exceed the input feature map size:
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`
  - The L1/L0 cache space constraints must be satisfied (the same as the [`basic_conv2d_tla` constraints](#basic_conv2d_tla)). Tile Shape:
    - `FmapL1TileShape<4, 8, 4>`, `FilterL1TileShape<48, 4>`, and `L0TileShape<16, 16, 16>`; estimated from the formulas above, the L1 usage is about 29% with the default configuration.
- Cross-core synchronization mechanism (3510 bulk synchronization model):
  - After an AIC core completes all Conv2D computation, it waits for all AIC cores via `CrossCoreBarrier`, then sets the completion flag with `CrossCoreSetFlag<0x2, PIPE_FIX>`.
  - AIV cores wait for AIC completion via `CrossCoreWaitFlag<0x2, PIPE_MTE2>`, then execute BiasAdd + SiLU and write out the results.
  - Workspace is used for intermediate result transfer from AIC to AIV. AIC writes it in half precision, and its size is `batch * ho * wo * coutRound * sizeof(fp16)`.
  - The 3510 platform does not require `hardwareSyncAddr` (unlike Atlas A2).

## Usage Example

### basic_conv2d_tla

#### Command Line Parameters

```bash
56_ascend950_basic_conv2d_tla [batch] [hi] [wi] [cin] [cout] [kh] [kw] [padLeft] [padRight] [padTop] [padBottom] [strideH] [strideW] [dilationH] [dilationW] [deviceId]
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
| `strideH` | None | Convolution stride along the height direction |
| `strideW` | None | Convolution stride along the width direction |
| `dilationH` | None | Dilation coefficient along the height direction |
| `dilationW` | None | Dilation coefficient along the width direction |
| `deviceId` | `0` | ID of the device on which the program runs |

#### Example

1. Go to the project root directory and compile the sample code to generate the corresponding operator executable file. This example is an Ascend 950 operator, so `-DCATLASS_ARCH=3510` must be specified.

    ```bash
    bash scripts/build.sh 56_ascend950_basic_conv2d_tla -DCATLASS_ARCH=3510
    ```

2. Go to the compilation directory `output/bin` of the executable file and run the operator sample program.

    ```bash
    cd output/bin
    ./56_ascend950_basic_conv2d_tla 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes:

    ```text
    Compare success.
    ```

### conv2d_bias_silu

#### Command Line Parameters

```bash
56_ascend950_conv2d_bias_silu [batch] [hi] [wi] [cin] [cout] [kh] [kw] [padLeft] [padRight] [padTop] [padBottom] [strideH] [strideW] [dilationH] [dilationW] [deviceId]
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
| `strideH` | None | Convolution stride along the height direction |
| `strideW` | None | Convolution stride along the width direction |
| `dilationH` | None | Dilation coefficient along the height direction |
| `dilationW` | None | Dilation coefficient along the width direction |
| `deviceId` | `0` | ID of the device on which the program runs |

#### Example

1. Go to the project root directory and compile the sample code to generate the corresponding operator executable file. This example is an Ascend 950 operator, so `-DCATLASS_ARCH=3510` must be specified.

    ```bash
    bash scripts/build.sh 56_ascend950_conv2d_bias_silu -DCATLASS_ARCH=3510
    ```

2. Go to the compilation directory `output/bin` of the executable file and run the operator sample program.

    ```bash
    cd output/bin
    ./56_ascend950_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 2 2 1 1 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes (strong validation: the AIC workspace conv result and the AIV final output are compared separately):

    ```text
    Workspace(AIC conv) vs golden conv comparison: success.
    Output(AIV) vs golden SiLU(conv+bias) comparison: success.
    ```
