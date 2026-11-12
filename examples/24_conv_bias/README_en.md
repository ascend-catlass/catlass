# ConvBias Example Readme

## Code Organization

```text
├── 24_conv_bias
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
|   ├── gen_data.py   # Generates input and benchmark data
│   └── conv_bias.cpp # Main file
```

## Function

- Implements 3D convolution functionality.
- Formula:
  Assuming the input tensor shape is $(N, C_{\text{in}}, D_i, H_i, W_i)$, the weight tensor shape is $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$, and the output tensor shape is $(N, C_{\text{out}}, D_o, H_o, W_o)$, the computation for each output element is defined as:

  $$
    \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
  $$

  Where $\star$ denotes the cross-correlation operation. $N$ represents the batch size, $C$ represents the channel count, and $D$, $H$, and $W$ represent depth, height, and width, respectively. The spatial dimensions of the output tensor are derived through:

  $$
    D_o=[(D_i + 2 * padding[0] - dilation[0] * (K_d - 1) - 1 ) / stride[0]] + 1 \\
    H_o=[(H_i + 2 * padding[1] - dilation[1] * (K_h - 1) - 1 ) / stride[1]] + 1 \\
    W_o=[(W_i + 2 * padding[2] - dilation[2] * (K_w - 1) - 1 ) / stride[2]] + 1
  $$

- The following basic constraints must be satisfied:
  - The dilation factors `dilations` and the convolution strides `strides` must not be zero.
  - The convolution kernel sizes `kd`, `kh`, and `kw` and the channel split factors `cin0` and `cin1` must not be zero.
  - The effective receptive field of the convolution kernel must not exceed the input feature map size, that is:
    - `di + 2 * padD >= dD * (kd - 1) + 1`
    - `hi + 2 * padH >= dH * (kh - 1) + 1`
    - `wi + 2 * padW >= dW * (kw - 1) + 1`

    where `di`, `hi`, and `wi` are the depth, height, and width of the input feature map, `kd`, `kh`, and `kw` are the depth, height, and width of the convolution kernel, `dD`, `dH`, and `dW` are the dilation factors along the three axes, and `padD`, `padH`, and `padW` are the padding sizes along the corresponding axes. If any constraint is violated, the program prints `[ERROR]Invalid input parameters!` and exits.

- Compared to a fully optimized CANN implementation, this basic Conv3D variant only supports full loading along the $W$-axis. Advanced performance optimizations, such as weight bypassing, L1 double-buffering, pointwise fusion, and $W$-axis tiling, are not enabled. Consequently, the memory footprints of the input, weight, and bias tensors in the L1 buffer must fit strictly within hardware limits, satisfying the following inequality:
  $$
    weightL1Size = K_h * K_w * 512 \\
    hoInL1Max = 16 / W_o + 2 \\
    hiInL1Max = (hoInL1Max - 1) * stride[1] + 1 + (K_h - 1) * dilation[1] \\
    hiInL1Max = min(H_i, hiInL1Max) \\
    inputL1Size = hiInL1Max * W_i * 32 \\
    biasL1Size = 64 \\
    weightL1Size + inputL1Size + biasL1Size < 524288
  $$

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Step 1: Execute `gen_data.py` with the command line to generate test vectors and validation assets.

```text
# python3 ./examples/24_conv_bias/gen_data.py |batch|cin|di|hi|wi|cout|kd|kh|kw|sD|sH|sW|dD|dH|dW|pD|pH|pW|dtype
# The final parameter explicitly defines the precision: float16 or bfloat16
python3 ./examples/24_conv_bias/gen_data.py 32 64 1 32 48 128 1 1 1 1 1 1 1 1 1 0 0 0 float16
```

After the command is executed, a data directory is generated in the current path, containing the operator input data and the golden data used for accuracy verification.

```text
├── data
│   ├── fmap.bin   # Input feature map (NDC1HWC0 private layout: [batch, di, cin1, hi, wi, cin0], where cin0 = 16, cin1 = ceilDiv(cin, cin0))
│   ├── weight.bin  # Convolution weight (FRACTAL_Z_3D private layout: [kdc1khkw, n1, n0, cin0], where n0 = 16, n1 = ceilDiv(cout, n0))
|   ├── bias.bin   # Convolution bias (ND layout: [cout])
│   └── golden.bin # Reference CPU convolution output (NDC1HWC0 private layout: [batch, do, cout1, ho, wo, cout0], where cout0 = 16, cout1 = ceilDiv(cout, cout0))
```

- Step 2: Execute the operator. Note that the input shape of the operator must match the shape of the data generated in the first step.

```bash
# Compile a specified test case.
bash scripts/build.sh 24_conv_bias
cd output/bin
# Executable file name |batch|di|cin1|hi|wi|cin0|cout|kd|kh|kw|sD|sH|sW|dD|dH|dW|pD|pH|pW|Device ID
# The device ID is optional. The default value is 0.
./24_conv_bias 32 1 4 32 48 16 128 1 1 1 1 1 1 1 1 1 0 0 0 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
