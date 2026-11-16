# W4A4MatmulPerTokenPerChannelDequant Example Readme

## Description

- Operator function: Performs matrix multiplication of the Int4 (`AscendC::int4b_t`) type, including per-token and per-channel dequantization coefficients.

- Formula

$$

out = perTokenScale \times x @ weight \times perChannelScale

$$

Where $x$ is the left input matrix of the matrix multiplication (in the shape of `(m, k)`), `weight` is the right input matrix (in the shape of `(k, n)`), `perChannelScale` is a one-dimensional vector in the shape of `(n)`, and `perTokenScale` is a one-dimensional vector in the shape of `(m)`.

## Parameters

The following are the running parameters of this example:

| Parameter  | Description                                                                                                           | Constraints                              |
| ---------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `m`        | Number of rows in the left matrix A (int4 format) in matrix multiplication                                            | -                                        |
| `n`        | Number of columns in the right matrix B (int4 format) in matrix multiplication                                        | Must be an even number.                  |
| `k`        | Number of columns in the left matrix A in matrix multiplication<br>(That is, the number of rows in the right matrix.) | Must be an even number.                  |
| `deviceId` | ID of the used NPU card (default: 0)                                                                                  | Within the valid range of the device NPU |

- The underlying processing mode of `AscendC::int4b_t` is to pack two `AscendC::int4b_t` data elements within `1 Byte`. For example, if a `1 Byte` block is used as the basic type view, the left matrix is in the shape of `(m, k/2)`, and the right matrix is in the shape of `(k, n/2)`.
- For more restrictions, see the restrictions description.

The key template parameters involved in this example are as follows:

| Parameter  | Description                        | Valid Range                 |
| ---------- | ---------------------------------- | --------------------------- |
| `ElementA` | Data type of the left matrix       | `AscendC::int4b_t`          |
| `ElementB` | Data type of the right matrix      | `AscendC::int4b_t`          |
| `ElementD` | Data type of the result matrix     | `bfloat16_t`                |
| `LayoutA`  | Layout format of the left matrix   | `layout::RowMajor`          |
| `LayoutB`  | Layout format of the right matrix  | `layout::zN`\| `layout::nZ` |
| `LayoutD`  | Layout format of the result matrix | `layout::RowMajor`          |

## Constraints

- `n` and `k` must be even numbers.
- When `LayoutB` is `layout::zN`:
  - `n` must be exactly divisible by 64.
  - `k` must be exactly divisible by 16.
- When `LayoutB` is set to `layout::nZ`:
  - `n` must be exactly divided by 16.
  - `k` must be exactly divided by 64.

## Code Organization

```text
├── 38_w4a4_matmul_per_token_per_channel_dequant
│   ├── CMakeLists.txt # CMake build file
│   ├── gen_data.py
│   ├── w4a4_matmul_per_token_per_channel_dequant.cpp
│   └── README.md
```

## Function

- Provides the matrix multiplication implementation in W4A4 quantization mode, using per-channel and per-token quantization stages.

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).

- Run `gen_data.py` to generate a test sample.

- Execute the operator.

The following is a complete shell script example:

```bash
# Compile the operator.
bash scripts/build.sh 38_w4a4_matmul_per_token_per_channel_dequant

# Generate test data.
cd examples/38_w4a4_matmul_per_token_per_channel_dequant/
# python gen_data.py <M> <N> <K>
python gen_data.py 256 512 1024
cd ../..

# Run execution verification.
cd output/bin/
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
./38_w4a4_matmul_per_token_per_channel_dequant 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```cpp
Compare success.
```

---

In the current example, the right-hand matrix uses the NZ layout format (that is, `LayoutB is layout::zN`, for details, see [layout.hpp](../../include/catlass/layout/layout.hpp)). To change the target format to `layout::nZ`, adjust `example/38_w4a4_matmul/w4a4_matmul.cppp` as follows:

```diff
- using LayoutB = layout::zN;
+ using LayoutB = layout::nZ;
```

In addition, set the `transB` parameter to `1` (the default value is `0`) when generating the test case. The complete test pipeline is as follows:

```bash
# Compile the operator.
bash scripts/build.sh 38_w4a4_matmul_per_token_per_channel_dequant --clean

# Generate test data.
cd examples/38_w4a4_matmul_per_token_per_channel_dequant/
# python gen_data.py <M> <N> <K> <transB>
python gen_data.py 256 512 1024 1

cd ../..

# Run execution verification.
cd output/bin/
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
./38_w4a4_matmul_per_token_per_channel_dequant 256 512 1024 0
```
