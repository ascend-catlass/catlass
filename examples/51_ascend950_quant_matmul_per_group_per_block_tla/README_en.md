# 51_ascend950_quant_matmul_per_group_per_block_tla Example Readme

## Code Organization

```text
├── 51_ascend950_quant_matmul_per_group_per_block_tla
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── matmul_quant_pertile.cpp # Main file
```

## Function

- This operator supports a mixed-precision quantization strategy where the left matrix uses per-group scaling and the right matrix uses per-block scaling.
- During computation on the Cube core, data is tiled according to the L0 buffer shape. After computing the baseK block data in each iteration, the partial results are staged to the UB. On the Vector core, the system fetches the corresponding scaling factors for both the left and right matrices, which are applied directly to the sub-block matrix elements.
- Finally, accumulation is resolved on the Vector core prior to generating the output tensor.

## Example

- After obtaining the code, compile the corresponding operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution). This test case is an Ascend 950 operator. During compilation, you need to add -DCATLASS_ARCH=3510.
- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 51_ascend950_quant_matmul_per_group_per_block_tla -DCATLASS_ARCH=3510
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./51_ascend950_quant_matmul_per_group_per_block_tla 128 128 128 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
