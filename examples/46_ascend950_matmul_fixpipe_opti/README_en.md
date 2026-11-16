# MatmulFixpipeOpti Example Readme

## Code Organization

```text
├── 46_ascend950_matmul_fixpipe_opti
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ ├── 46_ascend950_matmul_fixpipe_opti.md # Design document
│ └── matmul_fixpipe_opti.cpp # Main file
```

## Function

Each time the operator completes the computation of a basic block, the result data is moved to the UB through Fixpipe. When dualDstCtrl is enabled, the computation result matrix is divided into two parts and written in parallel to the dedicated UBs of two Vector cores (one Cube core corresponds to two Vector cores). The UB of each Vector core supports independent enabling of Double Buffer to accelerate pipeline efficiency.

## Example

- After obtaining the code, Build the corresponding operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution). In this example, the operator is an Ascend 950 operator. During compilation, add -DCATLASS_ARCH=3510.
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 46_ascend950_matmul_fixpipe_opti -DCATLASS_ARCH=3510
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./46_ascend950_matmul_fixpipe_opti 128 128 128 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```
