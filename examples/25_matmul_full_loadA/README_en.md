# Matmul Full LoadA Example Readme

## Code Organization

```text
├── 25_matmul_full_loadA
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ └── matmul_full_loadA.cpp # Main file
```

## Function

- This operator supports full loading of matrix A based on 00_basic_matmul (half of the L1 space needs to be allocated to the matrix with a data volume of `L1TileShape::M * problemShape.K`). When each basic block is computed, the entire block of matrix A is moved in, and then matrix B is moved in using ping-pong mode. If the L1 space is insufficient for full loading of matrix A, an error is returned.
- When matrix A is fully loaded, a larger N-axis indicates that a single core can reuse matrix A in L1 multiple times without transferring matrix A from GM or L2 cache. In this case, the performance gain is greater.
- When matrix A is fully loaded and the N-axis is small, matrix A cannot be reused, and the performance gain may deteriorate compared with that of `00_basic_matmul`.
- If problemShape.M <= L1TileShape::M (i.e., no M-dimension tiling or core splitting), the common GemmIdentityBlockSwizzle strategy is applicable.
- If `problemShape.M > L1TileShape::M`, a new `GemmIdentityBlockSwizzleL1FullLoad<SwizzleOffset, SwizzleDirection, AicCoreNum>` is provided. It makes the basic blocks processed by each core as contiguous as possible, improving inter-block reuse when matrix A is fully loaded across cores.
- Using 24 cube cores as an example, the common `GemmIdentityBlockSwizzle` strategy distributes basic blocks across cores in the order `0-1-2-...-22-23-0-1-2...-22-23-0-1-2...`, meaning basic blocks for each core are interleaved. The new GemmIdentityBlockSwizzleL1FullLoad strategy distributes basic blocks in the order `0-0-...-0-1-1-...-1-2-2-...-23`, meaning basic blocks for each core are consecutive.

## Example

- After obtaining the code, build the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 25_matmul_full_loadA
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./25_matmul_full_loadA 256 512 1024 0
```

If the following information is displayed, the accuracy comparison is successful.

```text
Compare success.
```
