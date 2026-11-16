# StridedBatchedMatmulTla Example Readme

## Code Organization

```text
├── 45_strided_batched_matmul_tla
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── strided_batched_matmul_tla.cpp # Main file
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 45_strided_batched_matmul_tla
cd output/bin
# Basic usage: Executable file name batch_axis | m_axis | n_axis | k_axis | Device_ID
# The device ID is optional. The default value is 0.
./45_strided_batched_matmul_tla 5 256 512 1024 0
```

## Batch-axis layout

The physical batch-axis position of A and B is selected by compile-time constants in the source:

```cpp
constexpr bool transA = true;
constexpr bool transB = false;
```

- `false`: the batch axis is leading, for example `(B,M,K)` for A and `(B,K,N)` for B.
- `true`: the batch axis is placed between the matrix axes, for example `(M,B,K)` for A and `(K,B,N)` for B.

Rebuild the same target after changing the constants to cover all NN, NT, TN, and TT batch-axis layouts.

```bash
# Layout customization (Supports row/col, case-insensitive; optional, defaults to row row)
# - layoutA: layout of matrix A(M,K)
# - layoutB: layout of matrix B(K, N)
# Layout parameters form an optional trailing group that can be appended to the end of any valid parameter combination.
./45_strided_batched_matmul_tla 5 256 512 1024 row col
./45_strided_batched_matmul_tla 5 256 512 1024 0 row col

# Stride customization (Unit: elements)
# - lda/ldb/ldc: leading dimension of A(M,K)/B(K,N)/C(M,N) respectively
#   - Matrix A: lda>=K if row-major; lda>=M if col-major
#   - Matrix B: ldb>=N if row-major; ldb>=K if col-major
#   - Matrix C: Fixed to row-major in this example, hence ldc>=N
# -  strideA/strideB/strideC: stride between adjacent matrix instances along the batch dimension
#   When transA/transB is true, the corresponding leading dimension must also span every batch slice inserted
#   between the matrix axes.
#
# Specifying lda/ldb/ldc only (batch strides are contiguous by default)
./45_strided_batched_matmul_tla 5 256 512 1024 0 1100 600 600
#
# Specifying leading dimensions along with batch strides (supports inter-batch padding)
./45_strided_batched_matmul_tla 5 256 512 1024 0 1100 600 600 300000 400000 500000

# Mixed layout and stride usage (layoutA/layoutB must be placed as the last two parameters on the command line)
./45_strided_batched_matmul_tla 5 256 512 1024 0 1100 600 600 300000 400000 500000 col row
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
