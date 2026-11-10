# SparseMatmulTla Example Readme

## Description

- Function: performs a 4:2 structured sparse matrix multiplication. Matrix `A` is a dense `int8` matrix. Matrix `B` is a sparse matrix that keeps two non-zero elements in every four consecutive elements, represented by a compressed matrix and an index matrix. The output is `int32`.
- Formula:

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  where $A$ is a dense matrix in the shape of `(m, k)`, and $B$ is a sparse matrix in the shape of `(k, n)` reconstructed from the compressed matrix `(k/2, n)` and the index matrix. $C$ is the output matrix in the shape of `(m, n)`.

- Supported products: Atlas A2 series products

## Parameters

| Parameter | Attribute | Shape | dtype | Description |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m, k)` | `int8` | Left matrix with the `RowMajor` layout |
| `B` | Input | `(k/2, n)` | `int8` | 4:2 sparse compressed storage of the right matrix with the `ColumnMajor` layout (transB); only two non-zero values are stored in each four-element block |
| `index` | Input | `(n, k/8)` | `uint8` | 4:2 sparse index matrix stored in the NZ fractal layout, indicating the positions of non-zero elements in `B` |
| `C` | Output | `(m, n)` | `int32` | Matrix multiplication result with the `RowMajor` layout |

## Usage Scope

The example uses batch=1 and calls `CanImplement` at startup to validate the problem shape. The example exits with an error if the shape is invalid.

Constraints:

- `m`, `n`, and `k` must be greater than 0.
- `k % 64 == 0`, which also satisfies the 8-element alignment required by 4:2 sparsity.
- `n % 16 == 0`.
- `k ≤ 65536`.
- The compile-time tile configuration `<128, 256, 128>` and `<128, 256, 64>` must fit the L0/L1 capacity; the current configuration satisfies this requirement.

This example is provided for functional verification and does not define a performance-recommended range.

## Usage Example

### Command Line Parameters

```bash
41_sparse_matmul_tla [m] [n] [k] [deviceId]
```

The command line parameters are described as follows:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `m` | None | Size of the m axis of matrix A |
| `n` | None | Size of the n axis of matrix B |
| `k` | None | Size of the k axis of matrices A and B |
| `deviceId` | `0` | ID of the device on which the program runs |

### Example

1. Generate test data in the project root directory.

    ```bash
    python3 examples/41_sparse_matmul_tla/sparse_gen_data.py 160 320 64
    ```

    `input/` contains `x1_gm.bin`, `x2_gm.bin`, and `index_gm.bin`; `output/` contains `golden.bin`.

2. Compile the example.

    ```bash
    bash scripts/build.sh 41_sparse_matmul_tla
    ```

3. Go to `output/bin` and run the example.

    ```bash
    cd output/bin
    ./41_sparse_matmul_tla 160 320 64 0
    ```

4. If the following result is displayed, the example is successfully executed and the precision verification passes:

    ```text
    Compare success.
    ```
