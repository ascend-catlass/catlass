# BasicMatmul Example Readme

## Description

- Function: performs basic matrix multiplication.
- Formula:

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  where $A$ and $B$ are input matrices in the shape of `(m, k)` and `(k, n)`, respectively. $C$ is the output matrix in the shape of `(m, n)`.
- Supported products: Atlas A2/A3 training/inference series products

## Parameters

| Parameter | Attribute | Shape   | dtype              | Description                                                                                                            |
|-----------|-----------|---------|--------------------|------------------------------------------------------------------------------------------------------------------------|
| `A`       | Input     | `(m,k)` | `float16/bfloat16` | Left matrix; the layout supports `RowMajor` (default) and `ColumnMajor`                                                |
| `B`       | Input     | `(k,n)` | `float16/bfloat16` | Right matrix; the layout supports `RowMajor` (default) and `ColumnMajor`; the data type is the same as the left matrix |
| `C`       | Output    | `(m,n)` | `float16/bfloat16` | Matrix multiplication result; only the `RowMajor` layout is supported; the data type is the same as the left matrix    |

## Usage Scope

This example does not include optimizations such as Padding, Preload, or split-K, and serves as the common baseline for the optimized examples.

Recommended range:

- `m ≥ 256, n ≥ 256, 256 < k ≤ 3072`
- The K axis and N axis are aligned to 512B.

### Command Line Parameters

```bash
00_basic_matmul [m] [n] [k] [deviceId]
```

The command line parameters are described as follows:

| Parameter  | Default Value | Description                                |
|------------|---------------|--------------------------------------------|
| `m`        | None          | Size of the m axis of matrix A             |
| `n`        | None          | Size of the n axis of matrix B             |
| `k`        | None          | Size of the k axis of matrices A and B     |
| `deviceId` | `0`           | ID of the device on which the program runs |

### Example

1. Go to the project root directory and compile the sample code to generate the corresponding operator executable file.

    ```bash
    bash scripts/build.sh 00_basic_matmul
    ```

2. Go to the compilation directory `output/bin` of the executable file and run the operator sample program.

    ```bash
    cd output/bin
    ./00_basic_matmul 256 512 1024 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes:

    ```text
    Compare success.
    ```
