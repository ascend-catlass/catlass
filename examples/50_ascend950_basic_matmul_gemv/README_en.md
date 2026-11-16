# BasicMatmulTlaGemv Example Readme

## Description

- Function: performs basic matrix-vector multiplication (GEMV).
- Formula:

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{j} &= \Sigma_{k} A_{k}B_{k,j}
    \end{aligned}
  $$

  where $A$ is the input vector in the shape of `(1, k)`, $B$ is the input matrix in the shape of `(k, n)`, and $C$ is the output vector in the shape of `(1, n)`.

- Supported products: Ascend 950PR&950DT series products

## Parameters

| Parameter | Attribute | Shape   | dtype   | Description                                                                     |
|-----------|-----------|---------|---------|---------------------------------------------------------------------------------|
| `A`       | Input     | `(1,k)` | `float` | Left vector; the layout is `VectorLayout`                                        |
| `B`       | Input     | `(k,n)` | `float` | Right matrix; the layout is `ColumnMajor`                                        |
| `C`       | Output    | `(1,n)` | `float` | Matrix-vector multiplication result; the layout is `RowMajor`                    |

## Usage Scope

Recommended range:

- `k ≤ 7168`

### Command Line Parameters

```bash
50_ascend950_basic_matmul_gemv [m] [n] [k] [deviceId]
```

The command line parameters are described as follows:

| Parameter  | Default Value | Description                                |
|------------|---------------|--------------------------------------------|
| `m`        | None          | Size of the m axis of matrix A, fixed to 1 |
| `n`        | None          | Size of the n axis of matrix B             |
| `k`        | None          | Size of the k axis of matrices A and B     |
| `deviceId` | `0`           | ID of the device on which the program runs |

### Example

1. Go to the project root directory and compile the sample code to generate the corresponding operator executable file. This example is an Ascend 950 operator, and `-DCATLASS_ARCH=3510` must be added during compilation. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).

    ```bash
    bash scripts/build.sh 50_ascend950_basic_matmul_gemv -DCATLASS_ARCH=3510
    ```

2. Go to the compilation directory `output/bin` of the executable file and run the operator sample program.

    ```bash
    cd output/bin
    ./50_ascend950_basic_matmul_gemv 1 128 127 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes:

    ```text
    Compare success.
    ```

## Instructions

The `DispatchPolicy MmadPingpong` used by `BasicMatmul` by default supports the following template parameters:

| Template Parameter | Default Value | Parameters                                                                                              |
| ------------------ | ------------- | ------------------------------------------------------------------------------------------------------- |
| ArchTag            | None          | Specifies the architecture model.                                                                       |
| enableUnitFlag     | false         | Whether to enable Unitflag. This parameter must be set to `false` when the L0C multi-buffer is enabled. |
| useHF32            | false         | Whether to enable HF32. Only the float type is supported.                                               |
| l0CStages          | 1             | Specifies the number of L0C buffers. Setting this parameter to `2` enables the L0C dual-buffer.         |
| enableL1Resident   | false         | Whether to enable L1 resident.                                                                          |
| l1AStages          | 1             | Number of buffers for loading matrix A to L1.                                                           |
| l1BStages          | 1             | Number of buffers for loading matrix B to L1.                                                           |
| l0AStages          | 1             | Number of buffers for loading matrix A to L0.                                                           |
| l0BStages          | 1             | Number of buffers for loading matrix B to L0.                                                           |

Assume that the matrix shape is `M N K`, the tile size on L1 is `m1 n1 k1`, the number of blocks in the M direction is `mTiles = CeilDiv(M, m1)`, the number of blocks in the N direction is `nTiles = CeilDiv(N, n1)`, and the total number of tasks is `taskBlocks = mTiles*nTiles`. In the following two cases, `enableL1Resident` can be enabled:

1. `mTiles = 1`, `nTiles > CoreNum`, and `K < 2 * k1`. In this case, you can also set `l0CStages=2` (`enableUnitFlag` must be disabled). If the space is insufficient and `l0CStages=2` cannot be set, set `n1` to half of the original value.

2. `nTiles = 1`, `mTiles > CoreNum`, and `K < 2 * k1`. In this case, you can also set `l0CStages=2` (`enableUnitFlag` must be disabled). If the space is insufficient and `l0CStages=2` cannot be set, set `m1` to half of the original value.
