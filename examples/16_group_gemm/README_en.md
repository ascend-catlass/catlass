# GroupGemm Example Readme

## Function

- Operator function: performs grouped matrix multiplication (Group Gemm), where each group uses its own m, n, k and scaling factors alpha, beta.
- Formula:

  $$
    D_i = \alpha_i \times (A_i \times B_i) + \beta_i \times X_i, \quad 0 \le i < g
  $$

  where $g$ is the number of groups, and the shapes of $A_i$, $B_i$, $X_i$, $D_i$ are `(m_i,k_i)`, `(k_i,n_i)`, `(m_i,n_i)`, `(m_i,n_i)` respectively. Tensors of all groups are stored contiguously in row order, and `alpha` and `beta` are per-group scalars.
- Supported products: Atlas A2 series products and Atlas A3 series products.

## Sample Parameters

| Parameter | Attribute | Shape | dtype | Description |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m_i,k_i)` | `float16` | Left matrices of each group, layout fixed to `RowMajor`, groups are concatenated along the row direction |
| `B` | Input | `(k_i,n_i)` | `float16` | Right matrices of each group, layout fixed to `RowMajor`, groups are concatenated along the row direction |
| `X` | Input | `(m_i,n_i)` | `float16` | Accumulation matrices of each group, layout fixed to `RowMajor`, groups are concatenated along the row direction |
| `alpha` | Input | `(g,)` | `float32` | Scaling factor of the product for each group |
| `beta` | Input | `(g,)` | `float32` | Accumulation factor for each group |
| `D` | Output | `(m_i,n_i)` | `float16` | Result of each group; in this example it shares the same device memory as `X` |

## Usage Scope

Recommended scope:

- The number of groups `g` satisfies `1 ≤ g ≤ 32`.

If the number of groups exceeds 32, the program prints `Unsupported group count: require groupCnt <= 32.` and exits with a non-zero code without launching the operator.

## Example

### Command-line Arguments

```bash
16_group_gemm [groupCnt] [mList] [nList] [kList] [deviceId]
```

The command-line arguments are described as follows:

| Parameter | Default | Description |
| --- | --- | --- |
| `groupCnt` | None | Number of groups, ranging from 1 to 32 |
| `mList` | None | m-axis size of each group, separated by commas in group order |
| `nList` | None | n-axis size of each group, separated by commas in group order |
| `kList` | None | k-axis size of each group, separated by commas in group order |
| `deviceId` | `0` | Device ID used for execution |

### Execution Example

1. Go to the project root directory and compile the sample code to generate the operator executable file.

    ```bash
    bash scripts/build.sh 16_group_gemm
    ```

2. Switch to the build output directory `output/bin` and run the operator sample program.

    ```bash
    cd output/bin
    ./16_group_gemm 3 "128,256,512" "256,512,128" "512,256,128" 0
    ```

3. If the following result is displayed, the sample runs successfully and precision verification is passed:

    ```text
    Compare success.
    ```
