# Basic SYRK End-to-End Example

This example uses CATLASS DSL to compute a symmetric matrix product on Ascend 950.

## Functionality

For an input `X` of shape `(M, K)` or `(B, M, K)`, the default operation is
`Y = X @ X.T`, applied separately to each batch. The output has shape `(M, M)`
or `(B, M, M)` and uses FP32 storage.

The example also supports:

$$
Y = \alpha X X^T + \beta R + dI
$$

Here, `R` is an optional symmetric residual and `d` is the diagonal coefficient.
The kernel computes the lower triangle and mirrors off-diagonal tiles through
UB transpose. Split-K uses FP32 partial results and applies the coefficients
after summing them.

## Code Organization

| File | Description |
| --- | --- |
| [basic_syrk.py](basic_syrk.py) | SYRK kernels, the `syrk(...)` Python interface, and command-line accuracy and symmetry checks. |
| [README.md](README.md) | Example description and usage. |

## Constraints

- Requires Ascend 950 and contiguous, row-major FP16 or BF16 NPU tensors. Products use FP32 accumulation; the output remains FP32.
- `M`, `K`, and `B` must be positive, with `M <= 8192`, `K <= 4096`, and both `B*M*K` and `B*M*M <= 67108864`. These limits bound memory use for the example's reference checks. Tail tiles are supported.
- When `beta` is nonzero, supply a symmetric residual matching the output shape and the input device and dtype. The command-line example generates one automatically.
- The Python interface waits for the kernel to finish before returning. Autograd is not supported.

The FP32 product and epilogue output share a UB buffer. At tile size 256,
each vector unit uses at most 224 KiB: two 64 KiB FP32 product buffers, a
64 KiB FP32 transpose buffer, and an optional 32 KiB FP16/BF16 residual buffer.
The CPU reference accumulates quantized inputs and coefficients in FP64 and
rounds once to FP32. Inputs use the range `[-5, 5)` in the command-line check.

## Usage

Follow the [Quick Start](../../../../../../../python/tla_dsl/docs/en/quick_start.md) to prepare the environment.

### Command-Line Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--m` | `1024` | Input row count and output matrix size. |
| `--k` | `1024` | Reduction dimension. |
| `--batch` | `1` | Number of independent matrix products. |
| `--dtype` | `bf16` | Input dtype: `f16` or `bf16`; output is always FP32. |
| `--device` | `0` | NPU device ID. |
| `--block-num` | `-1` | Core count; `-1` selects the available count automatically. |
| `--tile` | `256` | Output tile size: `64`, `128`, or `256`; `0` selects automatically. |
| `--splitk` | `1` | Requested K split count: `1`, `2`, `4`, `8`, `16`, or `32`; `0` selects automatically. The actual count may be lower for short K dimensions. |
| `--workspace-limit-mb` | `512` | Maximum split-K workspace in MiB. |
| `--alpha` | `1.0` | Product coefficient. |
| `--beta` | `0.0` | Residual coefficient; zero skips the residual load. |
| `--diagonal` | `0.0` | Value added to each diagonal element. |

### Examples

From the repository root:

```bash
# Default matrix product.
python examples/python/tla_dsl/tla/kernel/matmul/basic_syrk/basic_syrk.py

# Batched input with tail tiles, a residual, and automatic tiling and split-K.
python examples/python/tla_dsl/tla/kernel/matmul/basic_syrk/basic_syrk.py \
  --batch 2 --m 145 --k 1025 --dtype f16 \
  --alpha 2.0 --beta -1.0 --diagonal 3.0 --tile 0 --splitk 0

# Run on one core of device 1.
python examples/python/tla_dsl/tla/kernel/matmul/basic_syrk/basic_syrk.py --device 1 --block-num 1
```

A successful run ends with:

```text
passed=True symmetric=True
```

`passed` reports agreement with the CPU reference, and `symmetric` reports
agreement with the output's transpose, using the shared accuracy tolerances.

If the run reports a missing or mismatched BC, run `python -m catlass.bc_compile` first.
