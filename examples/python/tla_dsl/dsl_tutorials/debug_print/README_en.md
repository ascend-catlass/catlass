# `tla.print` examples

[中文](README.md)

## Description

These examples show printing inside `tla.cube()` / `tla.vector()` and SIMD/SIMT `tla.vec.func`, and check both output and computation results.

| Example | Demonstrates |
| --- | --- |
| [`debug_print.py`](debug_print.py), [`debug_print_format.py`](debug_print_format.py) | Outer scalar values, scalar expressions and format strings |
| [`vf_print.py`](vf_print.py) | SIMD scalars, registers and UB data |
| [`simt_print.py`](simt_print.py) | SIMT thread scalars, GM and UB data |
| [`vf_print_mixed.py`](vf_print_mixed.py) | Combined Cube, Vector, SIMD and SIMT calls |

Supported types, memory locations, `length`, synchronization and log capacity are described in the [printing guide](../../../../../python/tla_dsl/docs/en/kernel_development/evaluation/print.md). `tla.print` does not support pointer formatting.

## Running the examples

Run the commands below from the repository root. `--device` is a logical device index visible to the process. Examples using `tla.vec.func` compile without `--run`; add `--run --device` to compile and execute them.

### Scalar expressions

`debug_print.py --expression` prints `lhs + rhs` inside `tla.vector()`. `--value` and `--rhs` supply the two operands:

```bash
python examples/python/tla_dsl/dsl_tutorials/debug_print/debug_print.py --expression --dtype f32 --value 1.25 --rhs 0.75 --run --device 0
```

### SIMD printing

```bash
python examples/python/tla_dsl/dsl_tutorials/debug_print/vf_print.py --run --device 0 --calls 1
```

`--calls 1` launches once. The example prints the first four register results `[-7, -6.5, -6, -5.5]` and input/output UB prefixes, then reports `output_ok=True` when its checks pass.

### SIMT printing

`simt_print.py` shows how to select printing threads, synchronize shared UB across a thread block, and call the same compiled object again after updating its input.

```bash
# Compile only.
python examples/python/tla_dsl/dsl_tutorials/debug_print/simt_print.py

# Compile and run.
python examples/python/tla_dsl/dsl_tutorials/debug_print/simt_print.py --run --device 0
```

### Mixed printing

`vf_print_mixed.py` uses the C1V2 core ratio with two logical blocks. Each block has two Vector subblocks, each processing 64 elements; `subblock=0` and `subblock=1` identify them in the output. The example checks printed values and block/subblock identifiers. Output order across cores is unspecified.

```bash
python examples/python/tla_dsl/dsl_tutorials/debug_print/vf_print_mixed.py --run --device 0 --block-num 2
```
