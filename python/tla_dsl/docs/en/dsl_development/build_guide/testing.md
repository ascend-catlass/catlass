---
nav_order: 30
---

# Running Test Cases

## pytest: Frontend Lowering to TLA IR

```bash
cd /path/to/catlass/python/tla_dsl
python -m pytest -q tests
```

Use the same entry point to run a single test file, for example:

```bash
python -m pytest -q tests/test_frontend_lowering.py
```

## lit test: TLA IR Lowering to NPUIR

```bash
lit -sv csrc/mlir/build/tests/lit
```

## e2e test: End-to-End Verification

- This test additionally depends on an NPU environment and requires `torch` / `torch_npu` to be installed.

```bash
cd /path/to/catlass
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py --device 0
```

Specify the matrix shape, layout, and data type:

```bash
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py \
    --device 0 \
    --m 256 --n 512 --k 128 \
    --layout-a row --layout-b col \
    --dtype-a f16 --dtype-b f16 --dtype-c f32
```

On success the output contains:

```text
passed=True cache_key=<CACHE_KEY>
kernel.o=<CACHE_DIR>/<CACHE_KEY>/kernel.o
```

The available arguments are those shown by the script help:

```bash
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py --help
```

## Complete End-to-End Test

Run from the CATLASS repository root:

```bash
cd /path/to/catlass
bash tests/run_dsl_test.sh --device 0
```

The script loads CANN, checks AscendNPU-IR, builds CATLASS Python and runs the on-device regression tests in `tests/dsl_battery`.

Common variables are as follows:

| Variable                            | Meaning                                                  |
| ----------------------------------- | -------------------------------------------------------- |
| `ASCEND_HOME_PATH`                  | CANN Toolkit root directory; the script also tries to locate and load `set_env.sh` |
| `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` | Root of the shared AscendNPU-IR source and build tree, used as a fallback path after the in-repository submodule |
| `CATLASS_DSL_DIR`                   | Python subproject path; derived from the script location by default |
| `DEVICE_ID`                         | NPU device id; defaults to `1` and can be overridden by `--device` |
| `CATLASS_DSL_FORCE_RECOMPILE`       | Whether to force recompilation of runtime artifacts; the script sets it to `1` by default |

To see the cases and path resolution rules the script currently supports:

```bash
bash tests/run_dsl_test.sh --help
```
