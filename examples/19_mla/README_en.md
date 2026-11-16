# MLA Example Readme

## Code Organization

```text
├── 19_mla
│   ├── CMakeLists.txt # CMake build file
│   ├── gen_data.py
│   ├── kernel_common.hpp # Common variables and macros shared across kernel implementations
│   ├── main.cpp
│   ├── mla_kernel.cpp # MLA TP 2/4/8 template
│   ├── mla_kernel_tp1_spec.cpp # MLA TP 1 template
│   └── README.md
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Step 1: Execute `gen_data.py` with the command line to generate test vectors and validation assets.

```bash
# Execute in the ./examples/19_mla directory.
python gen_data.py 1 1 128 16 16 128 half
# Input parameters map to: batchSize, qSeqlen, kvSeqlen, qheadNum, numBlock, and blockSize.
# qSeqlen: Number of tokens to be inferred. Supported range is 1 to 4 (covers standard decode and MTP scenarios).
# kvSeqlen: Total sequence length of the input.
# blockSize: Currently accepts a fixed value of 128.
# The final parameter explicitly defines the data type: "half" or "bf16".
```

After the command is executed, a data directory is generated in the current path, containing the operator input data and the golden data used for accuracy verification.

```text
├── data
│   ├── block_table.bin
│   ├── golden.bin
│   ├── k.bin
│   ├── k_rope.bin
│   ├── kv_seqlen.bin
│   ├── q.bin
│   ├── q_ntokens.bin
│   ├── q_rope.bin
│   └── q_seqlen.bin
```

Step 2: Execute the operator. Note that the input shape of the operator must match the shape of the data generated in the first step.

```bash
# Compile the specified test case from the root of the CATLASS repository
bash scripts/build.sh 19_mla
cd output/bin
./19_mla 1 1 128 16 16 128
# Parameters must align with the data generator execution.
# Full syntax signature: batchSize, qSeqlen, kvSeqlen, qheadNum, numBlock, blockSize [--dtype DTYPE --datapath DATA_PATH --device DEVICE_ID] Options: --dtype defaults to half, --datapath defaults to ../../examples/19_mla/data, --device defaults to 0.
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```

## Accuracy regression

`gen_data.py` provides the shared attention reference. Suite generation uses its FP64
mode, while the original single-case command retains its FP32 reference. The generator
only creates data; it does not execute kernels, compare outputs or maintain tiling keys.
Install `numpy` and `ml_dtypes` before running it.

```bash
# Run from the repository root; --suite smoke generates only the basic cases.
python examples/19_mla/gen_data.py --suite full --binary ./output/bin/19_mla --output-dir ./mla_test_data --device 0
./output/bin/19_mla --test ./mla_test_data --device 0
# Run an individual case.
./output/bin/19_mla --test ./mla_test_data --device 0 --case bf16_repeated_underflow
```

The following successful output was recorded on an Ascend 910B2 with 24 AIC cores and the default seed.
`...` indicates omitted log lines. Data generation prints `Generated` for each case and writes
`cases.txt` to the output directory when complete:

```text
Generated half_zero
Generated half_cancel
...
Generated bf16_tail_tasks
```

A successful full run reports `[       OK ]` for every case and ends with:

```text
[ RUN      ] half_zero
...
Compare success.
[       OK ] half_zero
...
[       OK ] bf16_tail_tasks
MLA tests: 37 run, 0 failed
```

Running `bf16_repeated_underflow` individually produces:

```text
[ RUN      ] bf16_repeated_underflow
TILING_FORMERTASKNUM = 1
TILING_TAILTASKNUM = 0
TILING_KVSPLIT = 1536
TILING_KVCORENUM = 1
tilingKey : 5
elements=65536 mismatches=0 max_abs=2.95331e-42 rmse=2.95331e-42
Compare success.
[       OK ] bf16_repeated_underflow
MLA tests: 1 run, 0 failed
```

`mismatches=0` means no elements exceed the tolerance for that case. `0 failed` means all executed
cases passed, and the process exits with status 0. Tiling parameters, element counts and error
values may vary with the device core count and generated data.

The generator queries the selected device's actual AIC core count to construct multiwave
and tail-task inputs. It writes `cases.txt` only after all references have been generated.
Regenerate the data when switching to a device with a different core count. The test
entry point in `mla.cpp` uses the existing tiling and kernel launch code, with elementwise
comparison from `common/golden`. Underflow cases additionally require nonzero outputs
with the expected sign. Numerical failures, unmet kernel coverage assertions and missing
or incorrectly sized input files produce a nonzero exit status. The original single-case
entry point also returns a nonzero status on accuracy failure.
