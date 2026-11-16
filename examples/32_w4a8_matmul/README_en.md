# W4A8_Matmul Example Readme

## Code Organization

```text
├── 32_w4a8_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── gen_data.py
│   ├── w4a8.cpp
│   └── README.md
```

## Function

- Provides the Matmul implementation in W4A8 quantization mode.
- Matrix A is of `int8_t` type and matrix B is of `int4_t` type. After matrix B is converted to `int8_t`, matrix multiplication and per-tensor quantization are performed to output the `fp16_t (half)` type.

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).

- Run `gen_data.py` to generate test data. The test sample specifications are passed as command-line arguments. After the command is executed, a data directory is generated in the current path, containing the operator input data and the golden data used for accuracy verification.
- Execute the operator. Ensure that the input shape of the operator is the same as that of the test sample.

The following is a complete shell script example (executed in the sample directory `./examples/32_w4a8_matmul`):

```text
m=860
k=5712
n=4535
device=0

function build() {
    rm -rf ../../build
    rm -rf ../../output
    bash ../../scripts/build.sh 32_w4a8_matmul
}

function gen_data() {
    python3 gen_data.py $m $n $k
    echo "Data gen finished"
}

function run_kernel() {
    echo 'Case: m=' $m ' k=' $k ' n=' $n
    cd ../../output/bin/
    ./32_w4a8_matmul $m $n $k $device
}

build
gen_data
run_kernel
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
