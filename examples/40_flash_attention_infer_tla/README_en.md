# FlashAttentionInferTla Example Readme

## Code Organization

```text
├── 40_flash_attention_infer_tla
│   ├── CMakeLists.txt # CMake build configuration file
│   ├── gen_data.py
│   ├── kernel_common.hpp
│   ├── fai.cpp
│   ├── fai_kernel.cpp
│   ├── fai_tiling.cpp
│   └── README.md
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).

- Run `gen_data.py` to generate a test sample. The test sample needs to be input from the command line. After the command is executed, a `data` directory is generated in the current path, including the input data of the operator and the golden data used for precision verification.
- Then, execute the operator. Note that the input shape of the operator must be the same as the shape of the data generated in the first step.

The following is a complete shell script example.

```text
batch=1
qSeqlen=177
kvSeqlen=512
numHeads=1
kvHeads=1
headSize=128
isVariedLen=0
maskType=1
dtype="bf16"
cacheMode=1
layout_dtype=0
num_blocks=2048
inner_prec=0
lse_flag=0
device=0

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh 40_flash_attention_infer_tla
}

function gen_data() {
    python3 examples/40_flash_attention_infer_tla/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType "$dtype" $cacheMode $layout_dtype $num_blocks $inner_prec $lse_flag
    echo "Data gen finished"
}

function run_kernel() {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' D=' $headSize ' mask=' $maskType
    cd output/bin/
    ./40_flash_attention_infer_tla $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType --device $device --dtype $dtype
}

build
gen_data
run_kernel
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
