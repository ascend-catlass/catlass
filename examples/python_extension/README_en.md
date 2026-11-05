# Python Extension

To help developers use the CATLASS operator, the code repository provides examples of using Python to call the CATLASS operator based on pybind11 and torch.

## Code Structure

```bash
python_extension
├── CMakeLists.txt                      # CMake configuration file
├── README.md                           # Documentation
├── pyproject.toml                      # Project configuration file
├── setup.py                            # Python packaging script
├── src
│   ├── bindings
│   │   ├── pybind_bindings.cpp         # pybind11 binding file
│   │   └── torch_bindings.cpp          # torch binding file
│   ├── include
│   │   └── wrapper
│   │       └── catlass_kernel_wrapper.h   # Wrapper header file
│   └── wrapper
│       └── catlass_kernel_wrapper.cpp      # Wrapper file of the CATLASS operator
└── torch_catlass
    └── __init__.py                     # Initialization entry, used for packaging
tests
└── test_python_extension.py        # Test script
```

## Build Artifact Structure

```bash
output/python_extension
├── libcatlass_torch.so                             # torch dynamic link library
└── torch_catlass-0.1.0.20250330120000.cp310-cp310-linux-x86_64.whl  # Wheel package of the pybind11 dynamic link library
```

## Instructions

- Assume that developers have added the implementation and entry of the required operator to `shared_lib/`.

### pybind Interface Implementation

The input parameter of pybind is `at::Tensor` instead of the address pointer (`GM_ADDR`) in Ascend C. Therefore, the data transferred from the Python side needs to be processed.
The main steps are filling in the running information parameters based on the input tensor information and allocating the output memory.
This part is related to the operator parameters themselves. For details, see the existing `BasicMatmul` implementation.

### Compilation

Once all components are complete:

- Use `bash scripts/build.sh python_extension` to compile the pybind extension.
- Use `bash scripts/build.sh torch_library` to compile the torch extension.

The compilation environment is identical to the main [README](../../README.md), with the addition of the following Python dependencies:

- Required:
  - `pybind11`
  - gcc _9.0 or later_
  - torch _2.1 or later recommended_
- `TorchNPU` (the latest version matching your `torch` and `CANN` installation). See [Ascend/pytorch](https://gitcode.com/ascend/pytorch).
- Optional:
  - `pybind11-stubgen`

### Installation

- For the `torch` extension, you only need to add the following code snippet before using the operator:

```python
torch.ops.load_library("output/python_extension/libcatlass_torch.so")
```

- pybind Extension: The build artifact is a wheel package. Install it by running `pip install torch_catlass-xxxxx.whl`.

### Run

```python
import torch_catlass
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

class CatlassTest(TestCase):
    def test_basic_matmul(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        result = torch_catlass.basic_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)
    def test_basic_matmul_torch_lib(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        torch.ops.load_library("../../output/python_extension/libcatlass_torch.so") # Ensure that the correct path is loaded.
        result = torch.ops.CatlassTorch.basic_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)

if __name__ == "__main__":
    run_tests()
```

## Precautions

- This example is used only as a reference for integrating the CATLASS operator into msopgen. To ensure code simplicity, generalization is not supported, such as multiple operators or platforms.
- Currently, four reference operators are provided as examples:
  - `BasicMatmul`: Basic matrix multiplication, illustrating type-templatized implementation methods.
  - `GroupedMatmul`: Grouped matrix multiplication, providing examples of grouped inputs and outputs.
  - `OptimizedMatmul`: Optimized matrix multiplication, providing an example of CV fusion pipelines.
  - `ConvBias`: Convolution operator logic.
- The example supports only the following products:
  - `Atlas A2 and Atlas A3 series products` (`2201` architecture)
