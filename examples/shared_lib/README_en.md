# Packaging as a Shared Library

Sometimes, we want to add operators from the template library to an existing mature project to achieve computational acceleration without significantly modifying the project's build system. To this end, we can compile the template library operators into a shared library so that they can be easily called within the existing project.

## Code Structure

```bash
examples/shared_lib
├── include
│   └── catlass_kernel.h        # Header file
├── src
│   ├── common
│   │   └── common.hpp          # Common header file, reserved for template functions shared by multiple kernels
│   └── kernels                 # Operator implementation
│       ├── basic_matmul.cpp
│       └── ...
└── basic_matmul_shared_lib.cpp # CATLASS-style example usage file
```

## Build Artifact Structure

```bash
output
├── bin
│   └── basic_matmul_shared_lib         # Example usage executable
└── shared_lib
    ├── include
    │   └── catlass_kernel.h            # Header file
    └── lib
        ├── libcatlass_kernel.so        # Dynamic link library
        └── libcatlass_kernel_static.a  # Static link library
```

## Instructions

Call the interfaces provided in the header file directly:

```cpp
#include "catlass_kernel.h"
// ...
    CatlassKernel::KernelInfo kernelInfo;
    kernelInfo.inputAddr = {reinterpret_cast<uint8_t *>(deviceA), reinterpret_cast<uint8_t *>(deviceB)};
    kernelInfo.outputAddr = {reinterpret_cast<uint8_t *>(deviceC)};
    kernelInfo.inputDataType = ACL_FLOAT16;
    kernelInfo.outputDataType = ACL_FLOAT16;
    kernelInfo.m = m;
    kernelInfo.n = n;
    kernelInfo.k = k;

    CatlassKernel::BasicMatmul(aicCoreNum, stream, kernelInfo);
// ...
```

For a detailed sample code implementation, refer to [basic_matmul_shared_lib.cpp](./basic_matmul_shared_lib.cpp).

## Extension Description

Assume that the operator to be added is `custom_matmul`.

### Operator Kernel Implementation

Add the corresponding files and code as follows:

- Create `custom_matmul.hpp` in the `src/kernels` folder to implement the core operator logic.

```cpp
#include "catlass/catlass.hpp"
// catlass header file...

using namespace Catlass;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
CATLASS_GLOBAL
void custom_matmul(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC
    // Define input parameters as required...
)
{
    // Use the CATLASS API to define the operator...
}
```

- Create `custom_matmul.cpp` in the `src/kernels` folder to implement the host-side wrapper interface. This interface processes the execution arguments and invokes the device kernel via the kernel launch chevron syntax.

```cpp
// ...
void CustomMatmul(uint32_t blockNum, aclrtStream stream, kernelInfo kernelInfo) {
    Catlass::GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{kernelInfo.m, kernelInfo.k};
    LayoutB layoutB{kernelInfo.k, kernelInfo.n};
    LayoutC layoutC{kernelInfo.m, kernelInfo.n};
    custom_matmul<<<blockNum, nullptr, stream>>>(problemShape,
        kernelInfo.inputAddr.at(0), layoutA,
        kernelInfo.inputAddr.at(1), layoutB,
        kernelInfo.outputAddr.at(0), layoutC);
}
// ...
```

The parameter descriptions are outlined below:

| Parameter    | Type          | Purpose                                                                                               |
| ------------ | ------------- | ----------------------------------------------------------------------------------------------------- |
| `blockNum`   | `uint32_t`    | Specifies the AI Core count.                                                                          |
| `stream`     | `aclrtStream` | NPU Stream.                                                                                           |
| `kernelInfo` | `KernelInfo`  | Contains data addresses and shape details for operator execution, such as the m, n, and k dimensions. |

You can modify the parameters as required.

- Add the host wrapper prototype from `custom_matmul.cpp` to `include/catlass_kernel.h` to expose it for external calls.

```cpp
// ...
void CustomMatmul(uint32_t blockNum, aclrtStream stream, kernelInfo kernelInfo);
// ...
```

- If you add multiple operators that share identically defined template functions, duplicate symbol errors may occur during the linking stage. To resolve this, you can define these functions in `inline` form within the shared `common` path.

### Compilation

```bash
bash scripts/build.sh shared_lib
# Compile the executable file containing the example usage code
bash scripts/build.sh -DCATLASS_BUILD_USAGE shared_lib
```

## Precautions

- This example is used only as a reference for compiling CATLASS operators into a shared library or static library. To ensure code simplicity, generalization is not supported, such as multiple operators and platforms.
- Currently, four reference operators are provided as examples:
  - BasicMatmul: Basic matrix multiplication, illustrating type-templatized implementation methods.
  - GroupedMatmul: Grouped matrix multiplication, providing examples of grouped inputs and outputs.
  - OptimizedMatmul: Optimized matrix multiplication, providing an example of CV fusion pipelines.
  - ConvBias: Convolution operator logic.
- The example supports only the following products:
  - `Atlas A2 and Atlas A3 series products` (`2201` architecture)
