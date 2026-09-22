# Host-Side Code Assembly

## Overview

This document uses the `BasicMatmul` operator as an example to describe how to assemble host-side code based on the CATLASS template library to implement matrix multiplication. The host-side code is primarily responsible for environment initialization, resource management, data transfer, and operator execution.

## Host-Side Code Structure

The host-side code consists of the following core phases:

1. Environment initialization and resource allocation
2. Input data preparation
3. Device memory allocation and data copy
4. Operator parameter configuration and invocation
5. Result data copy and verification
6. Resource deallocation

The following sections use the `BasicMatmul` operator as an example to detail the implementation of each phase.

## Complete Sample Code

Create the `examples/basic_matmul` directory and the `examples/basic_matmul/basic_matmul.cpp` file. The following is an implementation example for the source file.

### Header Files and Configuration

```cpp
// Include necessary header files.
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "helper.hpp"
#include "golden.hpp"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

using namespace Catlass;

// Parse input parameters.
struct Options {
    const std::string HELPER = "basic_matmul m n k [device_id]";

    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv) {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};
```

### Core Implementation

```cpp
static void Run(const Options &options) {
    /* Step 1: Initialize streams and allocate space on the device */
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Initialize the shape parameters of the matmul matrix
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    // Elements counts: Matrix A (m × k), Matrix B (k × n), and Matrix C (m × n)
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    // Calculate memory sizes based on element counts and data type size
    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    // Initialize the tensor layout formats (RowMajor)
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    // Initialize host input buffers
    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    // Allocate memory for matrix A on the device and copy matrix A to the device
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate memory for matrix B on the device and copy matrix B to the device
    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate device memory for matrix C
    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the current hardware core count
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    /* Step 2: Select optimization policy */
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

    // Define the tiling strategy
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    /* Step 3: Select data type and assemble template sample components */
    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;

    // Define the component for performing matrix multiplication at the Block layer
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // Configure the Block scheduler and specify the swizzle order at Block granularity
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // Specify the kernel
    using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

    // Define the Device layer adapter
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};

    /* Step 4: Execute template sample */
    // Define adapter object
    MatmulAdapter matmulOp;
    // Check whether the kernel can execute related parameters
    if (matmulOp.CanImplement(arguments) == Status::kInvalid) {
        std::cerr << "matmulOp cannot implement current arguments." << std::endl;
        return;
    }
    size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // Initialize
    matmulOp.Initialize(arguments, deviceWorkspace);
    // Call and execute
    matmulOp(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    // Move out the output data
    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    // Calculate precision benchmark and compare with output data
    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    /* Step 5: Release resources */
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }

    try {
        Run(options);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
```

## Detailed Explanation of Core Implementation Code

### Environment Initialization and Resource Allocation

This section of the code performs the following tasks:

- Initializes the AscendCL environment
- Sets the device ID
- Creates a stream for computation
- Calculates matrix dimensions and memory sizes
- Defines the data layout format

```cpp
aclrtStream stream{nullptr};
ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(options.deviceId));
ACL_CHECK(aclrtCreateStream(&stream));

// Calculate matrix dimensions and memory sizes
uint32_t m = options.problemShape.m();
uint32_t n = options.problemShape.n();
uint32_t k = options.problemShape.k();
size_t lenA = static_cast<size_t>(m) * k;
size_t lenB = static_cast<size_t>(k) * n;
size_t lenC = static_cast<size_t>(m) * n;
size_t sizeA = lenA * sizeof(fp16_t);
size_t sizeB = lenB * sizeof(fp16_t);
size_t sizeC = lenC * sizeof(fp16_t);

// Defines the data layout format
using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;
LayoutA layoutA{m, k};
LayoutB layoutB{k, n};
LayoutC layoutC{m, n};
```

### Input Data Preparation

This section of the code performs the following tasks:

- Creates input data buffers on the host side
- Fills the buffers with random data

```cpp
std::vector<fp16_t> hostA(lenA);
std::vector<fp16_t> hostB(lenB);
golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);
```

### Device Memory Allocation and Data Copy

This section of the code performs the following tasks:

- Allocates memory on the Device side
- Copies data from the Host side to the Device side

```cpp
// Allocate memory for matrix A on the device and copy matrix A to the device
uint8_t *deviceA{nullptr};
ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

// Allocate memory for matrix B on the device and copy matrix B to the device
uint8_t *deviceB{nullptr};
ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

// Allocate device memory for matrix C
uint8_t *deviceC{nullptr};
ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
```

### Operator Parameter Configuration and Invocation

This section of the code performs the following tasks:

- Selects the architecture and dispatch policy
- Defines the tile shapes
- Assembles the Block layer and Kernel layer components
- Initializes the Device layer adapter
- Executes the operator

```cpp
// Select a dispatch policy
using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

// Define the tiling strategy
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;

// Select data types and assemble the template example components
using AType = Gemm::GemmType<half, LayoutA>;
using BType = Gemm::GemmType<half, LayoutB>;
using CType = Gemm::GemmType<half, LayoutC>;

// Define the component for performing matrix multiplication at the Block layer
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;

// Configure the Block scheduler
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// Specify the kernel
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

// Define the Device layer adapter
using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};

// Run the template example
MatmulAdapter matmulOp;
matmulOp.CanImplement(arguments);
size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
uint8_t *deviceWorkspace = nullptr;
if (sizeWorkspace > 0) {
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
}
matmulOp.Initialize(arguments, deviceWorkspace);
matmulOp(stream, aicCoreNum);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

### Result Data Copy and Verification

This section of the code performs the following tasks:

- Copies the compute results from the device side to the host side
- Calculates the benchmark reference results and compares them with the output data

```cpp
// Move out the output data
std::vector<fp16_t> hostC(lenC);
ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

// Calculate precision benchmark and compare with output data
std::vector<float> hostGolden(lenC);
golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
if (errorIndices.empty()) {
    std::cout << "Compare success." << std::endl;
} else {
    std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
}
```

### Resource Deallocation

This section of the code performs the following tasks:

- Frees device-side memory buffers
- Destroys the computational stream
- Resets the device
- Finalizes the AscendCL environment

```cpp
// Release resources
ACL_CHECK(aclrtFree(deviceA));
ACL_CHECK(aclrtFree(deviceB));
ACL_CHECK(aclrtFree(deviceC));
if (sizeWorkspace > 0) {
    ACL_CHECK(aclrtFree(deviceWorkspace));
}
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(options.deviceId));
ACL_CHECK(aclFinalize());
```

## Build and Run

### Compilation

Create a `examples/basic_matmul/CMakeLists.txt` file:

```cmake
# CMakeLists.txt
set_source_files_properties(basic_matmul.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(
    basic_matmul
    cube
    basic_matmul.cpp
)
```

Add the new example to the compilation list in [examples/CMakeLists.txt](../../../examples/CMakeLists.txt):

```diff
set(EXAMPLE_ATLASA2
    00_basic_matmul
    # ...
+   basic_matmul
)
```

After the CANN package is installed and the CANN environment is enabled, run the following command:

```bash
bash scripts/build.sh basic_matmul
```

### Run

```bash
cd output/bin
./basic_matmul 256 512 1024 0
```

Obtain the command output:

```bash
Compare success.
```
