# Generalized Matmul Project Structure Description

## Project Structure

```shell
├── CMakeLists.txt
├── README.md
├── dynamic_optimized_matmul.cpp
├── impl
│   ├── kernel
│   │   ├── common_matmul_kernel.h
│   │   ├── ......
│   ├── scripts
│   │   ├── templates
│   │   │   ├── common_matmul_template.py
│   │   │   ├── ......
│   │   ├── utils
│   │   │   └── config.py
│   │   └── wrapper_code_gen.py
│   └──wrapper # Automatically generated
│       ├── common_matmul_kernel_half_layout00.cpp # Automatically generated
│       ├── common_matmul_kernel_half_layout01.cpp # Automatically generated
│       ├── common_matmul_kernel_half_layout10.cpp # Automatically generated
│       ├── common_matmul_kernel_half_layout11.cpp # Automatically generated
│       ├── ......
└── include
    ├── do_tiling_b16.h
    ├── dynamic_optimized_matmul.h
    ├── launch_map.h # Automatically generated
    ├── platform_info.h
    ├── select_kernel_b16.h
    ├── tiling_params.h
    └── utils.h

```

### Project Compilation

(1) Call the Python script to generate code, including the peripheral code of each template (that is, the files in the wrapper folder) and `launch_map.h` (containing `tilingKey` and specific mapping).

For example, the content of `common_matmul_kernel_half_layout00.cpp` is as follows:

```cpp

#include "kernel/common_matmul_kernel.h"
void LaunchCommonMatmulKernelHalfLayout00(aclrtStream& stream, uint64_t fftsAddr,
    uint8_t* dA, uint8_t* dB, uint8_t* dC, uint8_t* dW, uint8_t* dTilingParams, TilingParams& tilingParams)
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    using LayoutA = Catlass::layout::RowMajor;
    using LayoutB = Catlass::layout::RowMajor;
    using LayoutC = Catlass::layout::RowMajor;
    LaunchCommonMatmulKernel<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
        stream, fftsAddr, dA, dB, dC, dTilingParams, tilingParams);
}

size_t CommonMatmulKernelHalfLayout00GetWorkspaceSize(TilingParams& tilingParams)
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    using LayoutA = Catlass::layout::RowMajor;
    using LayoutB = Catlass::layout::RowMajor;
    using LayoutC = Catlass::layout::RowMajor;
    return CommonMatmulKernelGetWorkspaceSize<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(tilingParams);
}

```

The following is an example of the generated `launch_map.h` file:

```cpp

#ifndef LAUNCH_MAP_H
#define LAUNCH_MAP_H

#include <unordered_map>
#include <string>

#include "acl/acl.h"
#include "tiling_params.h"

#define DECLARE_KERNEL_FUNC(kernelName) \
    void Launch##kernelName(aclrtStream&, uint64_t, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, TilingParams&); \
    size_t kernelName##GetWorkspaceSize(TilingParams&);

DECLARE_KERNEL_FUNC(CommonMatmulKernelHalfLayout00)
DECLARE_KERNEL_FUNC(CommonMatmulKernelHalfLayout01)
DECLARE_KERNEL_FUNC(CommonMatmulKernelHalfLayout10)
DECLARE_KERNEL_FUNC(CommonMatmulKernelHalfLayout11)

std::unordered_map<uint64_t, void(*)(aclrtStream&, uint64_t,
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, TilingParams&)> launchKernelFuncMap = {
{ 0x0000000000000000, LaunchCommonMatmulKernelHalfLayout00 },
{ 0x0000000000000010, LaunchCommonMatmulKernelHalfLayout01 },
{ 0x0000000000000100, LaunchCommonMatmulKernelHalfLayout10 },
{ 0x0000000000000110, LaunchCommonMatmulKernelHalfLayout11 }
};

using GetWorkspaceFunc = size_t(*)(TilingParams& tilingParams);
std::unordered_map<uint64_t, GetWorkspaceFunc> getWorkspaceFuncMap = {
{ 0x0000000000000000, CommonMatmulKernelHalfLayout00GetWorkspaceSize },
{ 0x0000000000000010, CommonMatmulKernelHalfLayout01GetWorkspaceSize },
{ 0x0000000000000100, CommonMatmulKernelHalfLayout10GetWorkspaceSize },
{ 0x0000000000000110, CommonMatmulKernelHalfLayout11GetWorkspaceSize },
};

// only for print kernel Info
std::unordered_map<uint64_t, std::string> funcNameMap = {
{ 0x0000000000000000, "CommonMatmulKernelHalfLayout00" },
{ 0x0000000000000010, "CommonMatmulKernelHalfLayout01" },
{ 0x0000000000000100, "CommonMatmulKernelHalfLayout10" },
{ 0x0000000000000110, "CommonMatmulKernelHalfLayout11" }
};

#endif // LAUNCH_MAP_H

```

(2) After the compilation is complete, two files are generated. One is the binary executable file `output/bin/102_dynamic_optimized_matmul`, and the other is the static library file `output/shared_lib/lib/libdynamic_optimized_kernel.a`. The binary file calls the static library file.

### Running Process

![Generalized Matmul running process](https://raw.gitcode.com/weixin_42818618/picture0/raw/main/泛化Matmul运行流程.jpg)

The content of TilingKey is as follows:

```c++
/*
 * Bit field layout description (little-endian):
 * -------------------------------------------------------------------------
 * | Bit Range | Size | Field Name            | Description                |
 * |-----------|------|-----------------------|----------------------------|
 * | 0-3       | 4    | layoutTagC            | Layout tag for C matrix    |
 * | 4-7       | 4    | layoutTagB            | Layout tag for B matrix    |
 * | 8-11      | 4    | layoutTagA            | Layout tag for A matrix    |
 * | 12-15     | 4    | paddingTagC           | Padding tag for C matrix   |
 * | 16-19     | 4    | paddingTagB           | Padding tag for B matrix   |
 * | 20-23     | 4    | paddingTagA           | Padding tag for A matrix   |
 * | 24-51     | 28   | reserveBit            | Reserved for future use    |
 * | 52-55     | 4    | dtype                 | Data type specification    |
 * | 56-63     | 8    | templateKernelSerial  | Template kernel serial ID  |
 * -------------------------------------------------------------------------
 */

union TilingKey {
    uint64_t value;
    struct {
        uint64_t layoutTagC : 4;  // 0-3
        uint64_t layoutTagB : 4;  // 4-7
        uint64_t layoutTagA : 4;  // 8-11
        uint64_t paddingTagC : 4; // 12-15
        uint64_t paddingTagB : 4; // 16-19
        uint64_t paddingTagA : 4; // 20-23
        uint64_t reserveBit : 28; // 24-51 May be used in the future
        uint64_t dtype : 4;       // 52-55
        uint64_t templateKernelSerial : 8; // 56-63
    } bits;
    ......
}
```

Set the TilingKey based on the information obtained after `DoTiling` and `SelectKernel` are used, and match the corresponding Matmul function based on the TilingKey.

## 2 Instructions

```cpp

// 1. Input the shape information and construct the tilingParams structure.
TilingParams tilingParams{m, n, k, layoutTagA, layoutTagB, layoutTagC};
// 2. This function consists of two phases:
// (1) Calculate the tiling parameters based on the shape information in tilingParams.
// (2) Select the template based on the shape information and the tiling parameters obtained in the previous step.
DoTilingAndSelectKernel<fp16_t>(tilingParams, platformInfo);

// 3. (Optional) Print the parameters of the tilingParams structure.
PrintTilingParams<fp16_t>(tilingParams, platformInfo);

// 4. Obtain the required workspace size.
size_t workspaceSize = DynamicOptimizedMatmulGetWorkspace(tilingParams);

// 5. Allocate device memory.
ACL_CHECK(aclrtMalloc((void **)&dA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMalloc((void **)&dB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMalloc((void **)&dC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMalloc((void **)&dTilingParams, sizeof(TilingParams), ACL_MEM_MALLOC_HUGE_FIRST));
if (workspaceSize > 0) {
    ACL_CHECK(aclrtMalloc((void **)&dW, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
// 6. Copy the TilingParams structure to the device memory.
 ACL_CHECK(aclrtMemcpy(
        dTilingParams, sizeof(TilingParams), &tilingParams, sizeof(TilingParams), ACL_MEMCPY_HOST_TO_DEVICE));

// 7. Obtain fftsAddr.
uint64_t fftsAddr{0};
uint32_t fftsLen{0};
RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

// 8. Perform Matmul computation.
ExecuteDynamicOptimizedMatmul(stream, fftsAddr, dA, dB, dC, dW, dTilingParams, tilingParams);
ACL_CHECK(aclrtSynchronizeStream(stream));
// 9. Obtain the computation result.
ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
```

## 3 Implementation Principles

### DoTiling Implementation

In this case, DoTiling determines the block size on L1 based on the shape (M, N, K, LayoutA, and LayoutB).

The rules are as follows:

1. First, ensure that the instruction transfer bandwidth is fully utilized. The key point is that the basic block parameters in the stride direction must be 512-byte aligned.
2. Load balancing is preferred.
3. The number of computation rounds should be minimized.

### SelectKernel Implementation

```c++
bool PaddingMatmulB16Handler(TilingParams &params, PlatformInfo& platformInfo)
{
    uint8_t kernelSerial = 2;
    if (params.paddingTagA || params.paddingTagB || params.paddingTagC) {
        params.tilingKey.SetTilingKey(kernelSerial,
            params.layoutTagA, params.layoutTagB, 0, params.paddingTagA, params.paddingTagB, params.paddingTagC);
        return true;
    }
    return false;
}

bool CommonMatmulB16Handler(TilingParams &params, PlatformInfo& platformInfo)
{
    uint8_t kernelSerial = 0;
    uint32_t taskBlocks = CeilDiv(params.m, params.m1) * CeilDiv(params.n, params.n1);
    params.blockDim = taskBlocks > platformInfo.coreNum ? platformInfo.coreNum : taskBlocks;

    // kernelSerial, layoutTagA, layoutTagB, layoutTagC, paddingTagA, paddingTagB, paddingTagC, dtype(default 0).
    params.tilingKey.SetTilingKey(kernelSerial, params.layoutTagA, params.layoutTagB, 0, 0, 0, 0);
    return true;
}

using HandlerPtr = bool (*)(TilingParams& tilingParams, PlatformInfo& platformInfo);
HandlerPtr handlers[] = {
    SmallMatmulB16Handler,
    PaddingMultiCoreSplitkMatmulB16Handler,
    PaddingMatmulB16Handler,
    CommonMatmulB16Handler
};

for (auto handler : handlers) {
    if (handler(tilingParams, platformInfo)) {
        break;
    }
}
```

Each template is configured with its own conditions. The templates are traversed in a specific order. If the current shape meets the conditions of the current template, the current template is used for computation. Otherwise, the next template is traversed until a template that can process the current shape is found.
