# Host侧代码组装详解

## 概述

本文以`BasicMatmul`算子为例，详细讲解如何基于CATLASS模板库组装Host侧代码，实现完整的矩阵乘法运算。Host侧代码主要负责环境初始化、资源管理、数据传输和算子调用等功能。

## Host侧代码结构

Host侧代码通常包含以下几个核心部分：

1. 环境初始化与资源申请
2. 输入数据准备
3. 设备内存申请与数据拷贝
4. 算子参数配置与调用
5. 结果数据拷贝与验证
6. 资源释放

下面以`BasicMatmul`算子为例，详细讲解每个部分的实现。

## 完整示例代码

创建`examples/basic_matmul`文件夹和`examples/basic_matmul/basic_matmul.cpp`文件，以下为该cpp文件的编写示例。

### 头文件与配置

```cpp
// 引入必要的头文件
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;

using Options = GemmOptions;

```

### 核心实现

```cpp
static void Run(const Options &options) {
    /* 第一步，流初始化与设备侧空间申请 */
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // 初始化matmul矩阵的shape参数
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    // 矩阵A的元素数量为m*k，矩阵B的元素数量为k*n，矩阵C的元素数量为m*n
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    // 根据矩阵元素数量和数据类型计算矩阵占用内存大小
    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    // 初始化数据排布格式，RowMajor表示行优先
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    // 初始化输入数据
    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    // 申请A矩阵在device上的内存，并将A矩阵拷贝至device
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    // 申请B矩阵在device上的内存，并将B矩阵拷贝至device
    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // 申请C矩阵在device上的内存
    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    // 获取当前硬件核心数量
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    /* 第二步，选择优化策略 */
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

    // 定义tiling切分策略
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    /* 第三步，选择数据类型，并组装模板样例组件 */
    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;

    // 定义Block层进行矩阵乘计算的组件
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // 配置Block调度器，指定Block粒度的swizzle次序
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // 指定kernel
    using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

    // 定义Device层适配器
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};

    /* 第四步，执行模板样例 */
    // 定义适配器对象
    MatmulAdapter matmulOp;
    // 判断kernel对相关参数可执行
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
    // 初始化
    matmulOp.Initialize(arguments, deviceWorkspace);
    // 调用执行
    matmulOp(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    // 将输出数据搬出
    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    // 计算精度标杆并与输出数据比对
    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    /* 第五步，释放资源 */
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

## 核心实现代码详解

### 环境初始化与资源申请

这部分代码主要完成以下工作：

- 初始化ACL环境
- 设置设备ID
- 创建计算流
- 计算矩阵维度和内存大小
- 定义数据排布格式

```cpp
aclrtStream stream{nullptr};
ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(options.deviceId));
ACL_CHECK(aclrtCreateStream(&stream));

// 计算矩阵维度和内存大小
uint32_t m = options.problemShape.m();
uint32_t n = options.problemShape.n();
uint32_t k = options.problemShape.k();
size_t lenA = static_cast<size_t>(m) * k;
size_t lenB = static_cast<size_t>(k) * n;
size_t lenC = static_cast<size_t>(m) * n;
size_t sizeA = lenA * sizeof(fp16_t);
size_t sizeB = lenB * sizeof(fp16_t);
size_t sizeC = lenC * sizeof(fp16_t);

// 定义数据排布格式
using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;
LayoutA layoutA{m, k};
LayoutB layoutB{k, n};
LayoutC layoutC{m, n};
```

### 输入数据准备

这部分代码主要完成以下工作：

- 在Host侧创建输入数据缓冲区
- 填充随机数据

```cpp
std::vector<fp16_t> hostA(lenA);
std::vector<fp16_t> hostB(lenB);
golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);
```

### 设备内存申请与数据拷贝

这部分代码主要完成以下工作：

- 在Device侧申请内存
- 将Host侧数据拷贝到Device侧

```cpp
// 申请A矩阵在device上的内存，并将A矩阵拷贝至device
uint8_t *deviceA{nullptr};
ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

// 申请B矩阵在device上的内存，并将B矩阵拷贝至device
uint8_t *deviceB{nullptr};
ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

// 申请C矩阵在device上的内存
uint8_t *deviceC{nullptr};
ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
```

### 算子参数配置与调用

这部分代码主要完成以下工作：

- 选择架构和调度策略（Atlas A2/A3 产品设置 `ArchTag = Arch::AtlasA2`，Ascend 950PR/950DT 产品使用 `ArchTag = Arch::Ascend950`）
- 定义Tile形状
- 组装Block层和Kernel层组件
- 初始化Device层适配器
- 执行算子

```cpp
// 选择调度策略
using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

// 定义tiling切分策略
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;

// 选择数据类型，并组装模板样例组件
using AType = Gemm::GemmType<half, LayoutA>;
using BType = Gemm::GemmType<half, LayoutB>;
using CType = Gemm::GemmType<half, LayoutC>;

// 定义Block层进行矩阵乘计算的组件
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;

// 配置Block调度器
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// 指定kernel
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

// 定义Device层适配器
using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};

// 执行模板样例
MatmulAdapter matmulOp;
// matmulOp.CanImplement(arguments); //BasicMatmul 不涉及参数校验，可以不执行CanImplement并校验返回值
size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
uint8_t *deviceWorkspace = nullptr;
if (sizeWorkspace > 0) {
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
}
matmulOp.Initialize(arguments, deviceWorkspace);
matmulOp(stream, aicCoreNum);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

### 结果数据拷贝与验证

这部分代码主要完成以下工作：

- 将Device侧计算结果拷贝到Host侧
- 计算标杆结果并与结果进行比较

```cpp
// 将输出数据搬出
std::vector<fp16_t> hostC(lenC);
ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

// 计算精度标杆并与输出数据比对
std::vector<float> hostGolden(lenC);
golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
if (errorIndices.empty()) {
    std::cout << "Compare success." << std::endl;
} else {
    std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
}
```

### 资源释放

这部分代码主要完成以下工作：

- 释放Device侧内存
- 销毁计算流
- 重置设备
- 释放ACL环境

```cpp
// 释放资源
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

## 编译与运行

### 编译

创建`examples/basic_matmul/CMakeLists.txt`文件：

```cmake
# CMakeLists.txt
set_source_files_properties(basic_matmul.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(
    basic_matmul
    cube
    basic_matmul.cpp
)
```

在[examples/CMakeLists.txt](../../../examples/CMakeLists.txt)中将新增示例加入编译清单，注意根据样例支持的产品形态确认对应编译清单：

```diff
set(EXAMPLE_ATLASA2
    00_basic_matmul
    # ...
+   basic_matmul
)
```

完成CANN包安装和CANN环境使能后，执行编译命令（若为 Ascend 950PR/950DT，需要添加`-DCATLASS_ARCH=3510`）：

```bash
bash scripts/build.sh basic_matmul
```

### 运行

运行参数说明：

| 参数 | 含义 | 类型 | 取值范围 |
|------|------|------|---------|
| m | 矩阵 A 的行数 | 正整数 | ≥ 1 |
| n | 矩阵 B 的列数 | 正整数 | ≥ 1 |
| k | 矩阵 A 的列数 / 矩阵 B 的行数 | 正整数 | ≥ 1 |
| device_id | 设备 ID | 非负整数 | 0 ~ 设备总数-1，默认 0 |

```bash
cd output/bin
./basic_matmul 256 512 1024 0
```

获得回显

```text
Compare success.
```
