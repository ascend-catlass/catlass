# 精度分析基础

## 写在前面

该文档主要说明CATLASS样例开发中精度分析的基础知识，包括样例精度的含义、精度比对方式，以及如何调用CATLASS已有的Golden函数计算标杆结果并进行精度比对。

## 样例精度的含义

在CATLASS算子开发中，"样例精度"指的是**NPU上算子实际计算结果与CPU上标杆（Golden）计算结果之间的一致性程度**。精度是衡量算子正确性的核心指标，只有精度达标的算子才能用于实际业务场景。

精度分析的基本流程为：

1. 在CPU侧使用相同输入数据，以高精度（如`float`甚至`double`）计算理论正确结果，称为**标杆（Golden）**；
2. 将NPU侧算子的实际输出与标杆进行比对；
3. 根据数据类型和计算规模，判断误差是否在允许范围内。

## 精度比对方式

CATLASS针对不同数据类型采用不同的精度比对策略。

### 浮点类型：相对误差校验

对于`half`（fp16）、`float`（fp32）、`bfloat16`等浮点类型，由于NPU硬件计算与CPU计算在舍入方式、累加顺序等方面存在差异，允许一定的相对误差。比对公式为：

$$
|actual - expected| \le rtol \times \max(1.0, |expected|)
$$

其中`rtol`（相对误差容忍度）根据计算次数`computeNum`动态调整：

| 计算次数 | rtol  | 说明                                   |
| -------- | ----- | -------------------------------------- |
| < 2048   | 1/256 | 计算量较小，误差累积少，容忍度更严格   |
| ≥ 2048   | 1/128 | 计算量较大，误差累积多，容忍度适当放宽 |

对于`bfloat16`类型，由于尾数位更少、精度更低，容忍度进一步放宽：

| 计算次数 | rtol  |
| -------- | ----- |
| < 2048   | 1/128 |
| ≥ 2048   | 1/64  |

### 浮点标杆的升精度计算

**浮点类型的标杆计算必须采用升精度策略**，这是保证精度分析可靠性的关键。具体来说：

- 即使算子输入/输出为`half`或`bfloat16`等低精度类型，标杆计算也应使用`float`（甚至`double`）作为累加器类型；
- 在`ComputeMatmul`等Golden函数中，每次乘加操作都会将操作数`static_cast<ElementGolden>`（通常为`float`）后再计算，避免CPU侧因低精度累加引入额外误差；
- 标杆结果存储为`float`类型，与NPU输出（可能为`half`）比对时，NPU输出会先转换为`float`再参与误差计算。

以`basic_matmul.cpp`为例，输入A、B和输出C均为`half`（fp16）类型，但标杆Golden使用`float`计算：

```cpp
// 输入输出均为 half 类型
std::vector<fp16_t> hostA(lenA);
std::vector<fp16_t> hostB(lenB);
std::vector<fp16_t> hostC(lenC);

// 标杆使用 float 类型进行升精度计算
std::vector<float> hostGolden(lenC);
golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
```

### 整数类型：二进制一致性校验

对于`int32_t`等整数类型，由于整数运算不存在舍入误差，要求**NPU输出与标杆完全一致（二进制一致）**。比对时直接检查差值是否为0：

```cpp
// int32_t 特化版本：要求完全一致
template<>
std::vector<uint64_t> CompareData(const std::vector<int32_t>& result, const std::vector<int32_t>& expect,
    uint32_t computeNum)
{
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if (std::abs(static_cast<int32_t>(result[i]) - expect[i]) != 0) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}
```

### 误差指标说明

CATLASS还提供了更精细的误差指标`ErrorMetrics`，用于评估NPU输出相对于同精度CPU计算结果的误差比率：

| 指标 | 全称                         | 含义                              |
| ---- | ---------------------------- | --------------------------------- |
| MARE | Max Absolute Relative Error  | 最大绝对相对误差比率（NPU / CPU） |
| MERE | Mean Absolute Relative Error | 平均绝对相对误差比率（NPU / CPU） |
| RMSE | Root Mean Squared Error      | 均方根误差比率（NPU / CPU）       |

这些指标将NPU输出和同精度CPU输出分别与高精度Golden比对，计算两者的误差比率。若比率在阈值范围内（默认MARE ≤ 5、MERE ≤ 1.5、RMSE ≤ 1.5），则认为精度合格。这用于判断NPU计算精度是否与同精度CPU计算处于同一水平。

## CATLASS Golden函数调用

CATLASS在`examples/common/golden.hpp`中提供了统一的Golden函数入口，该头文件聚合了以下模块：

| 头文件                    | 功能           |
| ------------------------- | -------------- |
| `golden/fill_data.hpp`    | 随机数据生成   |
| `golden/matmul.hpp`       | 矩阵乘标杆计算 |
| `golden/compare_data.hpp` | 精度比对       |
| `golden/conv2d.hpp`       | 卷积标杆计算   |

使用时只需包含`golden.hpp`即可：

```cpp
#include "golden.hpp"
```

所有Golden函数位于`Catlass::golden`命名空间下。

### 生成随机测试数据：FillRandomData

`FillRandomData`用于生成指定范围内的随机数据，支持多种数据类型：

```cpp
template <class Element, class ElementRandom>
void FillRandomData(std::vector<Element>& data, ElementRandom low, ElementRandom high);
```

- `Element`：目标数据类型（如`half`、`float`、`int8_t`等）
- `low` / `high`：随机值的上下界

使用示例：

```cpp
std::vector<fp16_t> hostA(lenA);
std::vector<fp16_t> hostB(lenB);
golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);  // 生成 [-5.0, 5.0] 范围内的随机 half 数据
golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);
```

对于`int8_t`类型有特化实现，使用整数随机生成避免浮点转换损失：

```cpp
std::vector<int8_t> hostA(lenA);
golden::FillRandomData<int8_t, int>(hostA, -128, 127);  // int8_t 使用整数范围
```

### 计算标杆结果：ComputeMatmul

`ComputeMatmul`在CPU侧以升精度方式计算矩阵乘法的理论正确结果：

```cpp
template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmul(
    const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden);
```

**关键设计**：模板参数`ElementGolden`独立于输入类型`ElementA`/`ElementB`，允许标杆使用更高精度类型。内部累加器类型为`ElementGolden`，每次乘加都通过`static_cast<ElementGolden>`升精度后再计算：

```cpp
accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
```

使用示例：

```cpp
// 输入为 half，标杆输出为 float（升精度）
std::vector<float> hostGolden(lenC);
golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
```

除`ComputeMatmul`外，Golden模块还提供了其他矩阵运算的标杆函数：

| 函数                       | 用途                                      |
| -------------------------- | ----------------------------------------- |
| `ComputeGemm`              | 通用矩阵乘（含alpha/beta缩放和C矩阵累加） |
| `ComputeGemv`              | 矩阵-向量乘                               |
| `ComputeBatchedMatmul`     | 批量矩阵乘                                |
| `ComputeGroupedMatmul`     | 分组矩阵乘                                |
| `ComputeGroupGemm`         | 分组通用矩阵乘                            |
| `ComputeMatmulElemWiseAdd` | 矩阵乘后Element-Wise加                    |

上述标杆函数若不满足业务场景需要，开发者也可自行增加新的标杆函数。

### 精度比对：CompareData

`CompareData`将NPU实际输出与标杆结果进行比对，返回错误元素的索引列表：

```cpp
template<class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(
    const std::vector<ElementResult>& result,
    const std::vector<ElementCompare>& expect,
    uint32_t computeNum);
```

- `result`：NPU算子实际输出
- `expect`：CPU标杆计算结果
- `computeNum`：计算次数（通常为K维大小），用于动态选择误差阈值
- 返回值：错误元素的索引列表，为空表示精度通过

使用示例：

```cpp
std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
if (errorIndices.empty()) {
    std::cout << "Compare success." << std::endl;
} else {
    std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
}
```

### 完整示例

以下摘自`examples/00_basic_matmul/basic_matmul.cpp`，展示了一个完整的精度分析流程：

```cpp
#include "golden.hpp"

// 1. 生成随机输入数据（half 类型）
std::vector<fp16_t> hostA(lenA);
std::vector<fp16_t> hostB(lenB);
golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

// 2. 将输入数据拷贝到Device，执行NPU算子...
// （省略Device侧内存分配、数据拷贝、算子执行等代码）

// 3. 将NPU输出拷贝回Host
std::vector<fp16_t> hostC(lenC);
ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

// 4. 计算CPU标杆（float 升精度）
std::vector<float> hostGolden(lenC);
golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

// 5. 精度比对
std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
if (errorIndices.empty()) {
    std::cout << "Compare success." << std::endl;
} else {
    std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
}
```

## 总结

CATLASS的精度分析遵循"**升精度计算标杆 + 分类型比对**"的核心原则：

| 数据类型                    | 标杆计算                   | 比对方式   | 误差容忍                              |
| --------------------------- | -------------------------- | ---------- | ------------------------------------- |
| 浮点（half/bfloat16/float） | 升精度（float/float/double累加） | 相对误差   | 计算次数 < 2048：1/256；≥ 2048：1/128 |
| 整数（int32_t等）           | 同精度                     | 二进制一致 | 差值必须为0                           |

开发者只需包含`golden.hpp`头文件，调用`FillRandomData`生成测试数据、`ComputeMatmul`（或其他标杆函数）计算标杆、`CompareData`进行比对，即可快速完成算子精度验证。
