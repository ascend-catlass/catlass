# Precision Analysis Basics

## Before You Start

This document describes the basics of precision analysis in CATLASS sample development, including the meaning of sample precision, precision comparison methods, and how to use existing CATLASS golden functions to compute golden results and perform precision comparison.

## Sample Precision Definition

In CATLASS operator development, "sample precision" refers to **the extent of consistency between the actual computation result of an operator on the NPU and the golden computation result on the CPU**. Precision is a core metric for operator correctness. Only operators that meet the precision requirements can be used in real-world applications.

The basic process of precision analysis is as follows:

1. On the CPU, use the same input data to compute the theoretically correct result with high precision (such as `float` or even `double`). This result is called the golden result.
2. Compare the actual output of the operator on the NPU with the golden result.
3. Determine whether the error is within the allowed tolerance based on the data type and computation scale.

## Precision Comparison Methods

CATLASS uses different precision comparison methods for different data types.

### Floating-Point Types: Relative Error Validation

For floating-point types such as `half` (fp16), `float` (fp32), and `bfloat16`, slight relative error is allowed due to differences in rounding modes and accumulation order between NPU hardware computation and CPU computation. The comparison formula is:

$$
|actual - expected| \le rtol \times \max(1.0, |expected|)
$$

where `rtol` (relative tolerance) is dynamically adjusted based on the number of computations `computeNum`:

| computeNum| rtol | Description|
| --- | --- | --- |
| < 2,048| 1/256 | Small computation volume, less error accumulation, stricter tolerance|
| ≥ 2,048| 1/128 | Large computation volume, more error accumulation, looser tolerance|

For `bfloat16`, which has fewer mantissa bits and lower precision, the tolerance is further loosened:

| computeNum| rtol |
| --- | --- |
| < 2,048| 1/128 |
| ≥ 2,048| 1/64 |

### Higher-Precision Computation for Floating-Point Golden Functions

**Floating-point golden functions must use higher-precision computation**. This is the key to ensuring reliable precision analysis. Specifically,

- Even if the operator input/output is of a low-precision type such as `half` or `bfloat16`, the golden function should use `float` (or even `double`) as the accumulator type.
- In golden functions such as `ComputeMatmul`, each multiply-add operation first casts the operands to `ElementGolden` (typically `float`) using `static_cast`, and then performs the computation. This prevents additional errors introduced by low-precision accumulation on the CPU.
- The golden result is stored of `float` type. When compared with the NPU output (which may be `half`), the NPU output is first converted to `float` before error calculation.

Take `basic_matmul.cpp` as an example. The inputs A and B and the output C are all of the `half` (fp16) type, but the golden uses `float` for computation.

```cpp
// Input and output are half
std::vector<fp16_t> hostA(lenA);
std::vector<fp16_t> hostB(lenB);
std::vector<fp16_t> hostC(lenC);

// The golden uses float for higher-precision computation
std::vector<float> hostGolden(lenC);
golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
```

### Integer Types: Bitwise Identity Validation

For integer types such as `int32_t`, since integer operations involve no rounding errors, the NPU output must be **bitwise identical** to the golden result. The comparison directly checks whether the difference is zero:

```cpp
// int32_t specialization: Requires bitwise identity
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

### Error Metrics Description

CATLASS also provides more refined error metrics, `ErrorMetrics`, to evaluate the error ratio of NPU output compared with the same-precision CPU computation result.

| Metric| Full Name| Meaning|
| --- | --- | --- |
| MARE | Max Absolute Relative Error | Maximum absolute relative error ratio (NPU / CPU)|
| MERE | Mean Absolute Relative Error | Mean absolute relative error ratio (NPU / CPU)|
| RMSE | Root Mean Squared Error | Root mean squared error ratio (NPU / CPU)|

These metrics compare the NPU output and the CPU output against the higher-precision golden output, and calculate the error ratios between them. If the ratios are within the threshold (default: MARE ≤ 5, MERE ≤ 1.5, RMSE ≤ 1.5), the precision is considered acceptable. This determines whether the NPU computation precision is on par with the same-precision CPU computation.

## CATLASS Golden Function Call

CATLASS provides a unified golden function entry in `examples/common/golden.hpp`. This header file aggregates the following modules:

| Header File| Function|
| --- | --- |
| `golden/fill_data.hpp` | Random data generation|
| `golden/matmul.hpp` | Matrix multiplication golden computation|
| `golden/compare_data.hpp` | Precision comparison|
| `golden/conv2d.hpp` | Convolution golden computation|

Simply include `golden.hpp` to use these functions:

```cpp
#include "golden.hpp"
```

All golden functions are in the `Catlass::golden` namespace.

### Generating Random Test Data: FillRandomData

`FillRandomData` generates random data within a specified range. It supports multiple data types:

```cpp
template <class Element, class ElementRandom>
void FillRandomData(std::vector<Element>& data, ElementRandom low, ElementRandom high);
```

- `Element`: target data type (such as `half`, `float`, or `int8_t`)
- `low`/`high`: upper and lower bounds for random values

Example:

```cpp
std::vector<fp16_t> hostA(lenA);
std::vector<fp16_t> hostB(lenB);
golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);  // Generate random half data in [-5.0, 5.0]
golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);
```

For the `int8_t` type, there is a specialized implementation that uses integer random generation to avoid floating-point conversion loss:

```cpp
std::vector<int8_t> hostA(lenA);
golden::FillRandomData<int8_t, int>(hostA, -128, 127);  // Integer range used by int8_t
```

### Computing Golden Results: ComputeMatmul

`ComputeMatmul` computes the theoretically correct result of matrix multiplication on the CPU with higher precision:

```cpp
template<class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementGolden, class LayoutGolden>
void ComputeMatmul(
    const GemmCoord &problemShape,
    const std::vector<ElementA> &dataA, const LayoutA &layoutA,
    const std::vector<ElementB> &dataB, const LayoutB &layoutB,
    std::vector<ElementGolden> &dataGolden, const LayoutGolden &layoutGolden);
```

**Key design**: The template parameter `ElementGolden` is independent of the input type `ElementA`/`ElementB`, allowing the golden to use a higher precision type. The internal accumulator type is `ElementGolden`. Each multiply-add operation is performed after the precision is increased via `static_cast<ElementGolden>`.

```cpp
accumulator += static_cast<ElementGolden>(dataA[offsetA]) * static_cast<ElementGolden>(dataB[offsetB]);
```

Example:

```cpp
// The input is half, and the golden output is float (higher precision)
std::vector<float> hostGolden(lenC);
golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
```

In addition to `ComputeMatmul`, the golden module also provides other golden functions for matrix operations:

| Function| Purpose|
| --- | --- |
| `ComputeGemm` | General matrix multiplication (including alpha/beta scaling and matrix C accumulation)|
| `ComputeGemv` | Matrix-vector multiplication|
| `ComputeBatchedMatmul` | Batch matrix multiplication|
| `ComputeGroupedMatmul` | Grouped matrix multiplication|
| `ComputeGroupGemm` | Grouped general matrix multiplication|
| `ComputeMatmulElemWiseAdd` | Matrix multiplication followed by element-wise addition|

If the above golden functions do not meet the requirements of a specific need, you can also add new golden functions.

### Precision Comparison: CompareData

`CompareData` compares the actual output of the NPU with the golden result and returns the index list of error elements.

```cpp
template<class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(
    const std::vector<ElementResult>& result,
    const std::vector<ElementCompare>& expect,
    uint32_t computeNum);
```

- `result`: actual output of the NPU operator
- `expect`: golden result computed by CPU
- `computeNum`: number of computations (typically the K dimension size) for dynamically selecting the error tolerance
- Return value: list of indexes of error elements. If the list is empty, the precision test is passed.

Example:

```cpp
std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
if (errorIndices.empty()) {
    std::cout << "Compare success." << std::endl;
} else {
    std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
}
```

### Complete Sample

The following, taken from `examples/00_basic_matmul/basic_matmul.cpp`, demonstrates a complete precision analysis process:

```cpp
#include "golden.hpp"

// 1. Generate random input data (half type)
std::vector<fp16_t> hostA(lenA);
std::vector<fp16_t> hostB(lenB);
golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

// 2. Copy the input data to the device and execute the NPU operator
// (Device memory allocation, data copy, operator execution, etc. omitted)

// 3. Copy the NPU output back to the host
std::vector<fp16_t> hostC(lenC);
ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

// 4. Compute the CPU golden result (using float for higher precision)
std::vector<float> hostGolden(lenC);
golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

// 5. Compare the precision
std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
if (errorIndices.empty()) {
    std::cout << "Compare success." << std::endl;
} else {
    std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
}
```

## Summary

CATLASS precision analysis follows the principle of "higher-precision golden computation + type-specific comparison":

| Data Type| Golden Computation| Comparison Method| Error Tolerance|
| --- | --- | --- | --- |
| Floating-point (half/float/bfloat16)| Higher precision (float/double accumulation)| Relative error| computeNum < 2,048: 1/256; ≥ 2,048: 1/128|
| Integer (such as int32_t)| Same precision| Bitwise identity| Difference must be zero|

You only need to include the `golden.hpp` header file, call `FillRandomData` to generate test data, call `ComputeMatmul` (or other golden functions) to compute the golden result, and call `CompareData` for comparison to quickly complete operator precision validation.
