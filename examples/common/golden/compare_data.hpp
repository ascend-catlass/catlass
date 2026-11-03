/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "catlass/gemm_coord.hpp"

namespace Catlass::golden {

struct ErrorMetrics {
    bool passed;      // true if error ratios meet criteria
    double mareRatio; // Max Absolute Relative Error ratio(hostC / hostCpu)
    double mereRatio; // Mean Absolute Relative Error ratio(hostC / hostCpu)
    double rmseRatio; // Root Mean Squared Error ratio(hostC / hostCpu)
};

// Compute error metrics for NPU result vs CPU result against high-precision golden reference
// hostC: NPU output, hostCpu: same-precision CPU result, hostGolden: high-precision reference
template <class ElementC, class ElementCpu, class ElementGolden>
ErrorMetrics ComputeErrorMetrics(
    const std::vector<ElementC>& hostC, const std::vector<ElementCpu>& hostCpu,
    const std::vector<ElementGolden>& hostGolden, double mareThreshold = 5, double mereThreshold = 1.5,
    double rmseThreshold = 1.5)
{
    const double epsilon = 1e-7;
    size_t n = hostC.size();
    if (n == 0 || hostCpu.size() != n || hostGolden.size() != n) {
        return {false, 0.0, 0.0, 0.0};
    }

    // Compute error metrics for hostC against hostGolden
    double maxRelativeErrorC = 0.0;
    double sumRelativeErrorC = 0.0;
    double sumSquaredErrorC = 0.0;

    // Compute error metrics for hostCpu against hostGolden
    double maxRelativeErrorCpu = 0.0;
    double sumRelativeErrorCpu = 0.0;
    double sumSquaredErrorCpu = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double cVal = static_cast<double>(hostC[i]);
        double cpuVal = static_cast<double>(hostCpu[i]);
        double goldenVal = static_cast<double>(hostGolden[i]);

        if (!std::isfinite(cVal) || !std::isfinite(cpuVal) || !std::isfinite(goldenVal)) {
            const double invalid = std::numeric_limits<double>::infinity();
            return {false, invalid, invalid, invalid};
        }

        // Error for hostC
        double diffC = std::fabs(cVal - goldenVal);
        double relativeErrorC = diffC / (std::fabs(goldenVal) + epsilon);
        maxRelativeErrorC = std::max(maxRelativeErrorC, relativeErrorC);
        sumRelativeErrorC += relativeErrorC;
        sumSquaredErrorC += diffC * diffC;

        // Error for hostCpu
        double diffCpu = std::fabs(cpuVal - goldenVal);
        double relativeErrorCpu = diffCpu / (std::fabs(goldenVal) + epsilon);
        maxRelativeErrorCpu = std::max(maxRelativeErrorCpu, relativeErrorCpu);
        sumRelativeErrorCpu += relativeErrorCpu;
        sumSquaredErrorCpu += diffCpu * diffCpu;
    }

    double mareC = maxRelativeErrorC;
    double mereC = sumRelativeErrorC / n;
    double rmseC = std::sqrt(sumSquaredErrorC / n);

    double mareCpu = maxRelativeErrorCpu;
    double mereCpu = sumRelativeErrorCpu / n;
    double rmseCpu = std::sqrt(sumSquaredErrorCpu / n);

    // Compute error ratios (hostC / hostCpu)
    // An exact CPU reference only passes when the device is also exact.
    auto errorRatio = [](double deviceError, double cpuError) {
        return cpuError > 0 ? deviceError / cpuError :
                              (deviceError == 0 ? 0.0 : std::numeric_limits<double>::infinity());
    };
    double mareRatio = errorRatio(mareC, mareCpu);
    double mereRatio = errorRatio(mereC, mereCpu);
    double rmseRatio = errorRatio(rmseC, rmseCpu);

    bool passed = (mareRatio <= mareThreshold && mereRatio <= mereThreshold && rmseRatio <= rmseThreshold);

    return {passed, mareRatio, mereRatio, rmseRatio};
}

// Explicit tolerances are useful when comparing directly against a high-precision
// reference rather than a same-precision CPU baseline.
struct ElementwiseTolerance {
    double absolute;
    double relative;
    bool preserveNonzero = false; // Require nonzero references to retain their sign and a nonzero result.
};

struct ElementwiseErrorMetrics {
    bool passed;
    uint64_t mismatchCount;
    double maxAbsoluteError;
    double rmse;
};

template <class ElementResult, class ElementGolden>
ElementwiseErrorMetrics CompareElementwise(
    const std::vector<ElementResult>& result, const std::vector<ElementGolden>& golden,
    const ElementwiseTolerance& tolerance)
{
    const double invalid = std::numeric_limits<double>::infinity();
    if (result.empty() || result.size() != golden.size() || !std::isfinite(tolerance.absolute) ||
        !std::isfinite(tolerance.relative) || tolerance.absolute < 0 || tolerance.relative < 0) {
        return {false, std::max(result.size(), golden.size()), invalid, invalid};
    }
    uint64_t mismatches = 0;
    double maxError = 0.0;
    double squaredError = 0.0;
    for (size_t i = 0; i < result.size(); ++i) {
        const double actual = static_cast<double>(result[i]);
        const double expected = static_cast<double>(golden[i]);
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++mismatches;
            maxError = squaredError = invalid;
            continue;
        }
        const double error = std::fabs(actual - expected);
        bool failed = error > tolerance.absolute + tolerance.relative * std::fabs(expected);
        if (tolerance.preserveNonzero && expected != 0.0) {
            failed |= actual == 0.0 || std::signbit(actual) != std::signbit(expected);
        }
        mismatches += failed;
        maxError = std::max(maxError, error);
        squaredError += error * error;
    }
    return {mismatches == 0, mismatches, maxError, std::sqrt(squaredError / result.size())};
}

template <class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(
    const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect, uint32_t computeNum)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

template <>
std::vector<uint64_t> CompareData(
    const std::vector<int32_t>& result, const std::vector<int32_t>& expect, uint32_t computeNum)
{
    using ElementCompare = int32_t;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        if (actualValue != expectValue) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

template <class ElementResult>
std::vector<uint64_t> CompareDataBfloat16(
    const std::vector<ElementResult>& result, const std::vector<float>& expect, uint32_t computeNum)
{
    /*
     * 高性能 浮点计算通过标准：
     * 计算次数<2048, errThres = 2^{-7}
     * 大于>2048, errThres = 2^{-6}
     *
     * 当$abs(golden)>=smallValThres$时，使用相对误差校验：
     * $$
     *  RE = \frac { abs(actual - golden)} { abs{golden} + 1e^{-7}} \le errThres
     * $$
     * 反之，采用绝对误差校验：
     * $$
     *  AE= abs(actual - golden) \le errThres
     * $$
     * 判断公式：
     * $$
     *  \left \| actual - expected \right\| \le
          errThres \times \max(smallValThres, abs(expected))
     * $$
    */
    using ElementCompare = float;
    const uint32_t computeNumThreshold = 2048;
    const float smallValThres = 1.0f / 256;
    const float rtolGeneral = 1.0f / 128;
    const float rtolOverThreshold = 1.0f / 64;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(smallValThres, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

// Compare for GroupedMatmul slicing M
template <class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(
    const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect, uint32_t computeNum,
    uint32_t validNum)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < validNum; ++i) {
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

// Compare for GroupedMatmul slicing K
template <class ElementResult, class ElementCompare, class T>
std::vector<uint64_t> CompareData(
    const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect, uint32_t computeNum,
    const std::vector<T>& groupList, uint32_t stride)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    T prevGroupValue = 0;
    uint64_t currentIndex = 0;
    for (const auto& groupValue : groupList) {
        if (groupValue == prevGroupValue) {
            currentIndex += stride;
            prevGroupValue = groupValue;
            continue;
        }
        for (uint64_t i = 0; i < stride; ++i) {
            if (currentIndex >= result.size())
                break;
            ElementCompare actualValue = static_cast<ElementCompare>(result[currentIndex]);
            ElementCompare expectValue = expect[currentIndex];
            ElementCompare diff = std::fabs(actualValue - expectValue);
            if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
                errorIndices.push_back(currentIndex);
            }
            currentIndex++;
        }
        prevGroupValue = groupValue;
    }
    return errorIndices;
}

} // namespace Catlass::golden

#endif // EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
