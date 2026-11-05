/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_ASCEND950_SYRK_EXAMPLE_H
#define EXAMPLES_ASCEND950_SYRK_EXAMPLE_H

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

#include "helper.hpp"

namespace SyrkExample {

inline void CheckAcl(aclError status)
{
    if (status != ACL_ERROR_NONE) {
        throw std::runtime_error("ACL error: " + std::to_string(status));
    }
}

inline uint32_t ParseInteger(const char* text)
{
    size_t end = 0;
    auto value = std::stoll(text, &end);
    if (text[end] != '\0' || value < 0 || value > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("expected a non-negative 32-bit integer");
    }
    return static_cast<uint32_t>(value);
}

struct Options {
    uint32_t m;
    uint32_t k;
    uint32_t batch;
    int32_t device = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    float diagonal = 0.0f;

    void Validate() const
    {
        constexpr uint64_t MAX_ELEMENTS = 67108864;
        if (m < 32 || m > 8192 || k == 0 || k > 65536 || batch == 0 || uint64_t{batch} * m * k > MAX_ELEMENTS ||
            uint64_t{batch} * m * m > MAX_ELEMENTS) {
            throw std::invalid_argument(
                "require 32<=M<=8192, 1<=K<=65536, batch>=1 and each tensor <=67108864 elements");
        }
    }
};

class Runtime {
public:
    aclrtStream stream = nullptr;
    explicit Runtime(int32_t device) : device_(device)
    {
        CheckAcl(aclInit(nullptr));
        CheckAcl(aclrtSetDevice(device_));
        CheckAcl(aclrtCreateStream(&stream));
    }
    ~Runtime()
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_);
        aclFinalize();
    }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

private:
    int32_t device_;
};

class Buffer {
public:
    uint8_t* ptr = nullptr;
    explicit Buffer(size_t bytes, const void* host = nullptr)
    {
        if (bytes != 0) {
            CheckAcl(aclrtMalloc(reinterpret_cast<void**>(&ptr), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            if (host != nullptr) {
                CheckAcl(aclrtMemcpy(ptr, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            }
        }
    }
    ~Buffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

template <class Output, class Launch>
int Run(const Options& options, Launch launch)
{
    options.Validate();
    const uint32_t m = options.m, k = options.k;
    const size_t inputElements = size_t{options.batch} * m * k;
    const size_t outputElements = size_t{options.batch} * m * m;
    std::vector<bfloat16> x(inputElements), xt(inputElements);
    std::vector<Output> residual(options.beta != 0.0f ? outputElements : 0), output(outputElements);
    std::mt19937 generator(3097);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for (size_t i = 0; i < inputElements; ++i) {
        x[i] = bfloat16(distribution(generator));
        xt[i] = bfloat16(distribution(generator));
    }
    // Separate input buffers intentionally produce a non-symmetric product.
    // The kernel mirrors its lower triangle, including within diagonal tiles.
    if (!residual.empty()) {
        for (uint32_t b = 0; b < options.batch; ++b) {
            for (uint32_t row = 0; row < m; ++row) {
                for (uint32_t column = 0; column <= row; ++column) {
                    Output value = static_cast<Output>(distribution(generator));
                    residual[(size_t{b} * m + row) * m + column] = value;
                    residual[(size_t{b} * m + column) * m + row] = value;
                }
            }
        }
    }
    Runtime runtime(options.device);
    Buffer deviceX(inputElements * sizeof(bfloat16), x.data());
    Buffer deviceXt(inputElements * sizeof(bfloat16), xt.data());
    Buffer deviceResidual(residual.size() * sizeof(Output), residual.data());
    Buffer deviceOutput(outputElements * sizeof(Output));
    auto cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    launch(
        deviceX.ptr, deviceXt.ptr, deviceResidual.ptr != nullptr ? deviceResidual.ptr : deviceOutput.ptr,
        deviceOutput.ptr, runtime.stream, cores);
    CheckAcl(aclrtSynchronizeStream(runtime.stream));
    CheckAcl(aclrtMemcpy(
        output.data(), outputElements * sizeof(Output), deviceOutput.ptr, outputElements * sizeof(Output),
        ACL_MEMCPY_DEVICE_TO_HOST));

    size_t errors = 0;
    for (uint32_t b = 0; b < options.batch; ++b) {
        for (uint32_t row = 0; row < m; ++row) {
            for (uint32_t column = 0; column <= row; ++column) {
                double product = 0.0;
                for (uint32_t inner = 0; inner < k; ++inner) {
                    product += static_cast<double>(x[(size_t{b} * m + row) * k + inner]) *
                               static_cast<double>(xt[(size_t{b} * m + column) * k + inner]);
                }
                const size_t offset = (size_t{b} * m + row) * m + column;
                const size_t mirrored = (size_t{b} * m + column) * m + row;
                double expected = options.alpha * product + (row == column ? options.diagonal : 0.0f);
                if (!residual.empty()) {
                    expected += options.beta * static_cast<double>(residual[offset]);
                }
                double actual = static_cast<double>(output[offset]);
                double tolerance =
                    sizeof(Output) == 4 ? 2e-6 + 2e-5 * std::abs(expected) : 1e-2 * std::max(1.0, std::abs(expected));
                if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance ||
                    std::memcmp(&output[offset], &output[mirrored], sizeof(Output)) != 0) {
                    ++errors;
                }
            }
        }
    }
    if (errors != 0) {
        std::cerr << "Compare failed. Error count: " << errors << std::endl;
        return 1;
    }
    std::cout << "Compare success (batch=" << options.batch << ", distinct operands, exact symmetry)." << std::endl;
    return 0;
}

} // namespace SyrkExample
#endif
