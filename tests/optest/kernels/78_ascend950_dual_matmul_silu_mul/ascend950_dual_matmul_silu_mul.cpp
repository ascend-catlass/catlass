/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "catlass_kernel.h"
#include "catlass/detail/alignment.hpp"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace DualMatmulTiling {

struct Family {
    uint16_t mTile;
    uint16_t nTile;
    uint16_t l0NTile;
    uint16_t l1KTile;
    uint16_t l0KTile;
    uint8_t l1AStages;
    uint8_t l1BStages;
    uint8_t l0AStages;
    uint8_t l0BStages;
    uint16_t scheduler;
    bool fullLoadA;
};

constexpr uint32_t kFamilyCount = 4;
constexpr std::array<Family, kFamilyCount> kFamilies = {{
    Family{128, 128, 128, 256, 64, 2, 3, 2, 2, 100, false},
    Family{128, 64, 64, 128, 64, 2, 2, 2, 2, 100, false},
    Family{256, 64, 64, 128, 64, 2, 3, 2, 3, 100, false},
    Family{32, 32, 32, 512, 64, 2, 3, 2, 3, 100, false},
}};

} // namespace DualMatmulTiling

namespace {

constexpr uint32_t kWideNThreshold = 1024;
constexpr uint64_t kAscend950L1Size = 512ULL * 1024;
constexpr uint64_t kAscend950L0ASize = 64ULL * 1024;
constexpr uint64_t kAscend950L0BSize = 64ULL * 1024;
constexpr uint64_t kAscend950L0CSize = 256ULL * 1024;
constexpr uint64_t kAscend950UbSize = 248ULL * 1024;

// Path selects one of the small set of validated tiling families.
enum class DualMatmulPath : uint32_t
{
    General = 0,
    SkinnyN = 1,
    WideN = 2,
    KLargePingpong = 3,
    LargeM = 5,
    SmallShape = 6,
};

struct PathSpec {
    DualMatmulPath path;
    uint32_t mTile;
    uint32_t nTile;
    uint32_t l0NTile;
    uint32_t l1KTile;
    uint32_t l0KTile;
    uint32_t l1AStages;
    uint32_t l1BStages;
    uint32_t l0AStages;
    uint32_t l0BStages;
    uint32_t blockScheduler;
    bool fullLoadA;
};

template <aclDataType Dtype>
constexpr uint32_t DtypeBytes()
{
    static_assert(Dtype == ACL_FLOAT || Dtype == ACL_FLOAT16 || Dtype == ACL_BF16);
    return Dtype == ACL_FLOAT ? 4 : 2;
}

DualMatmulPath SelectPathKind(uint32_t m, uint32_t n, uint32_t k)
{
    if (m <= 256 && k <= 512 && n < kWideNThreshold) {
        return DualMatmulPath::SmallShape;
    }
    if (k == 1024 && n >= 8192) {
        return DualMatmulPath::WideN;
    }
    if (k >= 1024) {
        return DualMatmulPath::KLargePingpong;
    }
    if (m >= 1024) {
        return DualMatmulPath::LargeM;
    }
    if (n >= kWideNThreshold) {
        return DualMatmulPath::WideN;
    }
    if (n <= 256) {
        return DualMatmulPath::SkinnyN;
    }
    return DualMatmulPath::General;
}

bool CanUseDualMainloop(uint32_t k, const PathSpec& spec)
{
    // Both supported input/output formats are 16-bit; accumulation is always FP32.
    constexpr uint32_t inputBytes = DtypeBytes<ACL_FLOAT16>();
    constexpr uint32_t outputBytes = DtypeBytes<ACL_BF16>();
    constexpr uint32_t accumulatorBytes = DtypeBytes<ACL_FLOAT>();
    if (spec.fullLoadA && spec.l1AStages != 1) {
        return false;
    }
    if (spec.l1AStages + 2 * spec.l1BStages > 8 || spec.l0AStages + 2 * spec.l0BStages > 8) {
        return false;
    }
    if (spec.l0KTile > spec.l1KTile || spec.l0NTile > spec.nTile) {
        return false;
    }

    uint64_t l1ABytes = spec.fullLoadA ?
                            static_cast<uint64_t>(spec.mTile) * k * inputBytes :
                            static_cast<uint64_t>(spec.mTile) * spec.l1KTile * spec.l1AStages * inputBytes;
    uint64_t l1BBytes = static_cast<uint64_t>(spec.nTile) * spec.l1KTile * spec.l1BStages * 2 * inputBytes;
    uint64_t l0ABytes = static_cast<uint64_t>(spec.mTile) * spec.l0KTile * spec.l0AStages * inputBytes;
    uint64_t l0BBytes = static_cast<uint64_t>(spec.l0KTile) * spec.l0NTile * spec.l0BStages * 2 * inputBytes;
    uint64_t l0CBytes = static_cast<uint64_t>(spec.mTile) * spec.nTile * 4 * 2;

    uint32_t aivSplitM = CeilDiv(spec.mTile, 2U);
    uint32_t ubStrideN = CeilDiv(spec.nTile, 32U) * 32;
    uint64_t ubResultElems = static_cast<uint64_t>(aivSplitM) * ubStrideN;
    uint64_t ubBytes = 3ULL * ubResultElems * accumulatorBytes + ubResultElems * outputBytes;

    return l1ABytes + l1BBytes <= kAscend950L1Size && l0ABytes <= kAscend950L0ASize && l0BBytes <= kAscend950L0BSize &&
           l0CBytes <= kAscend950L0CSize && ubBytes <= kAscend950UbSize;
}

PathSpec FamilySpec(const DualMatmulTiling::Family& family, DualMatmulPath path)
{
    return PathSpec{
        path,
        family.mTile,
        family.nTile,
        family.l0NTile,
        family.l1KTile,
        family.l0KTile,
        family.l1AStages,
        family.l1BStages,
        family.l0AStages,
        family.l0BStages,
        family.scheduler,
        family.fullLoadA,
    };
}

PathSpec SelectPathSpec(
    uint32_t m, uint32_t n, uint32_t k)
{
    DualMatmulPath path = SelectPathKind(m, n, k);
    uint32_t familyIndex = 0;
    switch (path) {
        case DualMatmulPath::SkinnyN:
            familyIndex = 1;
            break;
        case DualMatmulPath::LargeM:
            familyIndex = 2;
            break;
        case DualMatmulPath::KLargePingpong:
            familyIndex = 3;
            break;
        case DualMatmulPath::WideN:
        case DualMatmulPath::SmallShape:
        case DualMatmulPath::General:
            familyIndex = 0;
            break;
    }
    PathSpec spec = FamilySpec(DualMatmulTiling::kFamilies[familyIndex], path);
    if (CanUseDualMainloop(k, spec)) {
        return spec;
    }
    for (uint32_t index = 0; index < DualMatmulTiling::kFamilyCount; ++index) {
        spec = FamilySpec(DualMatmulTiling::kFamilies[index], path);
        if (CanUseDualMainloop(k, spec)) {
            return spec;
        }
    }
    return {};
}

void ApplyKernelMacros(std::unordered_map<std::string, std::string>& macros, const PathSpec& spec)
{
    macros["CATLASS_JIT_DUAL_MATMUL_PATH_ID"] = std::to_string(static_cast<uint32_t>(spec.path));
    macros["CATLASS_JIT_DUAL_MATMUL_M_TILE"] = std::to_string(spec.mTile);
    macros["CATLASS_JIT_DUAL_MATMUL_N_TILE"] = std::to_string(spec.nTile);
    macros["CATLASS_JIT_DUAL_MATMUL_L0_N_TILE"] = std::to_string(spec.l0NTile);
    macros["CATLASS_JIT_DUAL_MATMUL_L1_K_TILE"] = std::to_string(spec.l1KTile);
    macros["CATLASS_JIT_DUAL_MATMUL_L0_K_TILE"] = std::to_string(spec.l0KTile);
    macros["CATLASS_JIT_DUAL_MATMUL_FULL_LOAD_A"] = spec.fullLoadA ? "1" : "0";
    macros["CATLASS_JIT_DUAL_MATMUL_L1A_STAGES"] = std::to_string(spec.l1AStages);
    macros["CATLASS_JIT_DUAL_MATMUL_L1B_STAGES"] = std::to_string(spec.l1BStages);
    macros["CATLASS_JIT_DUAL_MATMUL_L0A_STAGES"] = std::to_string(spec.l0AStages);
    macros["CATLASS_JIT_DUAL_MATMUL_L0B_STAGES"] = std::to_string(spec.l0BStages);
    macros["CATLASS_JIT_BLOCK_SCHEDULER"] = std::to_string(spec.blockScheduler);
}

} // namespace

namespace CatlassKernel {

extern "C" void Ascend950DualMatmulSiluMul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    const auto inputType = tParams.elem("A");
    const auto outputType = tParams.elem("D");
    if ((inputType != ACL_FLOAT16 && inputType != ACL_BF16) || tParams.elem("B") != inputType ||
        (outputType != ACL_FLOAT16 && outputType != ACL_BF16)) {
        throw std::invalid_argument("DualMatmul requires matching FP16/BF16 inputs and FP16/BF16 output");
    }
    PathSpec spec = SelectPathSpec(params.m, params.n, params.k);
    if (spec.mTile == 0) {
        throw std::runtime_error(
            "DualMatmul found no legal tiling configuration for shape (" + std::to_string(params.m) + ", " +
            std::to_string(params.n) + ", " + std::to_string(params.k) + ")");
    }
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_dual_matmul_silu_mul", tParams);
    ApplyKernelMacros(macros, spec);
    // The dual matmul always accumulates its fp16/bf16 inputs in fp32.
    macros["CATLASS_JIT_ELEMENT_C"] = "float";
    macros["CATLASS_JIT_KERNEL_NAME"] =
        JitMacroGenerator<TParams>::makeKernelName("ascend950_dual_matmul_silu_mul", macros);

    auto* entry =
        JitCompiler::instance().getKernel("ascend950_dual_matmul_silu_mul_impl.cpp", macros, JitKernelType::MIX);
    if (!entry) {
        throw std::runtime_error("DualMatmul JIT compilation did not produce a kernel entry point");
    }
    // The host wrapper supplies the device's AIC core count as blockNum.
    entry(std::max<uint32_t>(1, blockNum), stream, &params);
}

} // namespace CatlassKernel
