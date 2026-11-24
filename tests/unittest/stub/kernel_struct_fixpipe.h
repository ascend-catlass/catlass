/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_STRUCT_FIXPIPE_H
#define ASCENDC_STUB_KERNEL_STRUCT_FIXPIPE_H

#include <cstdio>
#include <string>
#include <cstdint>

#include "kernel_constants.h"

// ===============================
// QuantMode_t 
// ===============================
enum QuantMode_t
{
    NoQuant,      // 不使能量化功能
    F322F16,      // float cast成half, cast mode为CAST_RINT模式
    F322BF16,     // float cast成bfloat16_t, cast mode为CAST_RINT模式
    DEQF16,       // int32_t量化成half, scalar量化
    VDEQF16,      // int32_t量化成half，tensor量化
    QF322B8_PRE,  // float量化成int8_t/uint8_t，scalar量化
    VQF322B8_PRE, // float量化成int8_t/uint8_t，tensor量化
    REQ8,             // int32_t量化成int8_t/uint8_t，scalar量化
    VREQ8,            // int32_t量化成int8_t/uint8_t，tensor量化
    QS322BF16_PRE,    // int32_t量化成bf16，scalar量化
    VQS322BF16_PRE,   // int32_t量化成bf16，tensor量化
    QF322F16_PRE,     // float量化成half，scalar量化
    VQF322F16_PRE,    // float量化成half，tensor量化
    QF322BF16_PRE,    // float量化成bf16，scalar量化
    VQF322BF16_PRE,   // float量化成bf16，tensor量化
    QF322F32_PRE,     // float量化成float，scalar量化
    VQF322F32_PRE,    // float量化成float，tensor量化
    QF322FP8_PRE,     // float量化成fp8_e4m3fn_t，scalar量化
    VQF322FP8_PRE,    // float量化成fp8_e4m3fn_t，tensor量化
    QF322HIF8_PRE,    // float量化成hifloat8_t，scalar量化（Half to Away）
    VQF322HIF8_PRE,   // float量化成hifloat8_t，tensor量化（Half to Away）
};

namespace AscendC {
// ===============================
// CO2Layout
// ===============================
enum class CO2Layout : uint8_t {
    NZ = 0,
    ROW_MAJOR, // ND Row
    COLUMN_MAJOR // ND Column
};

struct FixpipeConfig {
    CO2Layout format;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    bool isToUB;
#endif // __NPU_ARCH__ == 3510

    constexpr FixpipeConfig(CO2Layout format_) : format(format_)
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    , isToUB(false)
#endif
    {}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    constexpr FixpipeConfig(CO2Layout format_, bool isToUB_) : format(format_), isToUB(isToUB_) {}
#endif

    std::string toString() const
    {
        char buffer[128];
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        snprintf(buffer, sizeof(buffer), "FixpipeConfig(format=%d, isToUB=%d)", 
            static_cast<uint8_t>(format), isToUB);
#endif 
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        snprintf(buffer, sizeof(buffer), "FixpipeConfig(format=%d)", 
            static_cast<uint8_t>(format));
#endif
        return std::string(buffer);
    }
};

// Pre-defined `FixpipeConfig`
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ, false};
constexpr FixpipeConfig CFG_ROW_MAJOR = {CO2Layout::ROW_MAJOR, false};
constexpr FixpipeConfig CFG_COLUMN_MAJOR = {CO2Layout::COLUMN_MAJOR, false};
#else
constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ};
constexpr FixpipeConfig CFG_ROW_MAJOR = {CO2Layout::ROW_MAJOR};
constexpr FixpipeConfig CFG_COLUMN_MAJOR = {CO2Layout::COLUMN_MAJOR};
#endif // __NPU_ARCH__ == 3510

// ===============================
// ReluMode & ClipReluMode
// ===============================
enum class ClipReluMode{
    NOCLIP_RELU = 0,
    CLIP_RELU = 1
};

enum class ReluMode{
    NO_RELU = 0,
    NORMAL_RELU = 1,
    SCALAR_RELU = 2,
    VECTOR_RELU = 3
};

// ===============================
// FixpipeParamsV220
// ===============================
struct FixpipeParamsV220 {
    FixpipeParamsV220() = default;

    FixpipeParamsV220(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn, const bool reluEnIn, const QuantMode_t quantPreIn, const int64_t deqScalarIn,
        const uint16_t ndNumIn, const uint16_t srcNdStrideIn, const uint16_t dstNdStrideIn, const uint8_t unitFlagIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          reluEn(reluEnIn),
          quantPre(quantPreIn),
          deqScalar(deqScalarIn),
          ndNum(ndNumIn),
          srcNdStride(srcNdStrideIn),
          dstNdStride(dstNdStrideIn),
          unitFlag(unitFlagIn)
    {}

    uint16_t nSize = 0;
    uint16_t mSize = 0;  // M-DirectionSize
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    // Params: used for Quant
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint64_t deqScalar;
    // Params: used for nz2nd
    uint16_t ndNum = 1;
    uint16_t srcNdStride = 0;
    uint16_t dstNdStride = 0;
    bool reluEn = false;
    uint8_t unitFlag = 0;
    bool isChannelSplit = false;

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer),
            "nSize: %u, mSize: %u, dstStride: %u, quantPre: %d, deqScalar: %llu, ndNum: %u, srcNdStride: %u, dstNdStride: %u, reluEn: %d, unitFlag: %u, isChannelSplit: %d",
            nSize, mSize, dstStride, static_cast<int>(quantPre), static_cast<unsigned long long>(deqScalar),
            ndNum, srcNdStride, dstNdStride, reluEn, unitFlag, isChannelSplit);
        return std::string(buffer);
    }
};

// ===============================
// Nz2NdParams and 
// ===============================

struct Nz2NdParams {
    Nz2NdParams() = default;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    Nz2NdParams(const uint16_t ndNumIn, const uint16_t srcNdStrideIn, const uint32_t dstNdStrideIn)
    {
        ndNum = ndNumIn;
        srcNdStride = srcNdStrideIn;
        dstNdStride = dstNdStrideIn;
    }

    uint16_t ndNum = 1; // loop3Size
    uint16_t srcNdStride = 0; // loop3SrcStride
    uint32_t dstNdStride = 0; // loop3DstStride
#else
    Nz2NdParams(const bool nz2ndEnIn, const uint16_t ndNumIn, const uint16_t srcNdStrideIn,
        const uint16_t dstNdStrideIn, const uint16_t originalNSizeIn)
    {
        nz2ndEn = nz2ndEnIn;
        ndNum = ndNumIn;
        srcNdStride = srcNdStrideIn;
        dstNdStride = dstNdStrideIn;
        originalNSize = originalNSizeIn;
    }

    bool nz2ndEn = false;
    uint16_t ndNum = 1;
    uint16_t srcNdStride = 0;
    uint16_t dstNdStride = 0;
    uint16_t originalNSize = 0;
#endif
};

struct Nz2DnParams {
    Nz2DnParams() = default;

    Nz2DnParams(const uint16_t dnNumIn, const uint16_t srcNzMatrixStrideIn,
        const uint32_t dstDnMatrixStrideIn, const uint16_t srcNzC0StrideIn)
    {
        dnNum = dnNumIn;
        srcNzMatrixStride = srcNzMatrixStrideIn;
        dstDnMatrixStride = dstDnMatrixStrideIn;
        srcNzC0Stride = srcNzC0StrideIn;
    }

    uint16_t dnNum = 1; // loop3Size
    uint16_t srcNzMatrixStride = 0; // loop3SrcStride
    uint32_t dstDnMatrixStride = 0; // loop3DstStride
    uint16_t srcNzC0Stride = 0; // loop0SrcStride
};

// ===============================
// FixpipeParams3510(Arch)
// ===============================
#if defined(__NPU_ARCH__) && __NPU_ARCH__==3510
template <CO2Layout format>
struct TransformParams {};

template <>
struct TransformParams<CO2Layout::NZ> {
    inline TransformParams(){};
    using PARAMS = uint8_t;
};

template <>
struct TransformParams<CO2Layout::ROW_MAJOR> {
    inline TransformParams(){};
    using PARAMS = Nz2NdParams;
};

template <>
struct TransformParams<CO2Layout::COLUMN_MAJOR> {
    inline TransformParams(){};
    using PARAMS = Nz2DnParams;
};

template <CO2Layout format = CO2Layout::ROW_MAJOR>
struct FixpipeParamsArch3510 {
    FixpipeParamsArch3510() = default;

    FixpipeParamsArch3510(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn)
    {
        nSize = nSizeIn;
        mSize = mSizeIn;
        srcStride = srcStrideIn;
        dstStride = dstStrideIn;
    }
    
    ReluMode preReluMode = ReluMode::NO_RELU;
    ClipReluMode preClipReluMode = ClipReluMode::NOCLIP_RELU;
    uint64_t reluScalar;
    uint64_t vectorRelu;
    uint16_t nSize = 0;
    uint16_t mSize = 0;  // M-DirectionSize
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    // Params: used for Quant
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint64_t deqScalar;
    bool reluEn = false;
    uint8_t unitFlag = 0;
    // c310 extend param
    uint8_t dualDstCtl = 0;
    bool subBlockId = false;
    typename TransformParams<format>::PARAMS params;
    bool isChannelSplit = false;

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), 
            "preReluMode: {}, preClipReluMode: {}, reluScalar: {}, vectorRelu: {}, nSize: {}, mSize: {}, srcStride: {}, dstStride: {}, quantPre: {}, deqScalar: {}, reluEn: {}, unitFlag: {}, dualDstCtl: {}, subBlockId: {}, params: {}, isChannelSplit: {}",
            preReluMode, preClipReluMode, reluScalar, vectorRelu, nSize, mSize, srcStride, dstStride, quantPre, deqScalar, reluEn, unitFlag, dualDstCtl, subBlockId, params.toString(), isChannelSplit);
        return std::string(buffer);
    }
};

template <CO2Layout format = CO2Layout::ROW_MAJOR>
struct FixpipeParamsC310 : FixpipeParamsArch3510<format> {
    FixpipeParamsC310() = default;

    FixpipeParamsC310(const uint16_t nSizeIn, const uint16_t mSizeIn, 
                                const uint16_t srcStrideIn, const uint32_t dstStrideIn)
    : FixpipeParamsArch3510<format>(nSizeIn, mSizeIn, srcStrideIn, dstStrideIn) {}
};
#endif // __NPU_ARCH__==3510
}

#endif // ASCENDC_STUB_KERNEL_STRUCT_FIXPIPE_H