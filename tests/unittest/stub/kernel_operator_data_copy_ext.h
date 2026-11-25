/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_DATA_COPY_EXT_H
#define ASCENDC_STUB_KERNEL_OPERATOR_DATA_COPY_EXT_H

#include "kernel_operator.h"
#include "kernel_tensor.h"
#include "kernel_struct_mm.h"
#include "kernel_constants.h"
#include "ascendc_logger.h"
#include "arg.h"

namespace AscendC {

struct DataCopyExtParams {   
    DataCopyExtParams() {}

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
    DataCopyExtParams(const uint16_t count, const uint32_t len, const int64_t srcStrideIn,
        const int64_t dstStrideIn, const uint32_t rsvIn)
#else
    DataCopyExtParams(const uint16_t count, const uint32_t len, const uint32_t srcStrideIn,
        const uint32_t dstStrideIn, const uint32_t rsvIn)
#endif
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          rsv(rsvIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint32_t blockLen = 0;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
    int64_t srcStride = static_cast<int64_t>(DEFAULT_DATA_COPY_STRIDE);
    int64_t dstStride = static_cast<int64_t>(DEFAULT_DATA_COPY_STRIDE);
#else
    uint32_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint32_t dstStride = DEFAULT_DATA_COPY_STRIDE;
#endif
    uint32_t rsv = 0; // reserved

    std::string toString() const
    {
        char buffer[256];
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        snprintf(
            buffer, sizeof(buffer), "DataCopyExtParams(blockCount=%u, blockLen=%u, srcStride=%ld, dstStride=%ld, rsv=%u)",
            blockCount, blockLen, srcStride, dstStride, rsv);
#else
        snprintf(
            buffer, sizeof(buffer), "DataCopyExtParams(blockCount=%u, blockLen=%u, srcStride=%u, dstStride=%u, rsv=%u)",
            blockCount, blockLen, srcStride, dstStride, rsv);
#endif
        return std::string(buffer);
    }
};

// ============================================================================
// GetPadValueType (from kernel_utils_constants.h, helper for DataCopyPadExtParams::TYPE)
// ============================================================================

template <typename T> struct GetPadValueType {
    using Type = T;
};

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
// To support FP8 datacopypad, pad type needs transfer to b8
template <> struct GetPadValueType<fp8_e5m2_t> {
    using Type = uint8_t;
};

template <> struct GetPadValueType<fp8_e4m3fn_t> {
    using Type = uint8_t;
};

template <> struct GetPadValueType<fp8_e8m0_t> {
    using Type = uint8_t;
};

template <> struct GetPadValueType<hifloat8_t> {
    using Type = uint8_t;
};

// To support FP4 datacopypad, pad type needs transfer to b8
template <> struct GetPadValueType<fp4x2_e1m2_t> {
    using Type = uint8_t;
};

template <> struct GetPadValueType<fp4x2_e2m1_t> {
    using Type = uint8_t;
};
#endif

// ============================================================================
// DataCopyPadExtParams (from kernel_struct_data_copy.h)
// ============================================================================

template <typename T> struct DataCopyPadExtParams {
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
    using TYPE = typename GetPadValueType<T>::Type;
     DataCopyPadExtParams()
    {
        isPad = false;
        leftPadding = 0;
        rightPadding = 0;
        paddingValue = 0;
    }
    DataCopyPadExtParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue,
        T padValue)
    {
        isPad = isPadValue;
        leftPadding = leftPadValue;
        rightPadding = rightPadValue;
        paddingValue = *(reinterpret_cast<TYPE *>(&padValue));
    }
    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    TYPE paddingValue = 0;
#else
    DataCopyPadExtParams() {}

    DataCopyPadExtParams(const bool isPadValue, const uint8_t leftPadValue, const uint8_t rightPadValue,
        T padValue)
        : isPad(isPadValue),
          leftPadding(leftPadValue),
          rightPadding(rightPadValue),
          paddingValue(padValue)
    {}

    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    T paddingValue = 0;
#endif
};

enum class CopyMode
{
    COPY_NONE = 0,
    COPY_D2D = 1,
    COPY_D2H = 2,
    COPY_H2D = 3
};

enum class PadMode
{
    PAD_NONE = 0,
    PAD_NZ = 1,
    PAD_N1 = 2,
    PAD_NZ2 = 3
};

enum class PaddingMode : uint8_t {
    Normal = 0,
    Compact,
};

struct PadParams {
    uint32_t padValue = 0;
    PadMode padMode = PadMode::PAD_NONE;
    uint32_t padN = 0;
    uint32_t padC = 0;
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t srcN = 0;
    uint32_t srcC = 0;
    uint32_t srcH = 0;
    uint32_t srcW = 0;
    uint32_t dstN = 0;
    uint32_t dstC = 0;
    uint32_t dstH = 0;
    uint32_t dstW = 0;

    std::string toString() const
    {
        char buffer[256];
        snprintf(
            buffer, sizeof(buffer), "PadParams(padMode=%d, padValue=%u, srcN=%u, srcH=%u, srcW=%u)", (int)padMode,
            padValue, srcN, srcH, srcW);
        return std::string(buffer);
    }
};

template <typename T>
inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {{Arg::MakeArg<T>()}};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(const GlobalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params, const PadParams& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(
    const GlobalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params, const PadParams& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

#if (!defined(__NPU_ARCH__) || __NPU_ARCH__==2201)
template <typename T>
inline void DataCopyPad(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params, const PadParams& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<T>& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, dataCopyParams, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, dataCopyParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}
#elif (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
template <typename T, PaddingMode mode = PaddingMode::Normal>
inline void DataCopyPad(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params, const PadParams& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue<PaddingMode>(mode)};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T, PaddingMode mode = PaddingMode::Normal>
inline void DataCopyPad(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& params, const DataCopyPadExtParams<T>& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue<PaddingMode>(mode)};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T, PaddingMode mode = PaddingMode::Normal>
inline void DataCopyPad(
    const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& params, const DataCopyPadExtParams<T>& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue<PaddingMode>(mode)};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T, PaddingMode mode = PaddingMode::Normal>
inline void DataCopyPad(
    const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue<PaddingMode>(mode)};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}
#endif

template <typename T>
inline void DataCopyPad(
    const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params, const PadParams& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void Copy(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask, const uint8_t repeatTimes,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, mask, repeatTimes, repeatParams};
    ASCENDC_LOG_CALL_T(Copy, argsT, args);
}

template <typename T>
inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, count};
    ASCENDC_LOG_CALL_T(Copy, argsT, args);
}

inline void SetFmatrix(uint16_t l1H, uint16_t l1W, const uint8_t padList[PAD_SIZE], const FmatrixMode& fmatrixMode)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<FmatrixMode>()};
    const std::vector<Arg> args = {l1H, l1W, padList, fmatrixMode};
    ASCENDC_LOG_CALL_T(SetFmatrix, argsT, args);
}

enum class BrcbMode
{
    BRCB_NORMAL = 0,
    BRCB_NZ = 1
};

inline void Brcb(
    const LocalTensor<uint16_t>& dst, const LocalTensor<uint16_t>& src, const uint32_t& n, const uint32_t& c,
    const uint32_t& h, BrcbMode mode)
{
    const std::vector<Arg> args = {dst, src, n, c, h, mode};
    ASCENDC_LOG_CALL(Brcb, args);
}

} // namespace AscendC

// CCE intrinsic stub: GM→L1 pad fill SPR (PAD_VAL_OUTTOL1). Real device code
// calls this from Catlass::Gemm::Tile::ProgramGmToL1PadFillSpr.
inline void set_pad_val_outtol1(uint64_t config)
{
    const std::vector<Arg> args = {config};
    ASCENDC_LOG_CALL(set_pad_val_outtol1, args);
}

#endif