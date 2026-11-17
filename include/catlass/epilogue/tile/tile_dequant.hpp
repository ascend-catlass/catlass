/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_DEQUANT_HPP
#define CATLASS_EPILOGUE_TILE_TILE_DEQUANT_HPP

#include <type_traits>

#include "catlass/catlass.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Epilogue::Tile {

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using namespace AscendC::Reg;
#endif

template <
    class ElementOut_, class ElementIn_, class ElementBias_, class ElementX1Scale_, class ElementX2Scale_,
    QuantMode X1_QUANT_MODE_, QuantMode X2_QUANT_MODE_>
class TileDequant {
public:
    using ElementOut = ElementOut_;
    using ElementIn = ElementIn_;
    using ElementBias = ElementBias_;
    using ElementX1Scale = ElementX1Scale_;
    using ElementX2Scale = ElementX2Scale_;
    using ElementBiasStorage = std::conditional_t<std::is_void_v<ElementBias>, float, ElementBias>;

    static constexpr QuantMode x1QuantMode = X1_QUANT_MODE_;
    static constexpr QuantMode x2QuantMode = X2_QUANT_MODE_;
    static constexpr bool hasBias = !std::is_void_v<ElementBias>;
    static constexpr uint32_t UB_ALIGN_SIZE = 32;

    static_assert(
        x1QuantMode == QuantMode::NONE || x1QuantMode == QuantMode::PERTENSOR_MODE ||
            x1QuantMode == QuantMode::PERTOKEN_MODE,
        "Unsupported x1 quantization mode for TileDequant");
    static_assert(
        x2QuantMode == QuantMode::PERTENSOR_MODE || x2QuantMode == QuantMode::PERCHANNEL_MODE,
        "Unsupported x2 quantization mode for TileDequant");
    static_assert(
        (x1QuantMode == QuantMode::NONE &&
         (x2QuantMode == QuantMode::PERTENSOR_MODE || x2QuantMode == QuantMode::PERCHANNEL_MODE)) ||
            (x1QuantMode == QuantMode::PERTOKEN_MODE &&
             (x2QuantMode == QuantMode::PERTENSOR_MODE || x2QuantMode == QuantMode::PERCHANNEL_MODE)) ||
            (x1QuantMode == QuantMode::PERTENSOR_MODE && x2QuantMode == QuantMode::PERCHANNEL_MODE),
        "Unsupported quantization mode combination for TileDequant");

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    CATLASS_DEVICE_SIMD_VF void DoDequant(
        __ubuf__ ElementOut* dst, __ubuf__ ElementIn* l0cOut, __ubuf__ ElementX2Scale* scale2,
        __ubuf__ ElementX1Scale* x1Scale, __ubuf__ ElementBiasStorage* bias, float x1ScaleScalar, float x2ScaleScalar,
        uint16_t mSize, uint16_t nSize)
    {
        constexpr uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(ElementIn);
        uint32_t nSrcUbAligned = RoundUp(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(ElementIn)));
        uint32_t nDstUbAligned = RoundUp(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(ElementOut)));
        uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;

        constexpr static CastTrait ctInt322Fp32 = {
            RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait ctFp322Half = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait ctHalf2Fp32Zero = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait ctHalf2Fp32One = {
            RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

        MaskReg maskN4B16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                RegTensor<ElementIn> l0cOutReg;
                RegTensor<ElementX2Scale> scaleReg;
                RegTensor<ElementX1Scale> perTokenScaleReg;
                RegTensor<ElementBiasStorage> biasReg;
                RegTensor<float> castSrcOutReg, castScaleReg, castScaleOneReg, mulScaleOutReg, mulPtScaleOutReg,
                    castBiasReg, castBiasOneReg, addBiasOutReg;
                RegTensor<ElementOut> castResultOutReg;
                MaskReg maskN = UpdateMask<ElementIn>(elementNum);
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                LoadAlign(l0cOutReg, l0cOut + l0cOutOffset);
                // cast l0cOut from int32 to float
                if constexpr (AscendC::IsSameType<ElementIn, int32_t>::value) {
                    Cast<float, ElementIn, ctInt322Fp32>(castSrcOutReg, l0cOutReg, maskN);
                } else {
                    castSrcOutReg = l0cOutReg;
                }
                // l0c_out * scale2
                if constexpr (x2QuantMode == QuantMode::PERTENSOR_MODE) {
                    Muls(mulScaleOutReg, castSrcOutReg, x2ScaleScalar, maskN);
                } else {
                    LoadAlign(scaleReg, scale2 + vfBlockIdx * eleNumPerVf);
                    if constexpr (!AscendC::IsSameType<ElementX2Scale, float>::value) {
                        Cast<float, ElementX2Scale, ctHalf2Fp32Zero>(castScaleReg, scaleReg, maskN);
                        Cast<float, ElementX2Scale, ctHalf2Fp32One>(castScaleOneReg, scaleReg, maskN4B16);
                        Interleave(castScaleReg, castScaleOneReg, castScaleReg, castScaleOneReg);
                    } else {
                        castScaleReg = scaleReg;
                    }
                    Mul(mulScaleOutReg, castSrcOutReg, castScaleReg, maskN);
                }
                // out * x1Scale
                if constexpr (x1QuantMode == QuantMode::PERTENSOR_MODE) {
                    Muls(mulPtScaleOutReg, mulScaleOutReg, x1ScaleScalar, maskN);
                } else if constexpr (x1QuantMode == QuantMode::PERTOKEN_MODE) {
                    LoadAlign<ElementX1Scale, LoadDist::DIST_BRC_B32>(perTokenScaleReg, x1Scale + mIdx);
                    Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, maskN);
                } else {
                    mulPtScaleOutReg = mulScaleOutReg;
                }
                // out + bias
                if constexpr (hasBias) {
                    LoadAlign(biasReg, bias + vfBlockIdx * eleNumPerVf);
                    // cast bias from bf16/fp16 to float
                    if constexpr (
                        AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                        AscendC::IsSameType<ElementBias, half>::value) {
                        Cast<float, ElementBias, ctHalf2Fp32Zero>(castBiasReg, biasReg, maskN);
                        Cast<float, ElementBias, ctHalf2Fp32One>(castBiasOneReg, biasReg, maskN4B16);
                        Interleave(castBiasReg, castBiasOneReg, castBiasReg, castBiasOneReg);
                    } else {
                        castBiasReg = biasReg;
                    }
                    Add(addBiasOutReg, mulPtScaleOutReg, castBiasReg, maskN);
                } else {
                    addBiasOutReg = mulPtScaleOutReg;
                }
                // cast dequant result from float to fp16/bf16
                if constexpr (!AscendC::IsSameType<ElementOut, float>::value) {
                    Cast<ElementOut, float, ctFp322Half>(castResultOutReg, addBiasOutReg, maskN);
                } else {
                    castResultOutReg = addBiasOutReg;
                }
                // copy out from register to ub
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                if constexpr (AscendC::IsSameType<ElementOut, float>::value) {
                    StoreAlign<ElementOut, StoreDist::DIST_NORM_B32>(dst + dstUbOffset, castResultOutReg, maskN);
                } else {
                    StoreAlign<ElementOut, StoreDist::DIST_PACK_B32>(dst + dstUbOffset, castResultOutReg, maskN);
                }
            }
        }
    }
#endif
};

} // namespace Catlass::Epilogue::Tile

#endif // CATLASS_EPILOGUE_TILE_TILE_DEQUANT_HPP
