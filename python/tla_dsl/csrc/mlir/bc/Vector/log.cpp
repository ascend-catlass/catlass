// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

namespace {

// tla.log mode=subnormal. The software log's FTZ-disabled branch scales
// subnormal inputs (<= the largest subnormal) by 2^10 (f16) / 2^23 (f32),
// passes them through vln, then corrects with -ln(2^10) / -ln(2^23). The AVE
// vln intrinsic used by the default path flushes those inputs instead.
template <typename T>
__simd_callee__ inline VectorReg<T> logSubnormal(VectorReg<T> src, ave_preg preg)
{
    AscendC::Reg::RegTensor<T> in, out;
    in.reg[0] = src;
    AscendC::Reg::MaskReg mask = convertAVEPregToVecBool(preg);
    static constexpr AscendC::Reg::LogSpecificMode logMode = {
        AscendC::Reg::MaskMergeMode::ZEROING, AscendC::LogAlgo::PRECISION_1ULP_FTZ_FALSE};
    AscendC::Reg::Log<T, &logMode>(out, in, mask);
    return out.reg[0];
}

} // namespace

extern "C" {
__aiv__ __attribute__((always_inline)) vector_f32 log_subnormal_f32(vector_f32 src, ave_preg preg)
{
    return logSubnormal<float>(src, preg);
}

__aiv__ __attribute__((always_inline)) vector_f16 log_subnormal_f16(vector_f16 src, ave_preg preg)
{
    return logSubnormal<half>(src, preg);
}
}

#endif
