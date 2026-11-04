/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdexcept>

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {
extern "C" void Ascend950BmmBroadcast(
    uint32_t blockNum, aclrtStream stream, const TParams& tParams, const BmmBroadcastParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_bmm_broadcast", tParams);
    macros["CATLASS_BMM_HAS_BIAS"] = "0";
    if (params.biasMode != 0) {
        macros["CATLASS_BMM_HAS_BIAS"] = "1";
    }
    macros["CATLASS_BMM_HF32"] = "0";
    if (tParams.flagOn("hf32")) {
        macros["CATLASS_BMM_HF32"] = "1";
    }
    macros["CATLASS_JIT_KERNEL_NAME"] +=
        "_bias" + macros["CATLASS_BMM_HAS_BIAS"] + "_hf32" + macros["CATLASS_BMM_HF32"];
    auto* entry = JitCompiler::instance().getKernel("ascend950_bmm_broadcast_impl.cpp", macros, JitKernelType::AIC);
    if (!entry) {
        throw std::runtime_error("Failed to compile ascend950_bmm_broadcast");
    }
    entry(blockNum, stream, &params);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("BMM broadcast execution failed");
    }
}
} // namespace CatlassKernel
