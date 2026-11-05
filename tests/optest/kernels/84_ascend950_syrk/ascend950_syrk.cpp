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

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

/**
 * @brief Ascend950 SYRK with GM workspace AXPBY: D = alpha * X @ X^T + beta * Y.
 *
 * MIX kernel: AIC dual-writes float P/P^T to per-core ping-pong workspace;
 * AIV applies AXPBY and stores D. Runtime params: SyrkParams.
 */
extern "C" void Ascend950Syrk(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const SyrkParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_syrk", tParams);
    auto* entry = JitCompiler::instance().getKernel("ascend950_syrk_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
