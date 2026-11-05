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

#include <string>

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void Ascend950SyrkPolynomial(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const SyrkPolynomialParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_syrk_polynomial", tParams);
    macros["CATLASS_JIT_SYRK_TILE"] = std::to_string(params.tileMN);
    auto* entry = JitCompiler::instance().getKernel("ascend950_syrk_polynomial_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
}

} // namespace CatlassKernel
