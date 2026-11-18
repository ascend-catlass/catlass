/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_H
#define ASCENDC_STUB_KERNEL_OPERATOR_H

#include <cstdint>

#define __aicore__
#define __gm__
#define __forceinline__
#define __ca__ // mx table
#define __cb__
#define __cbuf__
#define __ubuf__

struct half { uint16_t _; };
struct bfloat16_t { uint16_t _; };
struct float8_e8m0_t { uint8_t _; };
struct float4_e2m1x2_t { };
struct float4_e1m2x2_t { };
struct float8_e4m3_t { uint8_t _; };
struct float8_e5m2_t { uint8_t _; };

#include "kernel_tensor.h"
#include "kernel_struct_mm.h"
#include "kernel_struct_fixpipe.h"
#include "kernel_struct_unary.h"
#include "kernel_struct_transpose.h"
#include "kernel_operator_mm_intf.h"
#include "kernel_operator_data_copy.h"
#include "kernel_operator_data_copy_ext.h"
#include "kernel_operator_fixpipe_intf.h"
#include "kernel_operator_sync.h"
#include "kernel_operator_sys_var_intf.h"
#include "kernel_operator_vec_transpose_intf.h"
#include "kernel_operator_vec_unary_intf.h"
#include "kernel_operator_vec_binary_intf.h"
#include "kernel_operator_vec_vconv_intf.h"

#include "kernel_fp_types.h"

namespace AscendC {

enum class TPosition : int32_t
{
    GM,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    C2PIPE2GM,
    MAX,
};

struct int4b_t {};

template <typename T, typename U>
struct IsSameType {
    static constexpr bool value = std::is_same_v<T, U>;
};

} // namespace AscendC

#endif // ASCENDC_STUB_KERNEL_OPERATOR_H