# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Phase-1 compile-time policy descriptors (Act / Bias / SwiGLU).
from ffn.policies.act import (
    ACT_POLICIES,
    ActPolicy,
    get_act_policy,
    parse_activation,
)
from ffn.policies.bias import (
    BIAS_POLICIES,
    BiasPolicy,
    get_bias_policy,
    parse_bias,
)
from ffn.policies.swiglu import SWIGLU_DUAL, SWIGLU_SINGLE, SwigluPolicy

__all__ = [
    "ACT_POLICIES",
    "ActPolicy",
    "BIAS_POLICIES",
    "BiasPolicy",
    "SWIGLU_DUAL",
    "SWIGLU_SINGLE",
    "SwigluPolicy",
    "get_act_policy",
    "get_bias_policy",
    "parse_activation",
    "parse_bias",
]
