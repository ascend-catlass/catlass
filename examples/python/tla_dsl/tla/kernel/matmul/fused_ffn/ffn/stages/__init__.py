# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# FFN DSL stages (Up/Down + Phase-1 epi helpers).

from ffn.stages.down_basic import stage_down_basic
from ffn.stages.down_full_a import stage_down_full_a
from ffn.stages.down_full_b import stage_down_full_b
from ffn.stages.down_streamk import stage_down_streamk
from ffn.stages.epi_silu import silu_apply_vl_1d, silu_apply_vl_2d
from ffn.stages.epi_swiglu_single import swiglu_apply_vl_1d
from ffn.stages.up_l0c2ub_silu import (
    C_UB_ELEMS,
    stage_up_l0c2ub_silu,
    swiglu_dual_ub_bytes,
    up_fixpipe_acc_elems,
)

__all__ = [
    "C_UB_ELEMS",
    "silu_apply_vl_1d",
    "silu_apply_vl_2d",
    "stage_down_basic",
    "stage_down_full_a",
    "stage_down_full_b",
    "stage_down_streamk",
    "stage_up_l0c2ub_silu",
    "swiglu_apply_vl_1d",
    "swiglu_dual_ub_bytes",
    "up_fixpipe_acc_elems",
]
