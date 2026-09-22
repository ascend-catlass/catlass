# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from .elem_types import (
    ELEM_CHOICES,
    ELEM_SPECS,
    ElemSpec,
    elem_spec,
    make_operand_tensors,
)
from .golden import compare, tolerance
from .params import TilingParams, SwizzleParams
from .utils import (
    UB_ALLOC_ALIGN_BYTES,
    compute_ub_slot_elems,
    create_tla_tensor,
    get_block_num,
    to_hf32,
)

__all__ = [
    # struct-like params
    "TilingParams",
    "SwizzleParams",
    # helper function
    "UB_ALLOC_ALIGN_BYTES",
    "compute_ub_slot_elems",
    "create_tla_tensor",
    "get_block_num",
    "to_hf32",
    # element-type table for the staging examples
    "ELEM_CHOICES",
    "ELEM_SPECS",
    "ElemSpec",
    "elem_spec",
    "make_operand_tensors",
    # golden compare
    "compare",
    "tolerance",
]
