# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fixed shapes and selector values of Flash MLA decoding.

The kernel is tiled for one latent-attention head: the cache carries 576
features per token, of which the first 512 are the absorbed values. Nothing here
depends on the DSL, so the golden and the driver share it.
"""

PAGE_SIZE = 128  # tokens per KV page, fixed by the paged cache layout
HEAD_DIM_QK = 576  # latent query/key features
HEAD_DIM_V = 512  # absorbed value features (the leading 512 of the 576)
KV_HEAD_NUM = 1  # MLA: every query head reads the same latent KV head

MASK_NONE = 0
MASK_RIGHT_DOWN_CAUSAL = 3
MASK_MODES = (MASK_NONE, MASK_RIGHT_DOWN_CAUSAL)

CACHE_LAYOUT_BBND = "PA_BBND"  # [pages, 128, 1, 576]
CACHE_LAYOUT_NZ = "PA_NZ"  # [pages, 1, 36, 128, 16]
CACHE_LAYOUTS = (CACHE_LAYOUT_BBND, CACHE_LAYOUT_NZ)

HEAD_CHOICES = (64, 96)  # query heads a single (token, group) task can hold
Q_BLOCK = 64  # unpaired tile of query heads per task
KV_BLOCK = 128  # logical KV tokens per pipeline tile
QUERY_CHUNK = 32  # query rows per golden chunk, bounds the golden's P buffer

NPU_ARCH = "--npu-arch 3510"  # Ascend 950PR/DT

#: dtype name -> torch dtype, resolved lazily so importing this module needs no torch
DTYPE_NAMES = ("float16", "bfloat16")
