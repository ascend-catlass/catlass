# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import catlass.tla as tla


def _aic_wait_both_arrive(aiv_arrive, pipe) -> None:
    """Wait both AIV subs (mode=4). Matches CPP wait_intra ×2."""
    tla.cross_core_wait_flag(aiv_arrive, pipe, aiv_id=0)
    tla.cross_core_wait_flag(aiv_arrive, pipe, aiv_id=1)


def _aic_arrive_handshake(aiv_arrive, aic_ffts, aic_done) -> None:
    # CPP SyncAllImpl: PipeBarrier<ALL> on both sides, then AIC wait arrive →
    # FFTS → publish done. AIV sets arrive after its ALL.
    tla.pipe_barrier(tla.pipes.ALL)
    _aic_wait_both_arrive(aiv_arrive, tla.arch.SCALAR)
    tla.cross_core_set_flag(aic_ffts, tla.arch.FIX)
    tla.cross_core_wait_flag(aic_ffts, tla.arch.SCALAR)
    tla.cross_core_set_flag(aic_done, tla.arch.SCALAR)


@tla.jit
def stage_barrier_hard(
    aiv_arrive,
    aic_ffts,
    aic_done,
) -> None:
    """Compose hard SyncAll aligned with AscendC SyncAll<false>() on MIX 1:2.

    CPP ``SyncAllImpl``: ``PipeBarrier<ALL>`` then AIV ``set_intra(MTE3)``,
    AIC ``wait_intra(SCALAR)`` ×2 → FFTS (arrive set here after ALL).

    Flags (enclosing @tla.kernel):
      aiv_arrive = tla.cross_flag(..., mode=4)
      aic_ffts   = tla.cross_flag(..., mode=0)
      aic_done   = tla.cross_flag(..., mode=2)
    """
    with tla.cube():
        _aic_arrive_handshake(aiv_arrive, aic_ffts, aic_done)

    with tla.vector():
        # Inline arrive publish (must stay inside @tla.jit).
        tla.pipe_barrier(tla.pipes.ALL)
        aiv_sub = tla.arch.sub_block_idx()
        if aiv_sub == 0:
            tla.cross_core_set_flag(aiv_arrive, tla.arch.MTE3, aiv_id=0)
        else:
            tla.cross_core_set_flag(aiv_arrive, tla.arch.MTE3, aiv_id=1)
        tla.cross_core_wait_flag(aic_done, tla.arch.SCALAR)
