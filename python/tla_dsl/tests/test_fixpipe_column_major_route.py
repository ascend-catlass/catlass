# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""What the L0C -> GM fixpipe routes do and do not accept.

The ColumnMajor GM destination is the transposing (nz2dn) store. These pin its
parity with the RowMajor (nz2nd) store, and pin the one place the two C++
templates are NOT at parity -- quantization -- as currently unreachable for
BOTH, so that enabling it for one layout without the other is a test failure
rather than a surprise.
"""

from __future__ import annotations

import pytest

from catlass.core_api import _SUPPORTED_CUBE_FIXPIPE_ROUTES
from catlass.params import QuantMode

_GM_KEY_NO_RELU = ("gm", "{layout}", "-", "NO_QUANT", False)
_GM_KEY_RELU = ("gm", "{layout}", "-", "NO_QUANT", True)


def _gm_routes(layout: str):
    return {
        key: value
        for key, value in _SUPPORTED_CUBE_FIXPIPE_ROUTES.items()
        if key[0] == "gm" and key[1] == layout
    }


def test_column_major_gm_route_exists():
    """The transposing store is reachable at all."""
    assert _gm_routes("ColumnMajor"), "no L0C -> GM ColumnMajor fixpipe route"


@pytest.mark.parametrize("relu", [False, True])
def test_column_major_matches_row_major_dtypes(relu):
    """Same accumulator dtypes in, same destination dtypes out, relu included."""
    row = _SUPPORTED_CUBE_FIXPIPE_ROUTES[("gm", "RowMajor", "-", "NO_QUANT", relu)]
    col = _SUPPORTED_CUBE_FIXPIPE_ROUTES[("gm", "ColumnMajor", "-", "NO_QUANT", relu)]
    assert col == row, (
        "the ColumnMajor GM fixpipe route must accept exactly the dtypes the "
        f"RowMajor one does; row={row} col={col}"
    )


def test_fixpipe_quant_is_unreachable_for_every_layout():
    """Quant has C++ templates for RowMajor only -- keep it unreachable for both.

    copy_l0c_to_gm.hpp carries PER_TENSOR / PER_CHANNEL specializations for
    RowMajor with no ColumnMajor counterpart. No .bc symbol is registered for
    either, so today the asymmetry is invisible. If quant is ever opened up,
    this test fires: write the ColumnMajor specialization too, or the
    transposing store silently loses a feature the straight store has.
    """
    quantized = [
        key
        for key in _SUPPORTED_CUBE_FIXPIPE_ROUTES
        if key[3] != str(QuantMode.NO_QUANT)
    ]
    assert not quantized, (
        "a quantized fixpipe route was enabled; check that copy_l0c_to_gm.hpp "
        f"has a matching ColumnMajor specialization: {quantized}"
    )
