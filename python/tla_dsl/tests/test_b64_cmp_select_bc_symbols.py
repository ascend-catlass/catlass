from __future__ import annotations

from pathlib import Path

import pytest

_MODES = ("lt", "le", "gt", "ge", "eq", "ne")
_TYPES = ("int64_t", "uint64_t")
# Names composed by TlaVectorRegionSimdPass::lowerCmp / lowerWhere and
# exported by csrc/mlir/bc/Vector/cmp_sel_reg.cpp.
_EXPECTED_SYMBOLS = tuple(
    [f"cmp_{mode}_{ty}" for mode in _MODES for ty in _TYPES]
    + [f"cmps_{mode}_{ty}" for mode in _MODES for ty in _TYPES]
    + [f"select_{ty}" for ty in _TYPES]
)


def _find_aiv_aggregate(cache_root: Path) -> Path:
    matches = list(cache_root.rglob("meta_op.aiv.c310.bc"))
    if not matches:
        pytest.skip("AIV aggregate BC was not built in this environment")
    return max(matches, key=lambda path: path.stat().st_mtime)


def test_b64_cmp_select_symbols_exist_in_bc(require_bc: Path) -> None:
    """The pass synthesizes its callee declarations, so IR checks cannot catch
    a missing or renamed 64-bit stub. Assert the compiled AIV aggregate still
    exports every name the pass can emit."""
    aggregate = _find_aiv_aggregate(require_bc)
    payload = aggregate.read_bytes()
    missing = [name for name in _EXPECTED_SYMBOLS if name.encode() not in payload]
    assert not missing, (
        f"{aggregate} is missing 64-bit compare/select symbols: {missing}; "
        "keep cmp_sel_reg.cpp and TlaVectorRegionSimdPass b64 callee names in sync"
    )
