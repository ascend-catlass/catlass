"""The warp-shuffle device harness must not pass a non-finite output.

The device cases in ``tests/end_to_end/warp_shuffle_op.py`` are the only
lane-by-lane check the shuffle family has, so the predicate deciding whether a
lane is wrong is worth covering on the host: ``abs(got - want) > atol`` is false
for a NaN, which would let a kernel that writes nothing but NaNs report PASS.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def harness():
    source = (
        Path(__file__).resolve().parent / "end_to_end" / "warp_shuffle_op.py"
    )
    spec = importlib.util.spec_from_file_location("warp_shuffle_op", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_match_is_no_mismatch(harness):
    reference = [1.0, 2.0, 3.0, 4.0]
    assert harness.mismatches(list(reference), reference, 1e-5) == []


def test_wrong_lane_is_a_mismatch(harness):
    got = [1.0, 3.0, 3.0, 4.0]
    assert [i for i, _, _ in harness.mismatches(got, [1.0, 2.0, 3.0, 4.0], 1e-5)] == [1]


@pytest.mark.parametrize(
    "bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"]
)
def test_non_finite_output_is_a_mismatch(harness, bad):
    """Injecting one bad value must make the check fail, which is the whole point."""
    got = [1.0, bad, 3.0, 4.0]
    found = harness.mismatches(got, [1.0, 2.0, 3.0, 4.0], 1e-5)
    assert [i for i, _, _ in found] == [1]


def test_all_nan_output_fails_every_lane(harness):
    reference = [float(i + 1) for i in range(128)]
    got = [math.nan] * 128
    assert len(harness.mismatches(got, reference, 1e-5)) == 128
