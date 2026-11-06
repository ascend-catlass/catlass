import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from catlass.base_dsl.runtime import ascend_debug_fifo as fifo


@pytest.fixture(scope="module")
def battery():
    source = Path(__file__).resolve().parents[3] / "tests/dsl_battery/test_battery.py"
    spec = importlib.util.spec_from_file_location("battery_print_policy", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner(monkeypatch, run):
    monkeypatch.setitem(
        sys.modules,
        "example_runner",
        SimpleNamespace(resolve_script=lambda name: Path(__file__), run_case=run),
    )


def test_local_print_cases_have_independent_storage(battery):
    cases = dict(battery.battery_cases())
    for case, storage in [
        (battery.L1_PRINT_CASE_ID, "l1"),
        (battery.L0C_PRINT_CASE_ID, "l0c"),
    ]:
        assert cases[case]
        assert all(argv[argv.index("--storage") + 1] == storage for argv in cases[case])


def test_missing_debug_bus_skips_l1_and_keeps_l0c(battery, monkeypatch):
    monkeypatch.setattr(fifo, "_debug_bus_support_error", lambda: "unsupported HAL")
    calls = []
    _runner(monkeypatch, lambda argv: calls.append(argv) or 0)
    with pytest.raises(pytest.skip.Exception, match="unsupported HAL"):
        battery.test_case(battery.L1_PRINT_CASE_ID, 0)
    assert calls == []
    battery.test_case(battery.L0C_PRINT_CASE_ID, 0)
    assert calls


@pytest.mark.parametrize("storage", ["l1", "l0c"])
@pytest.mark.parametrize(
    "error_type", [fifo.AscDebugUnsupportedError, fifo.AscDebugFifoError, RuntimeError]
)
def test_local_print_skips_only_l1_missing_interfaces(
    battery, monkeypatch, storage, error_type
):
    monkeypatch.setattr(fifo, "_debug_bus_support_error", lambda: None)

    def fail(argv):
        raise error_type("injected")

    _runner(monkeypatch, fail)
    case = battery.L1_PRINT_CASE_ID if storage == "l1" else battery.L0C_PRINT_CASE_ID
    expected = (
        pytest.skip.Exception
        if storage == "l1" and error_type is fifo.AscDebugUnsupportedError
        else error_type
    )
    with pytest.raises(expected, match="injected"):
        battery.test_case(case, 0)
