from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest


# The DSL examples live outside the package (examples/python/tla_dsl), so put
# their root on sys.path to keep ``from dsl_tutorials...`` / ``helpers`` imports
# working no matter which directory pytest is started from.
_EXAMPLES_ROOT = Path(__file__).resolve().parents[3] / "examples" / "python" / "tla_dsl"
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.append(str(_EXAMPLES_ROOT))

# On-device scripts; covered by dsl_battery, not unit pytest collection.
collect_ignore_glob = ["end_to_end/*"]


def _load_pretest_bootstrap(repo_root: Path):  # type: ignore[no-untyped-def]
    module_path = repo_root / "tests" / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("catlass_test_bootstrap", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load pretest bootstrap module from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_sessionstart(session: pytest.Session) -> None:
    del session
    repo_root = Path(__file__).resolve().parents[1]
    # Point the BC cache at the repo-local directory before collection imports
    # `catlass`: that import builds a missing cache, and without this it would
    # build into the user-wide default instead of the isolated test cache.
    os.environ["CATLASS_DSL_CACHE_DIR"] = str(
        Path(__file__).resolve().parent / ".cache"
    )
    bootstrap = _load_pretest_bootstrap(repo_root)
    try:
        bootstrap.ensure_pretest_mlir_build(repo_root)
    except bootstrap.PretestBuildError as exc:
        # Fail fast instead of letting tests run against missing MLIR
        # binaries (which would fail later with confusing import errors).
        pytest.exit(
            f"Pre-test MLIR build check failed: {exc}\nRun `./build.sh` first.",
            returncode=1,
        )


def _compiler_tlair(kernel: Any, *, type_args: tuple[Any, ...] | None = None) -> str:
    """Return the generic Tla IR text consumed by the runtime compiler path."""
    from catlass.base_dsl import BaseDSL

    lowered = BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=type_args,
        location=kernel.decorator_location,
    )
    return lowered.asm(generic=True)


@pytest.fixture(scope="session")
def require_bc() -> Path:
    """Auto-compile BC into tests/.cache and return the cache root.

    BC compilation is an install-step responsibility, so the test session
    never touches the shared install/CLI cache. Artifacts live in the
    repo-local ``tests/.cache`` (gitignored), which also works on machines
    without a usable ``/tmp``. The cache is reused across sessions, so the
    compile happens only once; drop ``tests/.cache`` to force a rebuild.
    """
    from catlass.bc_compile import init

    cache_root = Path(__file__).resolve().parent / ".cache"
    os.environ["CATLASS_DSL_CACHE_DIR"] = str(cache_root)
    init()
    return cache_root


@pytest.fixture
def isolated_compile_cache(require_bc: Path, tmp_path: Path, monkeypatch) -> Path:
    """Isolate kernel artifacts while reusing the session's compiled BC."""
    (tmp_path / "bc").symlink_to(require_bc / "bc", target_is_directory=True)
    monkeypatch.setenv("CATLASS_DSL_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def compiler_tlair() -> Any:
    return _compiler_tlair
