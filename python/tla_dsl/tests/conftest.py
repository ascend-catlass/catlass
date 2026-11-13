from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Unit tests run off-device: keep torch from auto-importing torch_npu, which
# otherwise costs seconds in every process that imports torch.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


# The DSL examples live outside the package (examples/python/tla_dsl), so put
# their root on sys.path to keep ``from dsl_tutorials...`` / ``helpers`` imports
# working no matter which directory pytest is started from.
_EXAMPLES_ROOT = Path(__file__).resolve().parents[3] / "examples" / "python" / "tla_dsl"
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.append(str(_EXAMPLES_ROOT))

# On-device scripts; covered by dsl_battery, not unit pytest collection.
collect_ignore_glob = ["end_to_end/*"]

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--clean",
        action="store_true",
        default=False,
        help="Remove tests/.cache and rebuild the BC cache before running",
    )


def _clean_test_cache() -> None:
    """Drop tests/.cache under the lock BC builds into it use.

    ``catlass.bc_compile._lock_path()`` keeps the lock beside the guarded tree,
    so builds of ``tests/.cache/bc`` serialize on ``tests/.cache/bc.lock``.
    Taking that same inode is what makes ``--clean`` wait for a concurrent
    builder instead of deleting the tree under it. The lock file itself stays:
    unlinking it would let a later builder create a second lock inode and run
    alongside the current holder.
    """
    import fcntl
    import shutil

    root = _CACHE_ROOT.resolve()
    bc_root = root / "bc"
    lock_path = bc_root.parent / f"{bc_root.name}.lock"
    existed = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not existed:
            print(f"cache not found: {root}")
            return
        for entry in root.iterdir():
            if entry == lock_path:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        print(f"cleaned: {root}")
    finally:
        os.close(fd)


def pytest_configure(config: pytest.Config) -> None:
    # Workers start after this hook; only the controller may wipe the cache.
    if config.getoption("--clean") and not hasattr(config, "workerinput"):
        _clean_test_cache()


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
    os.environ["CATLASS_DSL_CACHE_DIR"] = str(_CACHE_ROOT)
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

    cache_root = _CACHE_ROOT
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
