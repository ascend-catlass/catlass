#!/usr/bin/env python3
"""Replay end-to-end examples in-process, for the validation battery.

The battery used to spawn one ``python examples/...`` per case. Each of those
processes pays the same fixed cost -- importing torch_npu and bringing the
device up -- which dwarfs the case itself: on an Ascend950PR box a basic_mmad
case costs ~23 s as its own process but ~1 s as an extra case in a process that
is already up.

Nothing here manages the runtime. The examples call torch.npu.set_device()
themselves and no longer initialize or finalize the ACL context, so cases simply
share whatever the first one brought up.

So the examples are not modified at all. Every one exposes ``main()`` parsing
``sys.argv``, so a case is just an argv: import the example module once, and call
its main() with the argv the shell used to pass on the command line.

See test_battery.py for the case list, and conftest.py for the session fixture.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

EXAMPLES_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "python" / "tla_dsl"
)
#: On-device scripts that are not published under examples/ (battery-only).
END_TO_END_DIR = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "tla_dsl"
    / "tests"
    / "end_to_end"
)
#: Path prefix for scripts under ``python/tla_dsl/tests/end_to_end/``.
END_TO_END_SCRIPT_PREFIX = "end_to_end/"

_MODULE_CACHE: dict[Path, Any] = {}


def resolve_script(rel: str) -> Path:
    """Map a battery argv path to an on-disk script.

    Paths starting with ``end_to_end/`` resolve under
    ``python/tla_dsl/tests/end_to_end/``; everything else stays under
    ``examples/python/tla_dsl/``.
    """

    if rel.startswith(END_TO_END_SCRIPT_PREFIX):
        return END_TO_END_DIR / rel[len(END_TO_END_SCRIPT_PREFIX) :]
    return EXAMPLES_DIR / rel


class _SiblingFinder(importlib.abc.MetaPathFinder):
    """Answer bare imports with this example directory's .py files.

    Replicates the standalone-run semantics where the script's own directory
    is sys.path[0], e.g. ``from debug_print import DTYPE_SPECS`` finding the
    sibling debug_print.py. We cannot add that directory to sys.path instead:
    the entry would outlive the case and every .py beside it would then shadow
    same-named modules for the rest of the session (a sibling called types.py
    would break every later case). The finder is installed at the head of
    sys.meta_path, not as a trailing fallback: a lingering sys.path entry for
    examples/python/tla_dsl makes PathFinder answer ``debug_print`` with the
    namespace package dsl_tutorials/debug_print/ -- a *successful* resolution a
    fallback never sees. The stdlib check below is what keeps a same-named
    sibling out of the standard library's way in this position.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def find_spec(
        self, name: str, path: Any = None, target: Any = None
    ) -> Any | None:
        if path is not None or name in sys.stdlib_module_names:
            return None
        candidate = self.directory / f"{name}.py"
        if not candidate.is_file():
            return None
        return importlib.util.spec_from_file_location(name, candidate)


@contextlib.contextmanager
def _sibling_imports(directory: Path) -> Any:
    """Make `directory`'s modules importable by bare name, for this block only."""

    finder = _SiblingFinder(directory)
    sys.meta_path.insert(0, finder)  # head position: see the class docstring
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def load_example(path: Path) -> Any:
    """Import an example module once and reuse it for every later case."""

    cached = _MODULE_CACHE.get(path)
    if cached is not None:
        return cached
    name = f"_tla_case_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"example_runner: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with _sibling_imports(path.parent):
        spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise SystemExit(f"example_runner: {path} has no main(); it cannot be batched")
    _MODULE_CACHE[path] = module
    return module



def run_case(argv: list[str]) -> int:
    """Run one case by replaying its argv through the example's own main()."""

    script = Path(argv[0])
    module = load_example(script)
    saved_argv = sys.argv
    # Restored for the same reason as sys.argv: every case runs in this one
    # process, so a variable an example exports would otherwise stay set for all
    # the cases after it. Not hypothetical -- an example whose --cache-dir has a
    # default exports CATLASS_DSL_CACHE_DIR unconditionally, pointing at its own
    # artifacts directory, and that redirects the next case's compile *and BC*
    # cache lookups to a directory nothing has populated.
    saved_environ = dict(os.environ)
    sys.argv = list(argv)
    try:
        # A cached module may still import a sibling lazily, inside run().
        with _sibling_imports(script.parent):
            rc = module.main()
    except SystemExit as exc:  # some examples exit rather than return
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.argv = saved_argv
        os.environ.clear()
        os.environ.update(saved_environ)
    return int(rc or 0)


