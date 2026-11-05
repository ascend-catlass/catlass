"""Ascend Catlass package root.

Preferred DSL import::

    import catlass.tla as tla
    from catlass.tla.runtime import from_dlpack

Importing the package also makes the BC cache ready on a cold machine (see
:func:`catlass.bc_compile.autobuild_on_import`), so running an operator does not
need a separate precompile step.  A multi-card run starts one process per card;
those imports are serialized by a cross-process lock, so exactly one of them
compiles and the rest reuse the result.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback for source tree
    __version__ = "0.0.0"

# Submodules (import paths), not a flat DSL namespace.
from . import tla
from . import types
from . import runtime
from . import dsl
from . import core_api as core
from . import params
from .address_space import AddressSpace

# Runs last: the build is fail-soft and must not mask an import error above, and
# waiting for it should not interleave with the submodule imports.
from .bc_compile import autobuild_on_import as _autobuild_on_import

_autobuild_on_import()
del _autobuild_on_import

__all__ = [
    "__version__",
    "tla",
    "types",
    "runtime",
    "dsl",
    "core",
    "params",
    "AddressSpace",
]
