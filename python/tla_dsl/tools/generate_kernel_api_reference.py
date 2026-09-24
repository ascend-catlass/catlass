#!/usr/bin/env python3
"""Generate TLA DSL Kernel API reference docs from ``core_api.py``.

Signatures, parameters and the `Directory:` path come from the source AST. The
docstrings come from the imported module instead, because an op can append to
its own docstring at import time (`mmad` / `mmad_mx` add the table of dtypes they
accept), which the source text does not show. Importing needs the MLIR python
bindings on the path.
Section order and blurbs live in ``DIRECTORY_SECTIONS`` below.
Writes English Markdown to ``docs/en/api/kernel_api_reference.md``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

from common import (
    APIEntry,
    PACKAGE_ROOT,
    const_str,
    directory_path,
    function_entry,
    has_dsl_user_op,
    parse_all_names,
    render_reference,
    run_cli,
)


CORE_API_PATH = PACKAGE_ROOT / "catlass" / "core_api.py"
CORE_API_QUALNAME_PREFIX = "catlass.core_api."
TENSOR_API_PATH = PACKAGE_ROOT / "catlass" / "tla" / "tensor.py"
OUTPUT_PATH = PACKAGE_ROOT / "docs" / "en" / "api" / "kernel_api_reference.md"
GENERATED_BY = "python/tla_dsl/tools/generate_kernel_api_reference.py"

DIRECTORY_SECTIONS: list[tuple[str, str]] = [
    (
        "Basic Data Types and Operations",
        "Construction and views for front-end structured values such as "
        "Shape / Coord / Stride / Layout / Tensor, plus pointer helpers.",
    ),
    (
        "Data Movement",
        "Tensor copies between on-chip and global memory, and UB register load/store.",
    ),
    ("Matrix Compute", "Cube-side matrix multiply-accumulate (`tla.mmad`)."),
    (
        "Vector Compute",
        "Compute and mask ops on the register-vector path; usually must be "
        "called inside `tla.vec.func()`.",
    ),
    ("Vector Compute / Mask Compute", "Mask creation and tail-mask updates."),
    (
        "Vector Compute / Basic Arithmetic",
        "Element-wise arithmetic and unary math ops. `VectorSSA` overloads "
        "`+` / `-` / `*` / `/` for `add` / `sub` / `mul` / `div` when no "
        "`mask=` is needed.",
    ),
    ("Vector Compute / Logical Compute", "Bitwise and logical ops on Mask / Vector."),
    (
        "Vector Compute / Compare and Select",
        "Vector compares that produce masks, and masked select.",
    ),
    (
        "Vector Compute / Data Fill",
        "Constant fill and lane-index sequence construction.",
    ),
    (
        "Vector Compute / Discrete and Aggregate",
        "Gather elements from a UB tensor by index, and masked reductions "
        "(`VectorSSA.reduce`).",
    ),
    (
        "Vector Compute / Data Rearrange",
        "Interleave / deinterleave and related lane reshuffles.",
    ),
    ("Vector Compute / Data Compress", "Compress valid lanes under a mask."),
    (
        "Vector Compute / Type Conversion",
        "Element-type conversion on the register-vector path.",
    ),
    (
        "Sync Control",
        "In-core / cross-core flags, pipe barriers, mutexes, and local-memory "
        "barriers.",
    ),
    (
        "System Variable Access",
        "Architecture attributes on `tla.arch` (layout tags, pipe identifiers, "
        "block helpers, etc.).",
    ),
    ("Resource Management", "On-chip scratch allocation via `allocate`."),
    (
        "Debug APIs",
        "In-kernel scalar / tensor debug printing.",
    ),
    (
        "Scopes and Control Flow",
        "Cube / Vector / `vec.func` regions and kernel-side loop ranges.",
    ),
]
DIRECTORY_ORDER = [path for path, _ in DIRECTORY_SECTIONS]
DIRECTORY_INTROS = dict(DIRECTORY_SECTIONS)


_MAKE_UNARY_HELPERS = frozenset({"_make_unary_op", "_make_scalar_aware_unary_op"})


def _unary_op_template(tree: ast.Module) -> APIEntry | None:
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "_make_unary_op":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.FunctionDef)
                and child.name == "_unary"
                and has_dsl_user_op(child.decorator_list)
            ):
                return function_entry("_unary", child)
    return None


def _doc_from_make_unary_call(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "doc":
            return const_str(kw.value) or ""
    if len(call.args) >= 2:
        return const_str(call.args[1]) or ""
    return ""


def _function_defs_by_name(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _collect_core_functions(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_vec_func" and has_dsl_user_op(node.decorator_list):
            entry = function_entry("vec.func", node)
            entry.qualified_name = "catlass.core_api._vec_func"
            entries["vec.func"] = entry
            continue
        if node.name not in exported:
            continue
        if node.name == "print" or has_dsl_user_op(node.decorator_list):
            entries[node.name] = function_entry(node.name, node)
    return entries


def _collect_unary_aliases(
    tree: ast.Module, exported: set[str], template: APIEntry | None
) -> dict[str, APIEntry]:
    if template is None:
        return {}
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or target.id not in exported:
            continue
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in _MAKE_UNARY_HELPERS
        ):
            continue
        name = target.id
        entries[name] = APIEntry(
            name=name,
            qualified_name=f"catlass.core_api.{name}",
            source_line=node.lineno,
            params=list(template.params),
            returns=template.returns,
            docstring=_doc_from_make_unary_call(value),
        )
    return entries


def _collect_scalar_aware_binary_aliases(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    """Public names built by ``_make_scalar_aware_binary_op`` (e.g. ``max`` / ``min``).

    Docs and signatures live on the vector-form helper (``_vector_max`` / ``_vector_min``);
    the Assign line is the public symbol users import.
    """
    defs = _function_defs_by_name(tree)
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or target.id not in exported:
            continue
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_make_scalar_aware_binary_op"
        ):
            continue
        if len(value.args) < 2 or not isinstance(value.args[1], ast.Name):
            continue
        vector_def = defs.get(value.args[1].id)
        if vector_def is None:
            continue
        name = target.id
        entry = function_entry(
            name,
            vector_def,
            qualified_name=f"catlass.core_api.{name}",
        )
        entry.source_line = node.lineno
        entries[name] = entry
    return entries


def _collect_arch_namespace(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    if "arch" not in exported:
        return {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "arch"
            and target.attr == "__doc__"
        ):
            return {
                "arch": APIEntry(
                    name="arch",
                    qualified_name="catlass.core_api.arch",
                    source_line=node.lineno,
                    docstring=const_str(value) or "",
                    is_namespace=True,
                )
            }
    return {
        "arch": APIEntry(
            name="arch",
            qualified_name="catlass.core_api.arch",
            source_line=None,
            docstring="",
            is_namespace=True,
        )
    }


def _collect_class_methods(
    path: Path, class_names: set[str], *, label: str, module: str
) -> dict[str, APIEntry]:
    """Document methods of ``class_names`` that are ``@dsl_user_op`` and have ``Directory:``."""
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in class_names
        ),
        None,
    )
    if cls is None:
        return {}

    entries: dict[str, APIEntry] = {}
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not has_dsl_user_op(node.decorator_list):
            continue
        doc = ast.get_docstring(node) or ""
        if directory_path(doc) is None:
            continue
        entries[f"{label}.{node.name}"] = function_entry(
            f"{label}.{node.name}",
            node,
            qualified_name=f"{module}.{cls.name}.{node.name}",
            source_path=path.resolve(),
            drop_self=True,
        )
    return entries


def _import_core_api() -> ModuleType:
    """Import the op module, which is where the ``__modify_doc__`` hooks live."""
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    try:
        import catlass.core_api as core_api
    except ImportError as exc:
        raise RuntimeError(
            "generating the reference needs `catlass.core_api` importable (the "
            f"MLIR python bindings on PYTHONPATH): {exc}"
        ) from exc
    return core_api


def _apply_doc_hooks(entries: dict[str, APIEntry]) -> None:
    """Let ops that declare one extend their own docstring for the reference.

    ``copy`` / ``mmad`` / ``mmad_mx`` carry a ``__modify_doc__`` hook that
    appends the routes or dtypes they accept. Importing does not run it -- the
    runtime docstring stays as written -- so the reference asks for it here. A
    hook edits ``fn.__doc__`` in place, so the original is put back afterwards:
    a second pass must not stack the same table twice.
    """
    module = _import_core_api()
    for entry in entries.values():
        if not entry.qualified_name.startswith(CORE_API_QUALNAME_PREFIX):
            continue
        attr = entry.qualified_name[len(CORE_API_QUALNAME_PREFIX) :]
        op = getattr(module, attr, None)
        hook = getattr(op, "__modify_doc__", None)
        if hook is None:
            continue
        original = op.__doc__
        hook(op)
        entry.docstring = op.__doc__ or ""
        op.__doc__ = original


def parse_core_api(path: Path) -> dict[str, APIEntry]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    exported = parse_all_names(tree)
    entries: dict[str, APIEntry] = {}
    entries.update(_collect_core_functions(tree, exported))
    entries.update(_collect_unary_aliases(tree, exported, _unary_op_template(tree)))
    entries.update(_collect_scalar_aware_binary_aliases(tree, exported))
    entries.update(_collect_arch_namespace(tree, exported))
    entries.update(
        _collect_class_methods(
            TENSOR_API_PATH,
            {"Tensor", "_Tensor"},
            label="Tensor",
            module="catlass.tla.tensor",
        )
    )
    entries.update(
        _collect_class_methods(
            path, {"VectorSSA"}, label="VectorSSA", module="catlass.core_api"
        )
    )
    _apply_doc_hooks(entries)
    return entries


def generate(*, docs_dir: Path | None = None) -> str:
    if not CORE_API_PATH.is_file():
        raise FileNotFoundError(f"core_api not found: {CORE_API_PATH}")

    docs_dir = docs_dir or OUTPUT_PATH.parent
    return render_reference(
        parse_core_api(CORE_API_PATH),
        docs_dir=docs_dir,
        title="TLA DSL Kernel API Reference",
        intro=[
            "This document describes the **TLA DSL kernel-side Core APIs** "
            "(typically imported as `import catlass.tla as tla`). "
            "It covers data structures, compute / sync helpers, on-chip resources, "
            "and debug printing. Host-side compile / launch / tensor binding are in "
            "`docs/en/api/host_api_reference.md`; environment variables are in "
            "`docs/zh/kernel_development/core_concepts/env_vars.md`.",
            "Interface descriptions and examples come from each op's source docstring "
            "(`Directory:` plus `Description:` / `Parameters:` / `Constraints:` / `Example:`).",
            "All APIs must be called inside a `@tla.kernel`-decorated kernel function body.",
        ],
        header_sources=(
            "Do not edit manually. Update docstrings/Examples in catlass/core_api.py",
            "(or the defining module for imported types) instead.",
        ),
        leftovers_title="Other Core APIs",
        leftovers_blurb=(
            "APIs still exported by the current source but not yet filed under "
            "the directory tree above."
        ),
        directory_order=DIRECTORY_ORDER,
        directory_intros=DIRECTORY_INTROS,
        default_source_path=CORE_API_PATH,
        generated_by=GENERATED_BY,
    )


def main() -> int:
    return run_cli(
        description=__doc__,
        default_output=OUTPUT_PATH,
        generate_fn=generate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
