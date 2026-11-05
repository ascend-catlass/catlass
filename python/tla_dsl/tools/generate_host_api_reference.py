#!/usr/bin/env python3
"""Generate CATLASS Python Host API reference docs from Host sources (AST only).

API docstrings carry ``Directory:`` plus Description / Parameters / Constraints /
Example. Environment variables are documented separately in
``docs/zh/kernel_development/core_concepts/env_vars.md`` (not scanned here).
Writes split English Markdown under ``docs/en/api/runtime/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from common import (
    APIEntry,
    PACKAGE_ROOT,
    ParamInfo,
    directory_path,
    function_entry,
    published_wording,
    render_reference,
    run_cli_pages,
    split_reference_document,
)


OUTPUT_PATH = PACKAGE_ROOT / "docs" / "en" / "api" / "runtime" / "index.md"
GENERATED_BY = "python/tla_dsl/tools/generate_host_api_reference.py"

HOST_PAGE_FILES: dict[str, str] = {
    "1": "runtime/kernel_entry.md",
    "2": "runtime/compile_launch.md",
    "3": "runtime/host_tensor.md",
}

HOST_PAGE_TITLES: dict[str, str] = {
    "1": "Kernel Entry and Language Boundary",
    "2": "Compile and Launch",
    "3": "Host Tensor Binding",
}
DEFAULT_SOURCE_PATH = PACKAGE_ROOT / "catlass" / "dsl.py"

# Host decorator docs cover ``@tla.kernel``, ``@tla.jit``, and ``@tla.extern``.
HOST_DIRECTORY_SECTIONS: list[tuple[str, str]] = [
    (
        "Decorators",
        "Host-side `@tla.kernel` entry, `@tla.jit` device helpers, "
        "`@tla.extern` declarations, and Host `@dataclass` packing. Kernel and "
        "extern declaration bodies are not executed as ordinary Host Python. "
        "A `@tla.jit` helper is inlined during kernel lowering; a top-level Host "
        "call runs the helper as ordinary Python.",
    ),
    (
        "Compile and Launch",
        "Compile a decorated kernel and launch it on the NPU. Use "
        "`tla.compile` to obtain a callable `JitCompiledFunction`; call it "
        "directly to lazily create and then reuse its executor. Cache / arch / IR-dump "
        "knobs that are not function arguments are in "
        "[Environment Variables](../kernel_development/core_concepts/env_vars.md).",
    ),
    (
        "Compile and Launch / Compile",
        "Build a device binary. Primary entry: `tla.compile`. "
        "`TlaJitFunction.compile` is a lower-level helper on the decorated "
        "function.\n\n"
        "`options` is one string, split on whitespace. A key option takes a "
        "value; a switch takes none and is on once written.\n\n"
        "| Option | Form | Meaning |\n"
        "| --- | --- | --- |\n"
        "| `--npu-arch <chip>` | key | Target chip, e.g. `3510`. |\n"
        "| `--cce-disable-asc-reserved-ubuf` | switch | Release the 2 KB of "
        "Unified Buffer the compiler holds back for Ascend C. |\n"
        "| `--cce-disable-vf-stack-reserved-ubuf` | switch | Release the 6 KB "
        "of Unified Buffer the compiler holds back for the VF stack. |\n\n"
        "The Unified Buffer is 256 KB, of which the compiler reserves 8 KB "
        "(2 KB Ascend C, 6 KB VF stack), leaving a kernel 248 KB. The two "
        "switches release those reserves, and both are spelled as bisheng "
        "spells them.\n\n"
        "`tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)` reports the "
        "whole 256 KB, not what is left after the reserve. A kernel that "
        "divides that figure into buffers without also passing the two "
        "switches asks for more than it is given, and is refused at launch.\n\n"
        "**`--cce-disable-vf-stack-reserved-ubuf` carries a risk.** The VF "
        "stack is where the compiler spills vector registers. Once it is "
        "released, a kernel that provokes a spill leaves the compiler nowhere "
        "to write, and nothing checks -- the result is a silent write over "
        "whatever sits next to it in the Unified Buffer. Use it only for a "
        "kernel you know does not spill.",
    ),
    (
        "Compile and Launch / Launch",
        "Run a compiled kernel on the NPU by calling the `JitCompiledFunction` "
        "returned by `tla.compile`.",
    ),
    (
        "Compile and Launch / Inspect",
        "Dump frontend TLA IR without building a device binary or launching. "
        "See `TlaJitFunction.dump_mlir`.",
    ),
    (
        "Host Tensor",
        "Build Host `tla.Tensor` objects and mark layout extents dynamic so one "
        "artifact can run at different shapes. See also "
        "[Static and Dynamic Layout](../kernel_development/core_concepts/layout.md).",
    ),
    (
        "Host Tensor / Binding",
        "Bind a real NPU buffer with `from_dlpack` or Host dual-mode "
        "`tla.make_ptr` / `tla.make_layout` / `tla.make_tensor`, or build a "
        "metadata-only sample with `make_fake_tensor`.",
    ),
    (
        "Host Tensor / Dynamic Layout",
        "Mark static layout extents dynamic. See also "
        "[Static and Dynamic Layout](../kernel_development/core_concepts/layout.md).",
    ),
]
DIRECTORY_ORDER = [path for path, _ in HOST_DIRECTORY_SECTIONS]
DIRECTORY_INTROS = dict(HOST_DIRECTORY_SECTIONS)

HOST_SOURCE_PATHS = (
    PACKAGE_ROOT / "catlass" / "dsl.py",
    PACKAGE_ROOT / "catlass" / "catlass_dsl" / "catlass.py",
    PACKAGE_ROOT / "catlass" / "base_dsl" / "compiler.py",
    PACKAGE_ROOT / "catlass" / "base_dsl" / "jit_executor.py",
    PACKAGE_ROOT / "catlass" / "execution_lowering.py",
    PACKAGE_ROOT / "catlass" / "base_dsl" / "runtime" / "jit_arg_adapters.py",
    PACKAGE_ROOT / "catlass" / "tla" / "ffi.py",
    PACKAGE_ROOT / "catlass" / "tla" / "runtime.py",
)

# Display names in the generated reference (source qualified names stay unchanged).
HOST_DISPLAY_NAMES = {
    "CompileCallable.__call__": "tla.compile",
    "JitCompiledFunction.__call__": "tla.JitCompiledFunction.__call__",
    "_Tensor.mark_layout_dynamic": "tla.Tensor.mark_layout_dynamic",
    "_Tensor.mark_compact_shape_dynamic": "tla.Tensor.mark_compact_shape_dynamic",
    # Stdlib ``@dataclass`` packing rules live on the struct-arg validator.
    "_validate_dataclass_kernel_arg": "dataclasses.dataclass",
    # Public ``tla.kernel`` / ``tla.jit`` are aliases of these classmethods.
    "CatlassBaseDSL.kernel": "tla.kernel",
    "CatlassBaseDSL.jit": "tla.jit",
}

# Signature / qualified-name overrides when the documented public face differs
# from the source helper (e.g. stdlib ``@dataclass``).
HOST_ENTRY_OVERRIDES: dict[str, dict[str, object]] = {
    "extern": {
        "params": [
            ParamInfo("source", "str", "keyword_only", None),
            ParamInfo("name", "str | None", "keyword_only", "None"),
            ParamInfo(
                "include_dirs",
                "str | os.PathLike[str] | Sequence[str | os.PathLike[str]]",
                "keyword_only",
                "()",
            ),
        ],
    },
    "_validate_dataclass_kernel_arg": {
        "qualified_name": "dataclasses.dataclass",
        "is_class": False,
        "params": [
            ParamInfo("cls", "type", "positional", None),
            ParamInfo("frozen", "bool", "keyword_only", "False"),
            ParamInfo("kw_only", "bool", "keyword_only", "False"),
        ],
        "returns": "type",
    },
    "CatlassBaseDSL.kernel": {
        "qualified_name": "catlass.dsl.kernel",
        "params": [
            ParamInfo("fn", "Callable[..., Any] | None", "positional", "None"),
            ParamInfo("auto_sync", "str | None", "keyword_only", "None"),
        ],
        "returns": ("TlaJitFunction | Callable[[Callable[..., Any]], TlaJitFunction]"),
    },
    "CatlassBaseDSL.jit": {
        "qualified_name": "catlass.dsl.jit",
        "params": [
            ParamInfo("fn", "Callable[..., Any] | None", "positional", "None"),
        ],
        "returns": "Callable[..., Any]",
    },
}

# Dual-mode Host path of ``tla.make_tensor`` (implementation is private in
# ``tla.runtime``; public entry is ``core_api.make_tensor``, same as ``make_ptr``).
HOST_MAKE_TENSOR_DOC = """\
Directory: Host Tensor / Binding
Description:
    Bind a pointer and layout as a TLA Host tensor without DLPack (dual-mode
    ``tla.make_tensor`` outside ``@tla.kernel``). Prefer this on the launch hot
    path when address and layout metadata are already known: it skips capsule
    export/parse/ownership and does not open ``_eager_capture``.

    Build ``ptr`` / ``layout`` with the same ``tla.make_ptr`` /
    ``tla.make_layout`` APIs used in Kernel code. Inside a kernel the same
    ``tla.make_tensor`` entry emits ``!tla.tensor`` IR (see Language API).

Parameters:
    - *`ptr`*: From ``tla.make_ptr`` outside a kernel (dtype, address,
      addrspace).
    - *`layout`*: From ``tla.make_layout`` outside a kernel (``make_shape`` /
      ``make_stride``, ``layoutTag``, optional origin).
    - *`coord`*: Optional Host ``tla.make_coord`` / int tree. When omitted,
      derived from the layout (typically zeros).
    - *`loc`*: Used on the Kernel path; ignored on the Host path.

Constraints:
    - Host path rejects Kernel SSA ``Pointer`` / ``Layout``; call
      ``tla.make_ptr`` / ``tla.make_layout`` on the Host (no ``@tla.kernel``
      context).
    - Hot-path bind trusts ``make_ptr`` / ``make_layout`` metadata (no
      Kernel trait checks / RM-CM compact gate). Keep buffer metadata
      consistent; prefer ``from_dlpack`` when you need those checks.
    - Does **not** own DLPack lifetime. Keep the live buffer alive across
      launch.
    - ``ptr.data_ptr == 0`` builds an unbound sample (`_external_binding` is
      false); non-zero marks an external binding.

Example:
    ```python
    import catlass.tla as tla

    ptr = tla.make_ptr(
        tla.Float16,
        int(buf.data_ptr()),
        tla.AddressSpace.gm,
    )
    layout = tla.make_layout(
        tla.make_shape(128, 64),
        tla.make_stride(64, 1),
        layoutTag=tla.arch.RowMajor,
    )
    host = tla.make_tensor(ptr, layout)
    ```
"""

# Injected language-boundary topic (not scraped from a runtime symbol).
CONSTEXPR_CALLABLE_DOC = """\
Directory: Decorators
Description:
    Covers the "Compile-time function" row in the [`tla.kernel`](#tlakernel) parameter table
    (`tla.Constexpr[Callable[...]]`, or `tla.Constexpr` when the value is callable).
    It does not cover compile-time constants.

    Passing forms: outer `def`, `lambda`, `functools.partial`, or an `@tla.jit`-decorated
    function. Pass at `tla.compile`; omit from `compiled(...)`. Different callables
    specialize differently.

Parameters:
    None. Language-boundary notes for Constexpr Callable kernel arguments; not a
    callable Host API.

Constraints:
    Body semantics (plain `def` / `lambda` / `partial`):

    - Runs only during `tla.compile` / `dump_mlir`; DSL ops enter the current kernel's device
      IR. The body is not run again when launching `compiled(...)`.
    - When called from a kernel, the body may use only interfaces from the
      [Language API](language/index.md); see each entry's Constraints.
    - Arbitrary Host-side Python is not device computation, including but not limited to
      third-party libraries, file/network I/O, reliance on local Host state, and treating DSL
      values as Host tensors or containers.
    - TLA control flow is not supported: `tla.range`, dynamic `if` / `while`, and similar.

    Body rules when the argument is `@tla.jit` are under [`tla.jit`](#tlajit).

    **Calls inside a kernel**

    - Outer plain `def`: same body semantics as above.
    - `@tla.jit` helper: see [`tla.jit`](#tlajit); inlined into the current kernel IR during lowering.
    - Constexpr Callable argument: same as this section.

Example:
    ```python
    def abs_epilogue(value):
        return tla.abs(value)

    @tla.kernel
    def transform(src: tla.Tensor, dst: tla.Tensor, epilogue: tla.Constexpr) -> None:
        tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
                dst_tile.store(epilogue(tile.load()))

    compiled_ep = tla.compile(transform, tx, ty, abs_epilogue, options="--npu-arch 3510")
    compiled_ep(tx, ty, block_num=1)  # abs_epilogue omitted at launch
    ```
"""


def _should_collect(name: str) -> bool:
    """Public symbols, or private helpers remapped into the Host reference."""
    return (not name.startswith("_")) or name in HOST_DISPLAY_NAMES


def _apply_overrides(source_name: str, entry: APIEntry) -> APIEntry:
    meta = HOST_ENTRY_OVERRIDES.get(source_name)
    if not meta:
        return entry
    return APIEntry(
        name=entry.name,
        qualified_name=str(meta.get("qualified_name", entry.qualified_name)),
        source_line=entry.source_line,
        docstring=entry.docstring,
        source_path=entry.source_path,
        is_class=bool(meta.get("is_class", entry.is_class)),
        is_concept=bool(meta.get("is_concept", entry.is_concept)),
        params=list(meta.get("params", entry.params)),  # type: ignore[arg-type]
        returns=str(meta.get("returns", entry.returns)),
    )


def _module_qualname(path: Path) -> str:
    path = path.resolve()
    catlass_root = (PACKAGE_ROOT / "catlass").resolve()
    try:
        rel = path.relative_to(catlass_root)
        return "catlass." + ".".join(rel.with_suffix("").parts)
    except ValueError:
        rel = path.relative_to(PACKAGE_ROOT.resolve())
        return ".".join(rel.with_suffix("").parts)


def _collect_host_file(path: Path) -> dict[str, APIEntry]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mod = _module_qualname(path)
    src = path.resolve()
    entries: dict[str, APIEntry] = {}

    def maybe_add(source_name: str, entry: APIEntry) -> None:
        entry = _apply_overrides(source_name, entry)
        display = HOST_DISPLAY_NAMES.get(source_name, source_name)
        entry.name = display
        entries[display] = entry

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if not _should_collect(node.name):
                continue
            doc = ast.get_docstring(node) or ""
            if directory_path(doc) is None:
                continue
            maybe_add(
                node.name,
                function_entry(
                    node.name,
                    node,
                    qualified_name=f"{mod}.{node.name}",
                    source_path=src,
                ),
            )
        elif isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node) or ""
            if directory_path(class_doc) is not None and _should_collect(node.name):
                maybe_add(
                    node.name,
                    APIEntry(
                        name=node.name,
                        qualified_name=f"{mod}.{node.name}",
                        source_line=node.lineno,
                        docstring=class_doc,
                        is_class=True,
                        source_path=src,
                    ),
                )
            for child in node.body:
                if not isinstance(child, ast.FunctionDef):
                    continue
                raw = f"{node.name}.{child.name}"
                if not _should_collect(child.name) and raw not in HOST_DISPLAY_NAMES:
                    continue
                doc = ast.get_docstring(child) or ""
                if directory_path(doc) is None:
                    continue
                maybe_add(
                    raw,
                    function_entry(
                        raw,
                        child,
                        qualified_name=f"{mod}.{node.name}.{child.name}",
                        source_path=src,
                        drop_self=True,
                    ),
                )
    return entries


def parse_host_apis() -> dict[str, APIEntry]:
    entries: dict[str, APIEntry] = {}
    for path in HOST_SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"host API source not found: {path}")
        entries.update(_collect_host_file(path))
    # Dual-mode Host bind: public API is ``tla.make_tensor`` (core_api); Host
    # body is private in ``tla.runtime`` (aligned with ``make_ptr`` — not
    # re-exported from ``catlass.tla.runtime``).
    entries["make_tensor"] = APIEntry(
        name="tla.make_tensor",
        qualified_name="catlass.core_api.make_tensor",
        source_line=None,
        docstring=HOST_MAKE_TENSOR_DOC,
        source_path=PACKAGE_ROOT / "catlass" / "core_api.py",
        is_concept=False,
        params=[
            ParamInfo("ptr", "Pointer", "positional", None),
            ParamInfo("layout", "TlaLayout", "positional", None),
            ParamInfo("coord", "CoordLike | None", "positional", "None"),
            ParamInfo("loc", "mlir_ir.Location | None", "keyword_only", "None"),
        ],
        returns="TlaTensor",
    )
    # Language-boundary topic: kept in the generator, not as a runtime stub.
    entries["Constexpr Callable arguments"] = APIEntry(
        name="Constexpr Callable arguments",
        qualified_name="Host language boundary · Constexpr Callable",
        source_line=None,
        docstring=CONSTEXPR_CALLABLE_DOC,
        source_path=None,
        is_concept=True,
    )
    for entry in entries.values():
        if entry.is_concept:
            continue
        if not entry.name.startswith(("tla.", "dataclasses.")):
            entry.name = f"tla.{entry.name}"
    return entries


def generate(*, docs_dir: Path | None = None) -> str:
    docs_dir = docs_dir or OUTPUT_PATH.parent.parent
    markdown = render_reference(
        parse_host_apis(),
        docs_dir=docs_dir,
        title="CATLASS Python Program Lifecycle Reference",
        intro=[
            "Start with the [API Quick Reference](quick_reference.md) to choose an API by task "
            "and execution context. This document covers the **CATLASS Python program lifecycle**: "
            "kernel entry and language-boundary declarations, `tla.compile` / "
            "`JitCompiledFunction` launch, and Host tensor binding. "
            "Environment variables are in "
            "[Environment Variables](../kernel_development/core_concepts/env_vars.md). "
            "Language APIs live in the [Language Reference](language/index.md).",
            "Interface descriptions and examples come from each API's source docstring "
            "(`Directory:` plus `Description:` / `Parameters:` / `Constraints:` / `Example:`).",
            "These APIs are called from Python Host scripts, **outside** a `@tla.kernel` "
            "function body.",
        ],
        header_sources=(
            "Do not edit manually. Update Host docstrings in catlass/dsl.py,",
            "catlass/catlass_dsl/catlass.py, catlass/base_dsl/compiler.py,",
            "catlass/base_dsl/jit_executor.py, catlass/execution_lowering.py,",
            "catlass/tla/ffi.py, and catlass/tla/runtime.py. Constexpr Callable",
            "language-boundary text lives in this generator (`CONSTEXPR_CALLABLE_DOC`).",
        ),
        leftovers_title="Other Host APIs",
        leftovers_blurb=(
            "APIs still collected from Host sources but not yet filed under "
            "the directory tree above."
        ),
        directory_order=DIRECTORY_ORDER,
        directory_intros=DIRECTORY_INTROS,
        default_source_path=DEFAULT_SOURCE_PATH,
        generated_by=GENERATED_BY,
    )
    return published_wording(markdown)


def generate_pages(*, docs_dir: Path | None = None) -> dict[Path, str]:
    """Return the split English Host reference keyed by relative path."""
    docs_dir = docs_dir or OUTPUT_PATH.parent.parent
    markdown = generate(docs_dir=docs_dir)
    return split_reference_document(
        markdown,
        index_path="runtime/index.md",
        page_files=HOST_PAGE_FILES,
        page_titles=HOST_PAGE_TITLES,
        reference_label="Program Lifecycle",
        quick_label="API Quick Reference",
    )


def main() -> int:
    return run_cli_pages(
        description=__doc__,
        default_output=OUTPUT_PATH,
        generate_pages_fn=generate_pages,
    )


if __name__ == "__main__":
    raise SystemExit(main())
