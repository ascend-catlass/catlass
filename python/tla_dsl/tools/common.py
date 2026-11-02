"""Shared AST parsing and Markdown rendering for CATLASS Python API references."""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Callable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # python/tla_dsl
HIDDEN_PARAMETERS = frozenset({"loc"})
SECTION_LABELS = ("Description", "Parameters", "Constraints", "Example")
ROOT_TITLE = "API Reference"


@dataclass(frozen=True)
class DocNode:
    title: str
    apis: tuple[str, ...] = ()
    children: tuple["DocNode", ...] = ()
    intro: str = ""


@dataclass
class ParamInfo:
    name: str
    annotation: str
    kind: str  # positional | var_positional | keyword_only | var_keyword
    default: str | None


@dataclass
class APIEntry:
    name: str
    qualified_name: str
    source_line: int | None
    params: list[ParamInfo] = field(default_factory=list)
    returns: str = ""
    docstring: str = ""
    is_class: bool = False
    is_namespace: bool = False
    is_env: bool = False
    # Language-boundary topic with no callable prototype (Host reference only).
    is_concept: bool = False
    source_path: Path | None = None


class MissingAPIDocError(RuntimeError):
    """Public API is missing required English docstring sections."""


def unparse(node: ast.AST | None, *, missing: str = "Any") -> str:
    if node is None:
        return missing
    try:
        return ast.unparse(node).replace("typing.", "")
    except Exception:
        return missing


def const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def decorator_basename(dec: ast.expr) -> str | None:
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return None


def has_dsl_user_op(decorators: list[ast.expr]) -> bool:
    return any(decorator_basename(d) == "dsl_user_op" for d in decorators)


def params_from_args(args: ast.arguments) -> list[ParamInfo]:
    params: list[ParamInfo] = []
    pos_args = list(args.posonlyargs) + list(args.args)
    default_offset = len(pos_args) - len(args.defaults)

    for index, arg in enumerate(pos_args):
        default = None
        if index >= default_offset:
            default = unparse(args.defaults[index - default_offset], missing="...")
        params.append(
            ParamInfo(arg.arg, unparse(arg.annotation), "positional", default)
        )

    if args.vararg is not None:
        params.append(
            ParamInfo(
                args.vararg.arg,
                unparse(args.vararg.annotation),
                "var_positional",
                None,
            )
        )

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        default = unparse(default_node, missing="...") if default_node else None
        params.append(
            ParamInfo(arg.arg, unparse(arg.annotation), "keyword_only", default)
        )

    if args.kwarg is not None:
        params.append(
            ParamInfo(
                args.kwarg.arg,
                unparse(args.kwarg.annotation),
                "var_keyword",
                None,
            )
        )

    return [p for p in params if p.name not in HIDDEN_PARAMETERS]


def function_entry(
    name: str,
    node: ast.FunctionDef,
    *,
    qualified_name: str | None = None,
    source_path: Path | None = None,
    drop_self: bool = False,
) -> APIEntry:
    params = params_from_args(node.args)
    if drop_self:
        params = [p for p in params if p.name != "self"]
    return APIEntry(
        name=name,
        qualified_name=qualified_name or f"catlass.core_api.{name}",
        source_line=node.lineno,
        params=params,
        returns=unparse(node.returns, missing="") if node.returns else "",
        docstring=ast.get_docstring(node) or "",
        source_path=source_path,
    )


def parse_all_names(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    return {
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
    return set()


def directory_path(doc: str) -> str | None:
    """Return the ``Directory:`` path, or None if missing."""
    if not doc:
        return None
    match = re.search(r"(?m)^Directory:\s*(.+?)\s*$", dedent(doc).strip())
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split("/") if p.strip()]
    return " / ".join(parts) if parts else None


def docstring_sections(doc: str) -> dict[str, str]:
    """Split a docstring into the four fixed English sections."""
    sections = {label: "" for label in SECTION_LABELS}
    if not doc:
        return sections
    text = re.sub(r"(?m)^\s*Examples\s*:", "Example:", dedent(doc).strip())
    hits = [
        (label, m.start(), m.end())
        for label in SECTION_LABELS
        for m in [re.search(rf"(?m)^[ \t]*{label}[ \t]*:", text)]
        if m
    ]
    hits.sort(key=lambda item: item[1])
    for i, (label, _start, content_start) in enumerate(hits):
        content_end = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        body = dedent(text[content_start:content_end]).strip()
        if label == "Example":
            code = re.search(r"```(?:python)?\n(.*?)```", body, flags=re.S)
            sections[label] = dedent(code.group(1)).strip() if code else body
        else:
            sections[label] = body
    return sections


def docstring_tail(doc: str) -> str:
    """Return the markdown an op appends after its last fenced code block.

    `_modify_doc_mmad` / `_modify_doc_copy`, which the reference generator runs
    through `__modify_doc__`, append supported-route / dtype tables at the end
    of the docstring (after the Example fence in the source text). The renderer
    places that tail between Constraints and Example so the tables read as
    path/constraint documentation rather than an afterthought.
    """
    text = dedent(doc)
    end = text.rfind("```")
    if end == -1:
        return ""
    lines = [line.strip() for line in text[end + 3 :].splitlines()]
    return "\n".join(lines).strip()


def require_docs(entries: dict[str, APIEntry]) -> None:
    problems: list[str] = []
    for name in sorted(entries):
        entry = entries[name]
        loc = entry.qualified_name + (
            f" (approx. L{entry.source_line})" if entry.source_line else ""
        )
        if directory_path(entry.docstring) is None:
            problems.append(f"- `{name}` @ {loc}: missing `Directory:`")
        sections = docstring_sections(entry.docstring)
        for label in SECTION_LABELS:
            if not sections[label]:
                problems.append(f"- `{name}` @ {loc}: missing or empty `{label}:`")
    if problems:
        raise MissingAPIDocError(
            "Incomplete API docs. Add `Directory:` plus docstring sections "
            "`Description:` / `Parameters:` / `Constraints:` / `Example:` "
            "in source:\n" + "\n".join(problems)
        )


def _path_prefixes(path: str) -> list[str]:
    parts = path.split(" / ")
    return [" / ".join(parts[:i]) for i in range(1, len(parts) + 1)]


def build_doc_tree(
    entries: dict[str, APIEntry],
    *,
    directory_order: list[str],
    directory_intros: dict[str, str],
    default_source_path: Path,
) -> DocNode:
    """Group APIs by ``Directory:`` and order sections via ``directory_order``."""
    leaf_apis: dict[str, list[str]] = {}
    first_line: dict[str, int] = {}
    ranked = {path: idx for idx, path in enumerate(directory_order)}

    pending = sorted(
        (
            (
                str(entry.source_path or default_source_path),
                entry.source_line or 0,
                name,
                directory_path(entry.docstring),
            )
            for name, entry in entries.items()
        )
    )
    for _path, line, name, directory in pending:
        if not directory:
            continue
        leaf_apis.setdefault(directory, []).append(name)
        first_line.setdefault(directory, line)

    path_set: set[str] = set()
    for leaf in leaf_apis:
        path_set.update(_path_prefixes(leaf))
    for path in directory_intros:
        path_set.update(_path_prefixes(path))

    def order_key(path: str) -> tuple[int, int, str]:
        if path in ranked:
            return (0, ranked[path], path)
        prefix = f"{path} / "
        child_ranks = [ranked[p] for p in ranked if p.startswith(prefix)]
        if child_ranks:
            return (0, min(child_ranks), path)
        return (1, first_line.get(path, 10**9), path)

    def build_node(path: str) -> DocNode:
        title = path.rsplit(" / ", 1)[-1]
        prefix = f"{path} / "
        child_titles = sorted(
            {
                candidate[len(prefix) :]
                for candidate in path_set
                if candidate.startswith(prefix)
                and " / " not in candidate[len(prefix) :]
            },
            key=lambda t: order_key(prefix + t),
        )
        children = tuple(build_node(prefix + t) for t in child_titles)
        return DocNode(
            title=title,
            # Keep APIs whose Directory is exactly this path even when nested
            # subsections exist (e.g. System Variable Access + Layout Tags).
            apis=tuple(leaf_apis.get(path, ())),
            children=children,
            intro=directory_intros.get(path, ""),
        )

    top = sorted({p for p in path_set if " / " not in p}, key=order_key)
    return DocNode(title=ROOT_TITLE, children=tuple(build_node(p) for p in top))


def _callable_display_name(name: str) -> bool:
    """True if ``name`` is a dotted identifier path (``foo``, ``Numeric.to``)."""
    return bool(name) and all(part.isidentifier() for part in name.split("."))


def format_signature(entry: APIEntry, *, default_source_path: Path) -> str:
    if entry.is_env:
        return entry.name
    if entry.is_namespace:
        return f"tla.{entry.name}"
    if entry.is_class:
        return f"class tla.{entry.name}"
    # Expression-form display names (e.g. ``~Integer``, ``tensor[i]``): as-is,
    # no ``()``. Assignment spellings (``tensor[i] = value``) omit ``->``.
    if not _callable_display_name(entry.name):
        if entry.returns and "=" not in entry.name:
            return f"{entry.name} -> {entry.returns}"
        return entry.name

    parts: list[str] = []
    saw_kwonly = False
    for param in entry.params:
        if param.kind == "keyword_only" and not saw_kwonly:
            parts.append("*")
            saw_kwonly = True
        if param.kind == "var_positional":
            piece = f"*{param.name}: {param.annotation}"
        elif param.kind == "var_keyword":
            piece = f"**{param.name}: {param.annotation}"
        else:
            piece = f"{param.name}: {param.annotation}"
            if param.default is not None:
                piece += f" = {param.default}"
        parts.append(piece)

    if entry.qualified_name.startswith("dataclasses."):
        callee = entry.qualified_name
    elif entry.name.startswith("tla."):
        callee = entry.name
    elif entry.name.startswith("Tensor."):
        method = entry.name.split(".", 1)[1]
        src = entry.source_path or default_source_path
        if src.name == "runtime.py" and "tla" in src.parts:
            callee = f"tensor.{method}"
        else:
            callee = f"tile.{method}"
    elif "." in entry.name:
        head, _sep, _rest = entry.name.partition(".")
        # PascalCase head => Class.method (Host JIT types); keep bare.
        # lowercase head => module attr (e.g. vec.func) => call as tla.vec.func.
        if head[:1].isupper():
            callee = entry.name
        else:
            callee = f"tla.{entry.name}"
    else:
        callee = f"tla.{entry.name}"
    sig = f"{callee}({', '.join(parts)})"
    return f"{sig} -> {entry.returns}" if entry.returns else sig


def source_markdown(
    entry: APIEntry, docs_dir: Path, *, default_source_path: Path
) -> str:
    # Concept / catalog entries: link to the declaration when we have a line;
    # otherwise omit Source (no "not a callable API" placeholder).
    if entry.is_concept and entry.source_line is None:
        return ""
    if entry.source_line is None:
        if entry.is_env:
            return f"**Source:** environment variable `{entry.name}`"
        return f"**Source:** `{entry.qualified_name}`"
    src = entry.source_path or default_source_path
    rel = Path(os.path.relpath(src, docs_dir))
    link = f"[`{entry.qualified_name}`]({rel.as_posix()}#L{entry.source_line})"
    if entry.is_env:
        return f"**Source:** environment variable {link}"
    return f"**Source:** {link}"


def _anchor(title: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff\- ]", "", title).strip().replace(" ", "-").lower()


def numbered_title(number: str | None, title: str) -> str:
    if not number:
        return title
    return f"{number} {title}" if "." in number else f"{number}. {title}"


def render_entry(
    entry: APIEntry,
    heading_level: int,
    docs_dir: Path,
    *,
    default_source_path: Path,
) -> str:
    sections = docstring_sections(entry.docstring)
    for label in SECTION_LABELS:
        if not sections[label]:
            raise MissingAPIDocError(
                f"`{entry.name}` missing {label}:; please update {entry.qualified_name}"
            )
    h = "#" * heading_level
    heading = f"{h} {entry.name}" if entry.is_concept else f"{h} `{entry.name}`"
    parts = [heading, ""]
    source = source_markdown(entry, docs_dir, default_source_path=default_source_path)
    if source:
        parts.extend([source, ""])
    parts.extend(
        [
            "Description:",
            "",
            sections["Description"],
            "",
        ]
    )
    if not entry.is_concept:
        parts.extend(
            [
                "Prototype:",
                "",
                "```python",
                format_signature(entry, default_source_path=default_source_path),
                "```",
                "",
                "Parameters:",
                "",
                sections["Parameters"],
                "",
            ]
        )
    parts.extend(
        [
            "Constraints:",
            "",
            sections["Constraints"],
            "",
        ]
    )
    tail = docstring_tail(entry.docstring)
    if tail:
        parts.extend([tail, ""])
    parts.extend(
        [
            "Example:",
            "",
            "```python",
            sections["Example"],
            "```",
            "",
            "---",
            "",
        ]
    )
    return "\n".join(parts)


def iter_apis(node: DocNode) -> list[str]:
    names = list(node.apis)
    for child in node.children:
        names.extend(iter_apis(child))
    return names


def render_toc(tree: DocNode) -> str:
    lines = ["## Table of Contents", ""]

    def walk(node: DocNode, number: str) -> None:
        heading = numbered_title(number, node.title)
        indent = "  " * number.count(".")
        lines.append(f"{indent}- [{heading}](#{_anchor(heading)})")
        for idx, child in enumerate(node.children, start=1):
            walk(child, f"{number}.{idx}")

    for idx, child in enumerate(tree.children, start=1):
        walk(child, str(idx))
    lines.append("")
    return "\n".join(lines)


def render_doc_node(
    node: DocNode,
    entries: dict[str, APIEntry],
    heading_level: int,
    docs_dir: Path,
    section_number: str,
    *,
    default_source_path: Path,
) -> str:
    parts = [f"{'#' * heading_level} {numbered_title(section_number, node.title)}", ""]
    if node.intro:
        parts.extend([node.intro, ""])
    # Nested catalog subsections first, then APIs whose Directory is exactly
    # this path (e.g. System Variable Access: Layout Tags / Pipes / …, then
    # `arch.block_idx` / `arch.get_dyn_ub`).
    for idx, child in enumerate(node.children, start=1):
        parts.append(
            render_doc_node(
                child,
                entries,
                heading_level + 1,
                docs_dir,
                f"{section_number}.{idx}",
                default_source_path=default_source_path,
            )
        )
    for name in node.apis:
        if name in entries:
            parts.append(
                render_entry(
                    entries[name],
                    heading_level + 1,
                    docs_dir,
                    default_source_path=default_source_path,
                )
            )
    return "\n".join(parts).rstrip() + "\n"


def render_reference(
    entries: dict[str, APIEntry],
    *,
    docs_dir: Path,
    title: str,
    intro: list[str],
    header_sources: tuple[str, ...],
    leftovers_title: str,
    leftovers_blurb: str,
    directory_order: list[str],
    directory_intros: dict[str, str],
    default_source_path: Path,
    generated_by: str,
) -> str:
    require_docs(entries)
    tree = build_doc_tree(
        entries,
        directory_order=directory_order,
        directory_intros=directory_intros,
        default_source_path=default_source_path,
    )
    used = set(iter_apis(tree))
    leftovers = sorted(
        (e for name, e in entries.items() if name not in used),
        key=lambda e: (str(e.source_path or ""), e.source_line or 0, e.name),
    )

    header = [
        "<!--",
        f"This file is generated by {generated_by}.",
        *header_sources,
        "-->",
        "",
        f"# {title}",
        "",
        *[paragraph.rstrip("\n") + "\n" for paragraph in intro],
        "",
        "---",
        "",
        render_toc(tree),
        "---",
        "",
    ]

    body = [
        render_doc_node(
            child,
            entries,
            2,
            docs_dir,
            str(idx),
            default_source_path=default_source_path,
        )
        for idx, child in enumerate(tree.children, start=1)
    ]
    if leftovers:
        body.append(f"## {leftovers_title}\n")
        body.append(leftovers_blurb + "\n")
        body.extend(
            render_entry(e, 3, docs_dir, default_source_path=default_source_path)
            for e in leftovers
        )

    return "\n".join(header + body).rstrip() + "\n"


_NUMBERED_HEADING = re.compile(r"^(#{2,6})\s+(\d+(?:\.\d+)*)\.?\s+(.+?)\s*$")
_MARKDOWN_LINK = re.compile(r"(\]\(\s*)([^)\s#]+)(#[^)\s]*)?(\s*\))")


def _trim_section(lines: list[str]) -> list[str]:
    """Drop separator rules and surrounding blank lines from one section."""
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and lines[-1].strip() == "---":
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return lines


def _shift_headings(lines: list[str], amount: int) -> list[str]:
    shifted: list[str] = []
    for line in lines:
        match = re.match(r"^(#{2,6})(\s+.*)$", line)
        if match:
            level = max(1, len(match.group(1)) - amount)
            line = "#" * level + match.group(2)
        shifted.append(line)
    return shifted


def _prefix_relative_links(lines: list[str], prefix: str) -> list[str]:
    """Rebase links from the old single-page location to a split page."""

    def replace(match: re.Match[str]) -> str:
        target = match.group(2)
        if target.startswith(("http://", "https://", "/", "mailto:")):
            return match.group(0)
        return f"{match.group(1)}{prefix}{target}{match.group(3) or ''}{match.group(4)}"

    return [_MARKDOWN_LINK.sub(replace, line) for line in lines]


def split_reference_document(
    markdown: str,
    *,
    index_path: str,
    page_files: dict[str, str],
    page_titles: dict[str, str] | None = None,
    reference_label: str = "API Reference",
    quick_label: str = "Quick Reference",
) -> dict[Path, str]:
    """Split one generated reference into section pages plus a landing page.

    ``page_files`` maps numeric section ids to output paths relative to the
    reference directory. A mapping for ``"5"`` whose path ends in ``index.md``
    turns section 5 into a group page and emits its numbered H3 subsections as
    separate pages.
    """
    lines = markdown.splitlines()
    h1_index = next(i for i, line in enumerate(lines) if line.startswith("# "))
    header_end = next(
        i for i in range(h1_index + 1, len(lines)) if lines[i].strip() == "---"
    )
    comment = lines[:h1_index]
    title_line = lines[h1_index]
    intro = _trim_section(lines[h1_index + 1 : header_end])

    top_sections: list[tuple[str, str, int, int]] = []
    for i, line in enumerate(lines):
        match = _NUMBERED_HEADING.match(line)
        if match and len(match.group(1)) == 2:
            top_sections.append((match.group(2), match.group(3), i, len(lines)))
    for idx in range(len(top_sections) - 1):
        number, title, start, _end = top_sections[idx]
        top_sections[idx] = (number, title, start, top_sections[idx + 1][2])
    if top_sections:
        last_number, last_title, last_start, _end = top_sections[-1]
        top_sections[-1] = (last_number, last_title, last_start, len(lines))

    pages: list[tuple[str, str, str, list[str]]] = []
    for number, title, start, end in top_sections:
        if number not in page_files:
            continue
        block = _trim_section(lines[start:end].copy())
        path = page_files[number]
        display_title = (page_titles or {}).get(number, title)
        group = path.endswith("/index.md")
        if group:
            first_child = next(
                (
                    i
                    for i, line in enumerate(block[1:], start=1)
                    if (m := _NUMBERED_HEADING.match(line))
                    and m.group(1) == "###"
                    and m.group(2).startswith(f"{number}.")
                ),
                None,
            )
            intro_block = _trim_section(block[:first_child]) if first_child else block
            pages.append((number, display_title, path, intro_block))
            if first_child is not None:
                child_starts: list[int] = []
                for i, line in enumerate(block[first_child:], start=first_child):
                    match = _NUMBERED_HEADING.match(line)
                    if match and match.group(1) == "###":
                        child_starts.append(i)
                for cidx, child_start in enumerate(child_starts):
                    child_end = (
                        child_starts[cidx + 1]
                        if cidx + 1 < len(child_starts)
                        else len(block)
                    )
                    child_match = _NUMBERED_HEADING.match(block[child_start])
                    assert child_match is not None
                    child_number = child_match.group(2)
                    child_path = page_files.get(child_number)
                    if child_path is None:
                        continue
                    pages.append(
                        (
                            child_number,
                            (page_titles or {}).get(child_number, child_match.group(3)),
                            child_path,
                            _trim_section(block[child_start:child_end].copy()),
                        )
                    )
        else:
            pages.append((number, display_title, path, block))

    result: dict[Path, str] = {}
    index_page = Path(index_path)
    table = ["## Reference Pages", "", "| Section | Page |", "| --- | --- |"]
    for number, title, path, _block in pages:
        target = Path(os.path.relpath(Path(path), index_page.parent)).as_posix()
        table.append(f"| {number} | [{title}]({target}) |")
    table.extend(["", "---", ""])
    index_prefix = "../" * len(index_page.parent.parts)
    index_body = (
        comment
        + [title_line, ""]
        + _prefix_relative_links(intro, index_prefix)
        + [""]
        + table
    )
    result[index_page] = "\n".join(index_body).rstrip() + "\n"

    for number, title, path, block in pages:
        page = Path(path)
        nav_order = int(number.replace(".", ""))
        shift = 1 if "." not in number else 2
        depth = len(page.parent.parts)
        prefix = "../" * depth
        shifted = _shift_headings(
            _prefix_relative_links(_trim_section(block.copy()), prefix), shift
        )
        if not shifted:
            continue
        back = Path(os.path.relpath(index_page, page.parent)).as_posix()
        quick = Path(os.path.relpath("quick_reference.md", page.parent)).as_posix()
        breadcrumb = f"[{reference_label}]({back}) | [{quick_label}]({quick})"
        page_body = (
            ["---", f"nav_order: {nav_order}", "---", ""]
            + comment
            + [f"# {title}", "", breadcrumb, ""]
            + _trim_section(shifted[1:].copy())
        )
        if path.endswith("/index.md"):
            children = [
                item
                for item in pages
                if item[2].startswith(f"{page.parent.as_posix()}/") and item[2] != path
            ]
            if children:
                page_body.extend(
                    ["", "## Subtopics", "", "| Section | Page |", "| --- | --- |"]
                )
                for child_number, child_title, child_path, _child_block in children:
                    child_target = Path(
                        os.path.relpath(child_path, page.parent)
                    ).as_posix()
                    page_body.append(
                        f"| {child_number} | [{child_title}]({child_target}) |"
                    )
        result[page] = "\n".join(page_body).rstrip() + "\n"
    return result


_PUBLISHED_WORDING_PROTECTIONS = (
    "CATLASS_DSL_",
    "DSLOptimizationWarning",
    "CatlassBaseDSL",
    "BaseDSL",
    "TlaDSL",
    "DSLLocation",
    "_TRUSTED_DSL_",
    "_INTERNAL_CHECKED_DSL_",
    "CuTe DSL",
    "python/tla_dsl",
    "dsl_development",
    "dsl_tutorials",
)


def published_wording(markdown: str) -> str:
    """Normalize terminology in generated API prose only.

    Source docstrings keep the project's internal DSL terminology. The
    generated API reference uses Python wording, except for identifiers,
    environment variables, paths, and external product names.
    """
    placeholders: dict[str, str] = {}
    for index, token in enumerate(_PUBLISHED_WORDING_PROTECTIONS):
        placeholders[token] = f"@@W{index}@@"
    for token, placeholder in placeholders.items():
        markdown = markdown.replace(token, placeholder)

    markdown = markdown.replace(
        "CATLASS Python (hereafter abbreviated as DSL)", "CATLASS Python"
    )
    markdown = markdown.replace("CATLASS Python （以下简称为DSL）", "CATLASS Python")
    markdown = markdown.replace("TLA DSL", "CATLASS Python")
    markdown = markdown.replace("Tla DSL", "CATLASS Python")
    markdown = markdown.replace("Tla base DSL", "CATLASS Python")
    markdown = markdown.replace("the DSL", "CATLASS Python")
    markdown = markdown.replace("in the DSL", "in CATLASS Python")
    for old, new in (
        ("DSL API", "Python API"),
        ("DSL call", "Python call"),
        ("DSL region", "Python region"),
        ("DSL lowering", "Python lowering"),
        ("DSL operation", "Python operation"),
        ("DSL op", "Python op"),
        ("DSL type", "Python type"),
        ("DSL value", "Python value"),
        ("DSL frontend", "Python frontend"),
        ("DSL", "Python"),
        ("Python Python", "Python"),
    ):
        markdown = markdown.replace(old, new)

    for old, new in (
        ("the Python does not support", "CATLASS Python does not support"),
        ("in the Python into", "in CATLASS Python into"),
        ("abstracted in the Python as", "abstracted as"),
        ("the Python depends on", "CATLASS Python depends on"),
        ("building the Python and", "building CATLASS Python and"),
        ("developing and testing the Python", "developing and testing CATLASS Python"),
        ("compilation of the Python depends", "compilation of CATLASS Python depends"),
        ("developed with the Python", "developed with CATLASS Python"),
        ("the Python expands the arguments", "CATLASS Python expands the arguments"),
        ("handled by the Python.", "handled by CATLASS Python."),
        ("the Python pipeline", "the CATLASS Python pipeline"),
        ("the Python decorator", "the CATLASS Python decorator"),
        ("to the Python.", "to CATLASS Python."),
        ("the Python constructor", "the CATLASS Python constructor"),
        ("in the Python subproject", "in the CATLASS Python subproject"),
        ("builds the Python, and runs", "builds CATLASS Python and runs"),
        ("from the Python --", "from CATLASS Python --"),
        ("the Python type template", "the CATLASS Python type template"),
        ("the Python call", "the CATLASS Python call"),
        ("the Python compilation pipeline", "the CATLASS Python compilation pipeline"),
        ("customize the Python", "customize CATLASS Python"),
        ("reachable from the Python --", "reachable from CATLASS Python --"),
        ("relative to the Python root", "relative to the CATLASS Python root"),
        ("return to the Python subproject", "return to the CATLASS Python subproject"),
        ("If the Python reimplemented", "If CATLASS Python reimplemented"),
        (
            "between the Python and the AscendC",
            "between CATLASS Python and the AscendC",
        ),
    ):
        markdown = markdown.replace(old, new)

    for token, placeholder in placeholders.items():
        markdown = markdown.replace(placeholder, token)
    return markdown


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"generated {path.relative_to(PACKAGE_ROOT)}")


def check_output(path: Path, content: str) -> bool:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing == content:
        print(f"up to date {path.relative_to(PACKAGE_ROOT)}")
        return True
    diff = "\n".join(
        difflib.unified_diff(
            existing.splitlines(),
            content.splitlines(),
            fromfile=str(path),
            tofile="generated",
            lineterm="",
        )
    )
    print(diff)
    print(f"out of date {path.relative_to(PACKAGE_ROOT)}")
    return False


def run_cli_pages(
    *,
    description: str,
    default_output: Path,
    generate_pages_fn: Callable[..., dict[Path, str]],
) -> int:
    """Write or check a reference split across multiple Markdown pages."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output landing page (default: {default_output.relative_to(PACKAGE_ROOT)})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write files; fail if any generated Markdown is stale.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PACKAGE_ROOT / args.output
    reference_root = output.parent.parent
    try:
        pages = generate_pages_fn(docs_dir=reference_root)
    except MissingAPIDocError as exc:
        print(exc)
        return 1
    ok = True
    for relative_path, content in pages.items():
        path = reference_root / relative_path
        if args.check:
            ok = check_output(path, content) and ok
        else:
            write_output(path, content)
    return 0 if ok else 1


def run_cli(
    *,
    description: str,
    default_output: Path,
    generate_fn: Callable[..., str],
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output path (default: {default_output.relative_to(PACKAGE_ROOT)})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write files; fail if generated Markdown is stale.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PACKAGE_ROOT / args.output
    try:
        content = generate_fn(docs_dir=output.parent)
    except MissingAPIDocError as exc:
        print(exc)
        return 1
    if args.check:
        return 0 if check_output(output, content) else 1
    write_output(output, content)
    return 0
