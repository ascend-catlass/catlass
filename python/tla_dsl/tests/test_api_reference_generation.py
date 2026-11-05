from __future__ import annotations

import ast
import difflib
import importlib
import pathlib
import re
import sys

import pytest


def _generated_kernel_section(relative: str, api_name: str) -> str:
    package_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root / "tools"))
    generator = importlib.import_module("generate_kernel_api_reference")
    page = generator.generate_pages(docs_dir=package_root / "docs" / "en" / "api")[
        pathlib.Path(relative)
    ]
    heading = re.search(rf"(?m)^#{{2,6}} `{re.escape(api_name)}`$", page)
    assert heading is not None, f"missing API section: {api_name}"
    section = page[heading.end() :]
    return re.split(r"(?m)^#{2,6} ", section, maxsplit=1)[0]


def test_generated_comparison_keeps_mask_outside_the_mode_table() -> None:
    section = _generated_kernel_section("language/simd/compare_select.md", "tla.cmp")
    # Without a blank line the Markdown tables extension consumes this parameter
    # as an extra table row instead of a separate list item.
    assert re.search(r"(?m)^[ \t]*\|[^\n]*\|[ \t]*\n(?:[ \t]*\n)+- `mask`", section)


@pytest.mark.parametrize(
    ("relative", "api_name"),
    [
        ("language/data_movement.md", "tla.Tensor.__getitem__"),
        ("language/system.md", "tla.arch.thread_idx"),
        ("language/system.md", "tla.arch.thread_block_dim"),
    ],
)
def test_generated_reference_separates_return_lists(
    relative: str, api_name: str
) -> None:
    section = _generated_kernel_section(relative, api_name)
    # A list cannot interrupt the preceding paragraph in Python-Markdown.
    assert re.search(r"(?m)^Returns:\n(?:[ \t]*\n)+- ", section)


def _assert_generated_pages_are_current(module_name: str, regen_hint: str) -> None:
    package_root = pathlib.Path(__file__).resolve().parents[1]
    docs_root = package_root / "docs" / "en" / "api"
    sys.path.insert(0, str(package_root / "tools"))
    gen = importlib.import_module(module_name)

    pages = gen.generate_pages(docs_dir=docs_root)
    assert pages, "generator returned no pages"
    for relative, expected in pages.items():
        doc = docs_root / relative
        assert doc.is_file(), f"missing checked-in API reference page: {doc}"
        existing = doc.read_text(encoding="utf-8")
        if existing == expected:
            continue
        diff = "\n".join(
            difflib.unified_diff(
                existing.splitlines(),
                expected.splitlines(),
                fromfile=str(doc),
                tofile="generated",
                lineterm="",
            )
        )
        raise AssertionError(
            f"{relative} is out of date. Regenerate with:\n"
            f"  cd python/tla_dsl && {regen_hint}\n"
            f"{diff}"
        )


def test_generated_api_reference_is_current() -> None:
    """Fail when the split Kernel reference pages are stale."""
    _assert_generated_pages_are_current(
        "generate_kernel_api_reference",
        "python3 tools/generate_kernel_api_reference.py",
    )


def test_generated_host_api_reference_is_current() -> None:
    """Fail when the split Host reference pages are stale."""
    _assert_generated_pages_are_current(
        "generate_host_api_reference",
        "python3 tools/generate_host_api_reference.py",
    )


def test_split_reference_document_creates_topic_pages() -> None:
    """Split a generated Markdown document into a landing page and topics."""
    package_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root / "tools"))
    common = importlib.import_module("common")
    markdown = """<!-- generated -->
# Reference

Intro.

---

## Table of Contents

- [One](#one)

---

## One

First topic.

---

## Group

Group intro.

### Child

Child topic.

---

## Three

Third topic.

---
"""
    pages = common.split_reference_document(
        markdown,
        index_path="ref/index.md",
        page_files={
            "1": "ref/one.md",
            "2": "ref/group/index.md",
            "2.1": "ref/group/child.md",
            "3": "ref/three.md",
        },
    )
    assert {path.as_posix() for path in pages} == {
        "ref/index.md",
        "ref/one.md",
        "ref/group/index.md",
        "ref/group/child.md",
        "ref/three.md",
    }
    assert "# Child" in pages[pathlib.Path("ref/group/child.md")]
    assert "[Child](child.md)" in pages[pathlib.Path("ref/group/index.md")]


def test_host_api_reference_includes_extern() -> None:
    """Keep the public extern decorator in the generated Host API reference."""
    package_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root / "tools"))
    generator = importlib.import_module("generate_host_api_reference")

    extern = generator.parse_host_apis()["extern"]

    assert extern.qualified_name == "catlass.tla.ffi.extern"


def test_scalar_aware_aliases_have_public_signatures() -> None:
    """Do not expose the narrower vector-helper signature for scalar overloads."""
    package_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root / "tools"))
    generator = importlib.import_module("generate_kernel_api_reference")
    tree = ast.parse(
        (package_root / "catlass" / "core_api.py").read_text(encoding="utf-8")
    )

    unary = generator._collect_unary_aliases(
        tree,
        {"exp", "log", "sqrt", "abs", "neg"},
        generator._unary_op_template(tree),
    )
    for name in ("exp", "log", "sqrt", "abs"):
        assert unary[name].params[0].annotation == "VectorSSA | Numeric"
        assert unary[name].returns == "VectorSSA | Numeric"
    assert unary["neg"].params[0].annotation == "VectorSSA"
    assert unary["neg"].returns == "VectorSSA"

    binary = generator._collect_scalar_aware_binary_aliases(tree, {"max", "min"})
    assert set(binary) == {"max", "min"}
    assert all(entry.returns == "VectorSSA | Numeric" for entry in binary.values())
