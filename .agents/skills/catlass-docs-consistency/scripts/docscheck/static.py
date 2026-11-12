#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
"""Static documentation checks that never compile or run an example."""

from __future__ import annotations

from . import discover as d


def check_repo_paths(doc: d.Doc, index: d.RepoIndex) -> list[d.Finding]:
    findings: list[d.Finding] = []
    seen: set[tuple[str, int]] = set()
    created = d._cleaned_created_paths(doc)

    def add(path: str, line: int, following: str) -> None:
        path = d._clean_path_token(path)
        if not path or d._strip_placeholders(path):
            return
        if path.endswith("/") or following in d.PLACEHOLDER_CHARS:
            return
        if any(path == item or path.startswith(item + "/") for item in created):
            return
        if (path, line) in seen:
            return
        seen.add((path, line))
        if not index.exists(path, doc.path.parent):
            findings.append(
                d.Finding(
                    path=doc.rel,
                    line=line,
                    check="repo-path",
                    message=f"referenced path '{path}' does not exist",
                    section=doc.section_at(line),
                )
            )

    # Inline code spans outside fenced blocks.
    in_fence = False
    fence_marker = ""
    for number, line in enumerate(doc.text.splitlines(), start=1):
        match = d.FENCE_RE.match(line)
        if match:
            marker = match.group("fence")[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
                continue
            if marker == fence_marker:
                in_fence = False
                continue
        if in_fence:
            continue
        for candidate in d.INLINE_CODE_RE.findall(line):
            for path_match in d.REPO_PATH_RE.finditer(candidate):
                following = candidate[path_match.end() : path_match.end() + 1]
                add(path_match.group(1), number, following)

    # Paths written inside fenced blocks, ignoring comments and output dumps.
    for block in doc.blocks:
        for offset, line in enumerate(block.lines):
            stripped = line.lstrip()
            if stripped.startswith(("#", "//")):
                continue
            for path_match in d.REPO_PATH_RE.finditer(line):
                following = line[path_match.end() : path_match.end() + 1]
                add(path_match.group(1), block.start_line + offset, following)
    return findings


def check_build_commands(doc: d.Doc, index: d.RepoIndex) -> list[d.Finding]:
    findings: list[d.Finding] = []
    created = d._created_target_names(d._cleaned_created_paths(doc))
    for block in doc.blocks:
        for offset, line in enumerate(block.lines):
            if line.lstrip().startswith("#"):
                continue
            match = d.BUILD_CMD_RE.search(line)
            if not match:
                continue
            line_number = block.start_line + offset
            tokens = d._split_command(line[match.start(1) :])
            target = d._target_from_build_args(tokens)
            if target is None or target in d.BUILTIN_TARGETS or target in created:
                continue
            if target not in index.targets:
                findings.append(
                    d.Finding(
                        path=doc.rel,
                        line=line_number,
                        check="build-target",
                        message=f"build target '{target}' is not defined by any CMakeLists.txt",
                        section=doc.section_at(line_number),
                    )
                )
    return findings


def check_run_commands(doc: d.Doc, index: d.RepoIndex) -> list[d.Finding]:
    findings: list[d.Finding] = []
    created = d._created_target_names(d._cleaned_created_paths(doc))
    doc_dir = doc.path.parent
    try:
        doc_dir_rel = str(doc_dir.relative_to(d.repo_root()))
    except ValueError:
        doc_dir_rel = ""

    for block in doc.blocks:
        if block.lang not in ("", "bash", "sh", "shell", "console"):
            continue
        for offset, raw_line in enumerate(block.lines):
            line_number = block.start_line + offset
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path_match = d.RUN_PATH_RE.search(line)
            if path_match:
                binary = path_match.group("bin")
                args = d._split_command(line[path_match.end() :])
            else:
                match = d.RUN_CMD_RE.search(line)
                if not match or not line[match.start() :].lstrip().startswith("./"):
                    continue
                binary = match.group("bin")
                if (
                    not d._looks_like_binary(binary)
                    or binary in d.IGNORED_BINARIES
                    or binary in created
                ):
                    continue
                args = d._split_command(line[match.end() :])

            target = index.targets.get(binary)
            if target is None:
                if (
                    binary in d.BUILTIN_TARGETS
                    or binary in d.IGNORED_BINARIES
                    or binary in created
                ):
                    continue
                findings.append(
                    d.Finding(
                        path=doc.rel,
                        line=line_number,
                        check="run-binary",
                        message=f"executable '{binary}' is not defined by any CMakeLists.txt",
                        section=doc.section_at(line_number),
                    )
                )
                continue

            if doc_dir_rel.startswith("examples/") and target.cmake_dir != doc_dir_rel:
                findings.append(
                    d.Finding(
                        path=doc.rel,
                        line=line_number,
                        check="run-binary",
                        message=(
                            f"executable '{binary}' belongs to '{target.cmake_dir}', "
                            f"not to the documenting example '{doc_dir_rel}'"
                        ),
                        section=doc.section_at(line_number),
                    )
                )

            if any(d._strip_placeholders(arg) or "=" in arg for arg in args):
                continue
            argc_range = index.argc_range(binary)
            if argc_range is None:
                continue
            argc = len(args) + 1
            low, high = argc_range
            if not low <= argc <= high:
                findings.append(
                    d.Finding(
                        path=doc.rel,
                        line=line_number,
                        check="run-arity",
                        message=(
                            f"'{binary}' receives {len(args)} argument(s), but its parser accepts {low - 1}..{high - 1}"
                        ),
                        section=doc.section_at(line_number),
                    )
                )
    return findings


def check_includes(doc: d.Doc, index: d.RepoIndex) -> list[d.Finding]:
    findings: list[d.Finding] = []
    for block in doc.blocks:
        if block.lang not in ("cpp", "c++", "c", "h", "hpp", "asc", ""):
            continue
        for offset, line in enumerate(block.lines):
            match = d.INCLUDE_RE.match(line)
            if not match:
                continue
            include = match.group(1)
            candidates = [f"include/{include}", f"examples/common/{include}", include]
            if not any(index.exists(candidate) for candidate in candidates):
                findings.append(
                    d.Finding(
                        path=doc.rel,
                        line=block.start_line + offset,
                        check="include-path",
                        message=f'#include "{include}" does not resolve to a header in the repository',
                        section=doc.section_at(block.start_line + offset),
                    )
                )
    return findings


def _is_api_doc(rel: str) -> bool:
    return rel.startswith(("docs/zh/3_API/", "docs/en/3_API/"))


def check_api_symbols(doc: d.Doc, index: d.RepoIndex) -> list[d.Finding]:
    if not _is_api_doc(doc.rel):
        return []
    findings: list[d.Finding] = []
    for block in doc.blocks:
        if block.lang not in ("cpp", "c++", "hpp", "h"):
            continue
        for offset, line in enumerate(block.lines):
            for match in d.DECL_RE.finditer(line):
                symbol = match.group(1)
                if symbol in d.EXTERNAL_SYMBOLS or symbol.endswith("_"):
                    # A trailing underscore marks a template parameter, not a real symbol.
                    continue
                if symbol not in index.header_symbols:
                    findings.append(
                        d.Finding(
                            path=doc.rel,
                            line=block.start_line + offset,
                            check="api-symbol",
                            message=f"symbol '{symbol}' is not declared under include/",
                            section=doc.section_at(block.start_line + offset),
                        )
                    )
    return findings


def check_qualified_symbols(doc: d.Doc, index: d.RepoIndex) -> list[d.Finding]:
    """Check ``Catlass::...::Type`` references outside API reference pages."""
    if _is_api_doc(doc.rel):
        return []
    findings: list[d.Finding] = []
    seen: set[tuple[str, int]] = set()

    def add(symbol: str, line: int) -> None:
        key = (symbol, line)
        if symbol in index.include_tokens or key in seen:
            return
        seen.add(key)
        findings.append(
            d.Finding(
                path=doc.rel,
                line=line,
                check="qualified-symbol",
                message=f"'Catlass::{symbol}' is not defined under include/",
                section=doc.section_at(line),
            )
        )

    for block in doc.blocks:
        for offset, line in enumerate(block.lines):
            for match in d.QUALIFIED_SYMBOL_RE.finditer(line):
                add(match.group(1), block.start_line + offset)

    # Qualified names also appear in tables and prose, not only in code fences.
    in_fence = False
    fence_marker = ""
    for number, line in enumerate(doc.text.splitlines(), start=1):
        match = d.FENCE_RE.match(line)
        if match:
            marker = match.group("fence")[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
                continue
            if marker == fence_marker:
                in_fence = False
                continue
        if in_fence:
            continue
        for candidate in d.INLINE_CODE_RE.findall(line):
            for symbol_match in d.QUALIFIED_SYMBOL_RE.finditer(candidate):
                add(symbol_match.group(1), number)
    return findings


def run_checks(docs: list[str], index: d.RepoIndex) -> list[d.Finding]:
    findings: list[d.Finding] = []
    for rel in docs:
        path = d.repo_root() / rel
        if not path.exists():
            continue
        doc = d.load_doc(rel)
        findings.extend(check_repo_paths(doc, index))
        findings.extend(check_build_commands(doc, index))
        findings.extend(check_run_commands(doc, index))
        findings.extend(check_includes(doc, index))
        findings.extend(check_api_symbols(doc, index))
        findings.extend(check_qualified_symbols(doc, index))
    return sorted(findings, key=lambda item: (item.path, item.line, item.check))
