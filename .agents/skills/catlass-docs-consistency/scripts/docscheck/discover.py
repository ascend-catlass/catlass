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
"""Shared discovery for the CATLASS documentation checks.

The module owns everything that both check modes need: repository indexing,
Markdown parsing and extraction of the commands documented in code fences.
The repository root is explicit because the scripts can be installed together
with the skill, outside of the CATLASS checkout.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT: Path | None = None

# Directories whose Markdown files are user-facing guidance documents.
DOC_ROOTS = ("docs", "examples", "python")
ROOT_DOCS = (
    "README.md",
    "README_en.md",
    "CONTRIBUTING.md",
    "CONTRIBUTING_en.md",
    "AGENTS.md",
)

# Documents that describe past releases and intentionally mention removed paths.
SKIP_DOC_PATTERNS = (
    re.compile(r"(^|/)CHANGELOG(_en)?\.md$"),
    re.compile(r"(^|/)0x_new_versions/"),
)

PATH_PREFIXES = (
    "docs/",
    "examples/",
    "include/",
    "scripts/",
    "tests/",
    "python/",
    "tools/",
    "cmake/",
    "experimental/",
    "3rdparty/",
)

BUILTIN_TARGETS = {
    "catlass_examples",
    "python_extension",
    "torch_library",
    "mstuner_catlass",
    "test_self_contained_includes",
}

FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
REPO_PATH_RE = re.compile(
    r"(?<![\w./-])((?:%s)[A-Za-z0-9_./+@-]+)"
    % "|".join(re.escape(p) for p in PATH_PREFIXES)
)
BUILD_CMD_RE = re.compile(r"(?:^|[\s;&|(])(?:bash\s+)?(?:\./)?scripts/build\.sh\b(.*)$")
RUN_CMD_RE = re.compile(
    r"(?:^|[\s;&|(])(?:\./)?(?P<bin>[A-Za-z0-9_][A-Za-z0-9_.+-]*)(?=$|\s)"
)
RUN_PATH_RE = re.compile(
    r"(?:^|[\s;&|(])(?:\./)?output/bin/(?P<bin>[A-Za-z0-9_][A-Za-z0-9_.+-]*)"
)
INCLUDE_RE = re.compile(
    r'^\s*#include\s+["<]((?:catlass|tla|catlass_patch)/[^">]+)[">]'
)
DECL_RE = re.compile(r"\b(?:class|struct)\s+([A-Z][A-Za-z0-9_]*)\s*(?:[;:{<]|$)")
CMAKE_TARGET_RE = re.compile(
    r"\b(catlass_example_add_executable|add_executable|add_custom_target)\s*\(\s*([A-Za-z0-9_.+-]+)([^)]*)\)"
)
ARG_COMPARISON_RE = re.compile(r"\bargc\s*(>=|<=|==|>|<)\s*([^|&;]+)")
QUALIFIED_SYMBOL_RE = re.compile(
    r"\b(?:Catlass|catlass)::(?:[A-Za-z_]\w*::)*([A-Z][A-Za-z0-9_]*)"
)
BUILD_ARCH_RE = re.compile(r"-DCATLASS_ARCH=(\w+)")
PRODUCT_RE = re.compile(
    r"(?:支持产品型号|Supported\s+products?)\s*[：:]\s*(.+)", re.IGNORECASE
)
README_RE = re.compile(r"^examples/(?:[^/]+/)*README(_en)?\.md$")
CD_OUTPUT_RE = re.compile(r"(?:^|[;&|]\s*)cd\s+(?:\./)?output/bin/?\s*(?:$|[;&|])")
PLACEHOLDER_CHARS = set("<>[]${}*")

SHELL_LANGS = ("", "bash", "sh", "shell", "console")
OUTPUT_LANGS = ("text", "txt", "console")

# Shell snippets in build guides sometimes execute a directory, not a project
# executable (e.g. ``./build`` after ``cd build``).
IGNORED_BINARIES = {
    "build",
    "install",
    "out",
    "output",
    "dist",
    "bin",
    "lib",
    "test",
    "tests",
    "tools",
    "cmake",
    "ninja",
    "make",
    "setup",
    "cd",
    "cp",
    "mv",
    "rm",
    "mkdir",
    "echo",
    "export",
    "source",
    "git",
    "pip",
    "pip3",
    "python",
    "python3",
    "npu-smi",
    "msprof",
    "sudo",
}

# Symbols provided by Ascend C rather than CATLASS. They are allowed to appear
# in API reference snippets without a definition under include/.
EXTERNAL_SYMBOLS = {
    "CopyToUB",
    "LoadData3Dv2",
}


def set_repo_root(path: Path) -> Path:
    """Configure the CATLASS checkout that every check runs against."""
    global REPO_ROOT
    REPO_ROOT = path.resolve()
    return REPO_ROOT


def repo_root() -> Path:
    if REPO_ROOT is None:
        raise RuntimeError("repository root is not configured")
    return REPO_ROOT


@dataclass
class Finding:
    path: str
    line: int
    check: str
    message: str
    section: str = ""

    def render(self) -> str:
        location = f"{self.path}:{self.line}"
        section = f" [section: {self.section}]" if self.section else ""
        return f"{location}: [{self.check}] {self.message}{section}"


@dataclass
class CodeBlock:
    lang: str
    start_line: int  # 1-based line number of the first content line
    lines: list[str]


@dataclass
class Doc:
    path: Path
    rel: str
    text: str
    blocks: list[CodeBlock] = field(default_factory=list)
    headings: list[tuple[int, str]] = field(default_factory=list)

    def section_at(self, line: int) -> str:
        title = ""
        for heading_line, heading_text in self.headings:
            if heading_line > line:
                break
            title = heading_text
        return title


@dataclass
class Target:
    name: str
    cmake_dir: str
    sources: list[str]


@dataclass
class RunSpec:
    """A run command documented for an example binary."""

    binary: str
    args: list[str]
    line: str
    doc: str
    line_no: int
    workdir: str
    expected_output: list[str] = field(default_factory=list)


@dataclass
class BuildJob:
    """A documented build target together with the run commands that use it."""

    target: str
    line: str
    doc: str
    line_no: int
    kind: str
    declared_products: str | None = None
    runs: list[RunSpec] = field(default_factory=list)


def doc_kind(rel: str) -> str:
    """Return ``readme`` for example READMEs, ``guide`` for every other doc."""
    return "readme" if README_RE.match(rel) else "guide"


class RepoIndex:
    def __init__(self) -> None:
        self.tracked = self._git_ls_files()
        self.tracked_set = set(self.tracked)
        self.targets = self._scan_cmake_targets()
        self.header_symbols = self._scan_header_symbols()
        self.include_tokens = self._scan_include_tokens()
        self._argc_cache: dict[str, tuple[int, int] | None] = {}

    @staticmethod
    def _git_ls_files() -> list[str]:
        root = repo_root()
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        items = result.stdout.decode("utf-8", "replace").split("\0")
        return [item for item in items if item]

    def _scan_cmake_targets(self) -> dict[str, Target]:
        root = repo_root()
        targets: dict[str, Target] = {}
        for rel in self.tracked:
            if not rel.endswith("CMakeLists.txt") or not rel.startswith("examples/"):
                continue
            path = root / rel
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            cmake_dir = str(path.parent.relative_to(root))
            for match in CMAKE_TARGET_RE.finditer(text):
                name = match.group(2)
                if name in targets:
                    continue
                tokens = match.group(3).split()
                sources = []
                for tok in tokens:
                    if tok.endswith((".cpp", ".cc", ".c", ".asc")):
                        sources.append(str((path.parent / tok).relative_to(root)))
                targets[name] = Target(
                    name=name,
                    cmake_dir=cmake_dir,
                    sources=sources,
                )
            # Targets created through a small CMake wrapper function, e.g.
            # function(add_mx_grouped_matmul_variant target_name).
            for wrapper in re.finditer(
                r"function\s*\(\s*(\w+)\s+(\w+)[^)]*\)(.*?)endfunction", text, re.S
            ):
                wrapper_name, parameter, body = (
                    wrapper.group(1),
                    wrapper.group(2),
                    wrapper.group(3),
                )
                inner = re.search(
                    r"catlass_example_add_executable\s*\(\s*\$\{%s\}\s+([^)]*)\)"
                    % re.escape(parameter),
                    body,
                )
                if not inner:
                    continue
                tokens = inner.group(1).split()
                sources = []
                for tok in tokens:
                    if tok.endswith((".cpp", ".cc", ".c", ".asc")):
                        sources.append(str((path.parent / tok).relative_to(root)))
                for call in re.finditer(
                    r"\b%s\s*\(\s*([A-Za-z0-9_.+-]+)" % re.escape(wrapper_name), text
                ):
                    name = call.group(1)
                    if name in targets:
                        continue
                    targets[name] = Target(
                        name=name,
                        cmake_dir=cmake_dir,
                        sources=sources,
                    )
        return targets

    def _scan_header_symbols(self) -> dict[str, set[str]]:
        root = repo_root()
        symbols: dict[str, set[str]] = {}
        for rel in self.tracked:
            if not rel.startswith("include/") or not rel.endswith(
                (".hpp", ".h", ".cuh")
            ):
                continue
            path = root / rel
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in DECL_RE.finditer(text):
                symbols.setdefault(match.group(1), set()).add(rel)
        return symbols

    def _scan_include_tokens(self) -> set[str]:
        root = repo_root()
        tokens: set[str] = set()
        for rel in self.tracked:
            if not rel.startswith("include/") or not rel.endswith(
                (".hpp", ".h", ".cuh", ".asc", ".cpp")
            ):
                continue
            path = root / rel
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            tokens.update(re.findall(r"\b[A-Za-z_]\w*\b", text))
        return tokens

    def exists(self, rel_path: str, doc_dir: Path | None = None) -> bool:
        root = repo_root()
        normalized = os.path.normpath(rel_path)
        if normalized.startswith(".."):
            return False
        if "build" in normalized.split(os.sep):
            # Build outputs are produced by the user and are not tracked.
            return True
        if normalized in self.tracked_set or (root / normalized).exists():
            return True
        if normalized.startswith("include/"):
            # Sample-level documents use paths relative to their own include/ tree.
            if any(root.glob(f"examples/*/{normalized}")) or any(
                root.glob(f"examples/*/*/{normalized}")
            ):
                return True
        if doc_dir is None:
            return False
        # Example-level documents often use paths relative to their own directory.
        current = doc_dir
        while True:
            if (current / normalized).exists():
                return True
            if current == root or root not in current.parents:
                break
            current = current.parent
        return False

    def argc_range(self, target_name: str) -> tuple[int, int] | None:
        target = self.targets.get(target_name)
        if target is None or not target.sources:
            return None
        key = "|".join(target.sources)
        if key not in self._argc_cache:
            ranges = [parse_argc_range(src) for src in target.sources]
            ranges = [item for item in ranges if item is not None]
            if not ranges:
                self._argc_cache[key] = None
            else:
                low = max(item[0] for item in ranges)
                high = min(item[1] for item in ranges)
                self._argc_cache[key] = (low, high) if low <= high else None
        return self._argc_cache[key]


def parse_markdown_text(rel: str, text: str, path: Path) -> Doc:
    doc = Doc(path=path, rel=rel, text=text)
    lines = text.splitlines()

    fence_char = ""
    fence_len = 0
    fence_lang = ""
    fence_start = 0
    fence_lines: list[str] = []

    for number, line in enumerate(lines, start=1):
        match = FENCE_RE.match(line)
        if not fence_char:
            if match:
                fence_char = match.group("fence")[0]
                fence_len = len(match.group("fence"))
                fence_lang = (
                    match.group("info").strip().split()[0].lower()
                    if match.group("info").strip()
                    else ""
                )
                fence_start = number + 1
                fence_lines = []
                continue
            heading = HEADING_RE.match(line)
            if heading:
                doc.headings.append((number, heading.group(2).strip()))
            continue

        if (
            match
            and match.group("fence")[0] == fence_char
            and len(match.group("fence")) >= fence_len
        ):
            doc.blocks.append(
                CodeBlock(lang=fence_lang, start_line=fence_start, lines=fence_lines)
            )
            fence_char = ""
            fence_len = 0
            fence_lang = ""
            fence_lines = []
            continue
        fence_lines.append(line)

    if fence_char:
        doc.blocks.append(
            CodeBlock(lang=fence_lang, start_line=fence_start, lines=fence_lines)
        )
    return doc


def parse_markdown(path: Path) -> Doc:
    root = repo_root()
    return parse_markdown_text(
        str(path.relative_to(root)),
        path.read_text(encoding="utf-8", errors="replace"),
        path,
    )


def load_doc(rel: str) -> Doc:
    return parse_markdown(repo_root() / rel)


def _strip_placeholders(token: str) -> bool:
    return any(char in token for char in PLACEHOLDER_CHARS) or "..." in token


def _clean_path_token(token: str) -> str:
    token = token.strip().strip("`'\",;()<>[]{}")
    token = token.split("#")[0]
    token = token.rstrip(".:,;`'\"")
    while token.startswith("./"):
        token = token[2:]
    return token


def _cleaned_created_paths(doc: Doc) -> set[str]:
    """Paths that the document asks the reader to create from scratch."""
    created: set[str] = set()
    for line in doc.text.splitlines():
        if not re.search(r"创建|新建|新增|\bcreate\b", line, re.IGNORECASE):
            continue
        for match in REPO_PATH_RE.finditer(line):
            path = _clean_path_token(match.group(1))
            if path:
                created.add(path)
    return created


def _created_target_names(created: set[str]) -> set[str]:
    names: set[str] = set()
    for path in created:
        parts = path.split("/")
        if len(parts) >= 2:
            names.add(parts[1])
        names.add(Path(path).stem)
    return names


def _split_command(line: str) -> list[str]:
    line = line.strip()
    if not line or line.startswith("#"):
        return []
    line = re.split(r"\s*(?:#|&&|\|\||[|;])\s*", line, maxsplit=1)[0]
    line = re.sub(r"\s+\d*>&\d+", "", line)
    line = re.sub(r"\s+[<>]+\s*\S+", "", line)
    try:
        return shlex.split(line, comments=True)
    except ValueError:
        return line.split()


def _target_from_build_args(tokens: list[str]) -> str | None:
    for token in tokens:
        if token.startswith("-"):
            continue
        if _strip_placeholders(token):
            continue
        if token.startswith("./"):
            token = token[2:]
        return token
    return None


def _looks_like_binary(token: str) -> bool:
    return bool(
        re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", token)
    ) and not token.endswith((".sh", ".py", ".cmake", ".md"))


def _product_declaration(doc: Doc) -> str | None:
    for line in doc.text.splitlines():
        match = PRODUCT_RE.search(line)
        if match:
            return match.group(1).strip()
    return None


def _expected_output(block: CodeBlock | None) -> list[str]:
    """Return a small, stable-looking output sample that follows a run command."""
    if block is None or block.lang not in OUTPUT_LANGS:
        return []
    lines = [line.strip() for line in block.lines if line.strip()]
    if not lines or len(lines) > 3:
        return []
    for line in lines:
        if line.startswith(("$ ", "> ", "./", "bash ", "#")):
            return []
        if "|" in line or any(char.isdigit() for char in line):
            return []
        if not any(char.isalpha() for char in line):
            return []
        if len(line) > 120:
            return []
    return lines


def _parse_run_line(line: str) -> tuple[str, list[str], str, str] | None:
    """Return ``(binary, args, command, workdir)`` for a document run line."""
    if re.search(r"(^|\s)[<>]", line):
        return None
    path_match = RUN_PATH_RE.search(line)
    if path_match:
        binary = path_match.group("bin")
        command = line[path_match.start() :].strip()
        return binary, _split_command(line[path_match.end() :]), command, "."
    match = RUN_CMD_RE.search(line)
    if not match or not line[match.start() :].lstrip().startswith("./"):
        return None
    binary = match.group("bin")
    if not _looks_like_binary(binary) or binary in IGNORED_BINARIES:
        return None
    command = line[match.start() :].strip()
    workdir = (
        "output/bin" if CD_OUTPUT_RE.search(line) or "output/bin/" not in line else "."
    )
    if command.startswith("./output/bin/"):
        workdir = "."
    return binary, _split_command(line[match.end() :]), command, workdir


def extract_build_jobs(doc: Doc, index: RepoIndex) -> list[BuildJob]:
    """Extract documented build/run pairs from one document."""
    products = _product_declaration(doc)
    kind = doc_kind(doc.rel)
    created = _created_target_names(_cleaned_created_paths(doc))
    jobs: list[BuildJob] = []
    current: BuildJob | None = None

    for block_index, block in enumerate(doc.blocks):
        if block.lang not in SHELL_LANGS:
            continue
        for offset, raw in enumerate(block.lines):
            line_no = block.start_line + offset
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            build_match = BUILD_CMD_RE.search(line)
            if build_match:
                tokens = _split_command(line[build_match.start(1) :])
                target = _target_from_build_args(tokens)
                if (
                    target is None
                    or target in BUILTIN_TARGETS
                    or target in created
                    or _strip_placeholders(target)
                ):
                    current = None
                    continue
                current = BuildJob(
                    target=target,
                    line=line,
                    doc=doc.rel,
                    line_no=line_no,
                    kind=kind,
                    declared_products=products,
                )
                jobs.append(current)
                continue

            parsed = _parse_run_line(line)
            if parsed is None:
                continue
            binary, args, command, workdir = parsed
            if binary in created or any(_strip_placeholders(arg) for arg in args):
                continue
            if binary not in index.targets:
                continue
            if current is None or current.target != binary:
                current = BuildJob(
                    target=binary,
                    line=f"bash scripts/build.sh {binary}",
                    doc=doc.rel,
                    line_no=line_no,
                    kind=kind,
                    declared_products=products,
                )
                jobs.append(current)
            current.runs.append(
                RunSpec(
                    binary=binary,
                    args=args,
                    line=command,
                    doc=doc.rel,
                    line_no=line_no,
                    workdir=workdir,
                    expected_output=_expected_output(
                        doc.blocks[block_index + 1]
                        if block_index + 1 < len(doc.blocks)
                        else None
                    ),
                )
            )

    return [job for job in jobs if job.runs]


def collect_build_jobs(docs: list[str], index: RepoIndex) -> list[BuildJob]:
    """Collect and de-duplicate build jobs, preferring guide documents."""
    ordered = sorted(docs, key=lambda rel: (doc_kind(rel) != "guide", rel))
    merged: dict[str, BuildJob] = {}
    seen_runs: set[tuple[str, tuple[str, ...]]] = set()
    for rel in ordered:
        path = repo_root() / rel
        if not path.exists():
            continue
        doc = load_doc(rel)
        for job in extract_build_jobs(doc, index):
            primary = merged.get(job.target)
            if primary is None:
                primary = BuildJob(
                    target=job.target,
                    line=job.line,
                    doc=job.doc,
                    line_no=job.line_no,
                    kind=job.kind,
                    declared_products=job.declared_products,
                )
                merged[job.target] = primary
            if primary.declared_products is None and job.declared_products:
                # A guide may build an example without repeating the product
                # declaration that its README carries.
                primary.declared_products = job.declared_products
            for run in job.runs:
                key = (run.binary, tuple(run.args))
                if key in seen_runs:
                    continue
                seen_runs.add(key)
                primary.runs.append(run)
    return sorted(
        merged.values(), key=lambda job: (job.kind != "guide", job.doc, job.line_no)
    )


def _doc_scope() -> list[str]:
    files: list[str] = []
    for rel in _tracked_markdown():
        if any(pattern.search(rel) for pattern in SKIP_DOC_PATTERNS):
            continue
        if rel.startswith(DOC_ROOTS) or rel in ROOT_DOCS:
            files.append(rel)
    return sorted(files)


def _tracked_markdown() -> list[str]:
    root = repo_root()
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return sorted(str(path.relative_to(root)) for path in root.rglob("*.md"))
    return [line for line in result.stdout.splitlines() if line]


def _changed_files(base: str | None) -> list[str]:
    root = repo_root()
    commands = []
    if base:
        commands.append(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"]
        )
    commands.append(["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"])
    commands.append(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    changed: set[str] = set()
    for command in commands:
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            changed.update(line for line in result.stdout.splitlines() if line)
    return sorted(changed)


def _symbols_in_file(rel: str) -> set[str]:
    path = repo_root() / rel
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    return {match.group(1) for match in DECL_RE.finditer(text)}


def _doc_reference_index(doc: Doc) -> set[str]:
    """Return all repository paths and symbols a document is coupled to."""
    root = repo_root()
    refs: set[str] = set()
    for block in doc.blocks:
        for line in block.lines:
            for path_match in REPO_PATH_RE.finditer(line):
                refs.add(_clean_path_token(path_match.group(1)))
            include = INCLUDE_RE.match(line)
            if include:
                refs.add(f"include/{include.group(1)}")
            for name in re.findall(r"\b([A-Z][A-Za-z0-9_]+)\b", line):
                refs.add(f"symbol:{name}")
    for token in INLINE_CODE_RE.findall(doc.text):
        for path_match in REPO_PATH_RE.finditer(token):
            refs.add(_clean_path_token(path_match.group(1)))
    # Markdown links are handled by check_md_links.py; reuse them for impact analysis.
    for match in re.finditer(r"\]\(([^)\s]+)\)", doc.text):
        destination = match.group(1).split("#")[0]
        if not destination or destination.startswith(
            ("http://", "https://", "mailto:", "#")
        ):
            continue
        target = (doc.path.parent / destination).resolve()
        try:
            refs.add(str(target.relative_to(root)))
        except ValueError:
            continue
    return refs


def affected_documents(changed: list[str], index: RepoIndex) -> list[str]:
    md_files = set(_doc_scope())
    direct = {rel for rel in changed if rel in md_files}
    code_changes = [rel for rel in changed if rel not in md_files]
    if not code_changes:
        return sorted(direct)

    reference_map: dict[str, set[str]] = {}
    symbol_map: dict[str, set[str]] = {}
    for rel in md_files:
        refs = _doc_reference_index(load_doc(rel))
        reference_map[rel] = refs
        symbol_map[rel] = {ref[7:] for ref in refs if ref.startswith("symbol:")}

    affected = set(direct)
    for rel in code_changes:
        changed_dir = str(Path(rel).parent)
        changed_symbols = _symbols_in_file(rel) if rel.startswith("include/") else set()
        cmake_targets: set[str] = set()
        if rel.endswith("CMakeLists.txt"):
            text = (repo_root() / rel).read_text(encoding="utf-8", errors="replace")
            cmake_targets = {match.group(2) for match in CMAKE_TARGET_RE.finditer(text)}
        for doc_rel, refs in reference_map.items():
            if rel in refs:
                affected.add(doc_rel)
                continue
            if changed_dir.startswith("examples/") and doc_rel.startswith(
                changed_dir + "/"
            ):
                affected.add(doc_rel)
                continue
            if cmake_targets and any(target in refs for target in cmake_targets):
                affected.add(doc_rel)
                continue
            if changed_symbols & symbol_map[doc_rel]:
                affected.add(doc_rel)
                continue
            if rel == "scripts/build.sh" and any(
                ref.endswith("build.sh") for ref in refs
            ):
                affected.add(doc_rel)
                continue
    return sorted(affected)


def _brace_body(text: str, start: int) -> str:
    """Return the body of the first ``{...}`` block at or after ``start``."""
    open_pos = text.find("{", start)
    if open_pos < 0:
        return ""
    depth = 0
    for pos in range(open_pos, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_pos + 1 : pos]
    return text[open_pos + 1 :]


def _paren_condition(text: str, start: int) -> str:
    open_pos = text.find("(", start)
    if open_pos < 0:
        return ""
    depth = 0
    for pos in range(open_pos, len(text)):
        char = text[pos]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_pos + 1 : pos]
    return ""


def _enum_values(body: str) -> dict[str, int]:
    match = re.search(r"enum\s+class\s+ArgsIndex\s*\{([^}]*)\}", body)
    if not match:
        return {}
    values: dict[str, int] = {}
    current = 0
    for raw in match.group(1).split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            name, expression = (part.strip() for part in item.split("=", 1))
            try:
                current = int(expression, 0)
            except ValueError:
                continue
        else:
            name = item
        values[name] = current
        current += 1
    return values


def _resolve_value(
    expression: str, enum: dict[str, int], constants: dict[str, int]
) -> int | None:
    expression = re.sub(r"static_cast<[^>]+>", "", expression)
    expression = expression.strip().strip("() ")
    match = re.fullmatch(r"(?:ArgsIndex::)?(\w+)(?:\s*\+\s*(\d+))?", expression)
    if not match:
        return None
    name, offset = match.group(1), match.group(2)
    base = enum.get(name, constants.get(name))
    if base is None:
        try:
            base = int(name, 0)
        except ValueError:
            return None
    return base + (int(offset) if offset else 0)


def _find_parse_body(text: str) -> str:
    patterns = (
        r"\b(?:int|void|bool|auto|ArgsParams|[A-Za-z_]\w*)\s+ParseArguments\s*\([^)]*\bargc\b[^)]*\)",
        r"\bint\s+Parse\s*\([^)]*\bargc\b[^)]*\)",
        r"\bint\s+main\s*\([^)]*\bargc\b[^)]*\)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            body = _brace_body(text, match.start())
            if re.search(r"\bargc\s*(?:>=|<=|==|>|<)", body):
                return body
    return ""


def _options_struct_text(source_text: str, struct_name: str) -> str:
    if re.search(r"\b(?:struct|class)\s+%s\b" % re.escape(struct_name), source_text):
        return source_text
    common = repo_root() / "examples" / "common" / "options.hpp"
    if not common.exists():
        return ""
    text = common.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\bstruct\s+%s\b" % re.escape(struct_name), text)
    if not match:
        return ""
    return text[
        match.start() : match.start() + _brace_body(text, match.start()).__len__() + 2
    ]


def parse_argc_range(source_rel: str) -> tuple[int, int] | None:
    path = repo_root() / source_rel
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")

    parse_body = _find_parse_body(text)
    if not parse_body:
        alias = re.search(r"using\s+Options\s*=\s*(\w+)\s*;", text)
        struct_text = _options_struct_text(text, alias.group(1)) if alias else text
        if not struct_text:
            return None
        parse_body = _find_parse_body(struct_text)
    if not parse_body:
        return None

    enum = _enum_values(parse_body)
    constants: dict[str, int] = {}
    constant_re = re.compile(
        r"(?:constexpr\s+)?(?:static\s+)?(?:const\s+)?(?:unsigned\s+)?"
        r"(?:int32_t|uint32_t|int|size_t|auto)\s+(\w+)\s*=\s*(\d+)"
    )
    for match in constant_re.finditer(text):
        constants[match.group(1)] = int(match.group(2))

    lows: list[int] = []
    highs: list[int] = []
    for if_match in re.finditer(r"\bif\s*\(", parse_body):
        condition = _paren_condition(parse_body, if_match.start())
        body = _brace_body(parse_body, if_match.start())
        if not re.search(
            r"\breturn\s+-?1\s*;|\breturn\s+EXIT_FAILURE|\bexit\s*\(|\bthrow\b", body
        ):
            continue
        for comparison in ARG_COMPARISON_RE.finditer(condition):
            operator = comparison.group(1)
            value = _resolve_value(comparison.group(2), enum, constants)
            if value is None:
                continue
            if operator == "<":
                lows.append(value)
            elif operator == "<=":
                lows.append(value + 1)
            elif operator == ">":
                highs.append(value)
    if not lows and not highs:
        return None
    low = max(lows) if lows else 1
    high = min(highs) if highs else 64
    return (low, high) if low <= high else None
