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
"""Entry point for the CATLASS documentation consistency checks.

Two modes are available:

* ``static`` (default) checks documented build targets, run arguments, paths,
  includes and API symbols without touching a compiler or a device;
* ``onboard`` executes the build/run command pairs documented in the selected
  Markdown files and reports failures together with the documenting location.

Examples::

    python3 check_docs.py --repo-root /path/to/catlass
    python3 check_docs.py --repo-root /path/to/catlass --changed
    python3 check_docs.py --repo-root /path/to/catlass --mode onboard --profile a2a3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from docscheck import discover as d
from docscheck import onboard, static


def _git_toplevel() -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def resolve_repo_root(explicit: str | None) -> tuple[Path | None, str]:
    candidate = explicit or os.environ.get("CATLASS_REPO_ROOT")
    root = Path(candidate) if candidate else _git_toplevel()
    if root is None:
        return None, "cannot locate the CATLASS checkout; pass --repo-root"
    root = root.resolve()
    if (
        not (root / "CMakeLists.txt").is_file()
        or not (root / "include" / "catlass").is_dir()
    ):
        return None, f"'{root}' does not look like a CATLASS checkout"
    return root, ""


def _select_documents(args, index: d.RepoIndex) -> list[str]:
    if args.files:
        filters = args.files
        selected = (
            d.affected_documents(filters, index)
            if any(not item.endswith(".md") for item in filters)
            else filters
        )
    elif args.changed or args.base:
        selected = d.affected_documents(d._changed_files(args.base), index)
    else:
        selected = d._doc_scope()
    return sorted(
        {
            rel
            for rel in selected
            if rel.endswith(".md") and (d.repo_root() / rel).exists()
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check CATLASS documentation against the code base."
    )
    parser.add_argument(
        "files", nargs="*", help="documents or changed files to inspect"
    )
    parser.add_argument("--repo-root", help="path to the CATLASS checkout")
    parser.add_argument(
        "--mode",
        choices=("static", "onboard"),
        default="static",
        help="static checks (default) or on-board execution",
    )
    parser.add_argument(
        "--changed",
        action="store_true",
        help="only inspect documents affected by working tree changes",
    )
    parser.add_argument(
        "--base", help="only inspect documents affected since a git revision"
    )
    parser.add_argument(
        "--list-docs",
        action="store_true",
        help="print the selected documents without running checks",
    )

    onboard_group = parser.add_argument_group("on-board mode")
    onboard_group.add_argument(
        "--profile",
        default="a2a3",
        help="generation profile from profiles.json (a2a3, ascend950)",
    )
    onboard_group.add_argument("--profiles", help="alternative profiles.json")
    onboard_group.add_argument(
        "--examples",
        help="comma-separated target names or globs to execute",
    )
    onboard_group.add_argument(
        "--kind",
        choices=("all", "guide", "readme"),
        default="all",
        help="document kind to execute (default: all)",
    )
    onboard_group.add_argument(
        "--build-only",
        action="store_true",
        help="only run the documented build commands",
    )
    onboard_group.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned commands without executing them",
    )
    onboard_group.add_argument(
        "--verify-output",
        choices=("auto", "all", "guide", "none"),
        default="auto",
        help="verify documented output samples (default: guide documents only)",
    )
    onboard_group.add_argument(
        "--include-undeclared",
        action="store_true",
        help="also execute examples that do not declare product support",
    )
    onboard_group.add_argument(
        "--fail-on",
        choices=("all", "guide"),
        default="all",
        help="which document kind makes the run fail (default: all)",
    )
    onboard_group.add_argument(
        "--build-timeout", type=float, help="per-build timeout in seconds"
    )
    onboard_group.add_argument(
        "--run-timeout", type=float, help="per-run timeout in seconds"
    )
    onboard_group.add_argument(
        "--clean", action="store_true", help="force a clean build for the first job"
    )
    onboard_group.add_argument(
        "--no-clean",
        action="store_true",
        help="keep the build tree even when the profile changes",
    )
    onboard_group.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the toolchain/device preflight",
    )
    onboard_group.add_argument(
        "--json", action="store_true", help="print machine-readable results"
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    root, error = resolve_repo_root(args.repo_root)
    if root is None:
        print(f"error: {error}", file=sys.stderr)
        return 2
    d.set_repo_root(root)

    index = d.RepoIndex()
    selected = _select_documents(args, index)
    if args.list_docs:
        for rel in selected:
            print(rel)
        return 0

    if args.mode == "onboard":
        return onboard.run_onboard(args, selected, index)

    if not selected:
        print("No documentation affected; nothing to check.")
        return 0

    findings = static.run_checks(selected, index)
    for finding in findings:
        print(finding.render())

    affected = len({finding.path for finding in findings})
    print(
        f"\nchecked {len(selected)} document(s): {len(findings)} problem(s) in {affected} document(s)"
    )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
