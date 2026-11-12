# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for the documentation/code consistency checker."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".agents/skills/catlass-docs-consistency/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

from docscheck import discover as checker  # noqa: E402
from docscheck import onboard, static  # noqa: E402

checker.set_repo_root(ROOT)
INDEX = checker.RepoIndex()


def make_doc(text: str, rel: str = "docs/zh/probe.md"):
    return checker.parse_markdown_text(rel, text, ROOT / rel)


@pytest.mark.parametrize(
    "target,expected",
    [
        ("00_basic_matmul", (4, 5)),
        ("16_group_gemm", (5, 6)),
        ("55_ascend950_mx_grouped_matmul_slice_m", (6, 8)),
        ("57_ascend950_matmul_full_dequant", (5, 64)),
        ("75_symm", (4, 64)),
    ],
)
def test_argument_ranges_match_example_parsers(target, expected):
    assert checker.RepoIndex().argc_range(target) == expected


def test_cmake_wrapper_targets_are_discovered():
    assert "55_ascend950_mx_grouped_matmul_slice_m" in INDEX.targets
    assert "basic_matmul_aclnn" in INDEX.targets


def test_repository_documentation_is_consistent():
    assert static.run_checks(checker._doc_scope(), INDEX) == []


def test_stale_build_target_is_reported():
    doc = make_doc("# Probe\n\n```bash\nbash scripts/build.sh basic_matmul\n```\n")
    findings = static.check_build_commands(doc, INDEX)
    assert [finding.check for finding in findings] == ["build-target"]
    assert "basic_matmul" in findings[0].message


def test_wrong_argument_count_is_reported():
    doc = make_doc("# Probe\n\n```bash\n./00_basic_matmul 1 2\n```\n")
    findings = static.check_run_commands(doc, INDEX)
    assert [finding.check for finding in findings] == ["run-arity"]


def test_missing_repository_path_is_reported():
    doc = make_doc("# Probe\n\n`examples/basic_matmul/basic_matmul.cpp`\n")
    findings = static.check_repo_paths(doc, INDEX)
    assert [finding.check for finding in findings] == ["repo-path"]


def test_missing_include_is_reported():
    doc = make_doc(
        '# Probe\n\n```cpp\n#include "catlass/gemm/block/block_mmad_missing.hpp"\n```\n'
    )
    findings = static.check_includes(doc, INDEX)
    assert [finding.check for finding in findings] == ["include-path"]


def test_missing_api_symbol_is_reported():
    doc = make_doc(
        "# Probe\n\n```cpp\ntemplate <class ElementC_>\nstruct MissingTileCopy {\n    using ElementC = ElementC_;\n};\n```\n",
        rel="docs/zh/3_API/probe/README.md",
    )
    findings = static.check_api_symbols(doc, INDEX)
    assert [finding.check for finding in findings] == ["api-symbol"]
    assert "MissingTileCopy" in findings[0].message


def test_missing_qualified_symbol_is_reported():
    doc = make_doc(
        "# Probe\n\n```cpp\nCatlass::Gemm::Kernel::MissingKernel kernel;\n```\n"
    )
    findings = static.check_qualified_symbols(doc, INDEX)
    assert [finding.check for finding in findings] == ["qualified-symbol"]
    assert "MissingKernel" in findings[0].message


def test_document_kind_distinguishes_readme_from_guides():
    assert checker.doc_kind("examples/00_basic_matmul/README.md") == "readme"
    assert checker.doc_kind("examples/00_basic_matmul/README_en.md") == "readme"
    assert checker.doc_kind("docs/zh/1_Practice/01_quick_start.md") == "guide"


def test_onboard_extracts_documented_build_and_run():
    doc = make_doc(
        "# Probe\n\n"
        "## 执行示例\n\n"
        "```bash\nbash scripts/build.sh 00_basic_matmul\n```\n\n"
        "```bash\ncd output/bin\n./00_basic_matmul 256 512 1024 0\n```\n\n"
        "```text\nCompare success.\n```\n"
    )
    jobs = checker.extract_build_jobs(doc, INDEX)
    assert [job.target for job in jobs] == ["00_basic_matmul"]
    run = jobs[0].runs[0]
    assert run.args == ["256", "512", "1024", "0"]
    assert run.workdir == "output/bin"
    assert run.expected_output == ["Compare success."]


def test_onboard_skips_placeholder_arguments():
    doc = make_doc(
        "# Probe\n\n```bash\nbash scripts/build.sh 00_basic_matmul\n./00_basic_matmul [m] [n] [k]\n```\n"
    )
    assert checker.extract_build_jobs(doc, INDEX) == []


def test_onboard_filters_generations():
    profiles = onboard.load_profiles()
    a2a3 = onboard.get_profile(profiles, "a2a3")

    declared_a2a3 = checker.BuildJob(
        target="00_basic_matmul",
        line="bash scripts/build.sh 00_basic_matmul",
        doc="examples/00_basic_matmul/README.md",
        line_no=1,
        kind="readme",
        declared_products="Atlas A2系列产品、Atlas A3系列产品",
    )
    declared_950 = checker.BuildJob(
        target="57_ascend950_matmul_full_dequant",
        line="bash scripts/build.sh 57_ascend950_matmul_full_dequant",
        doc="examples/57_ascend950_matmul_full_dequant/README.md",
        line_no=1,
        kind="readme",
        declared_products="Ascend 950PR&950DT系列产品",
    )
    undeclared = checker.BuildJob(
        target="02_grouped_matmul_slice_m",
        line="bash scripts/build.sh 02_grouped_matmul_slice_m",
        doc="examples/02_grouped_matmul_slice_m/README.md",
        line_no=1,
        kind="readme",
    )

    selected, skipped = onboard.select_jobs(
        [declared_a2a3, declared_950, undeclared], a2a3
    )
    assert [job.target for job in selected] == ["00_basic_matmul"]
    assert skipped == {"arch-mismatch": 1, "undeclared": 1}

    selected, _ = onboard.select_jobs(
        [declared_a2a3, declared_950, undeclared], a2a3, include_undeclared=True
    )
    assert [job.target for job in selected] == [
        "00_basic_matmul",
        "02_grouped_matmul_slice_m",
    ]


def test_onboard_injects_generation_arch_once():
    profiles = onboard.load_profiles()
    a2a3 = onboard.get_profile(profiles, "a2a3")
    line = onboard.inject_arch("bash scripts/build.sh 00_basic_matmul", a2a3)
    assert line.endswith("-DCATLASS_ARCH=2201")

    explicit = (
        "bash scripts/build.sh 57_ascend950_matmul_full_dequant -DCATLASS_ARCH=3510"
    )
    assert onboard.inject_arch(explicit, a2a3) == explicit


def test_reserved_profile_is_rejected():
    profiles = onboard.load_profiles()
    with pytest.raises(ValueError, match="not enabled"):
        onboard.get_profile(profiles, "ascend960")


def test_onboard_dry_run_lists_documented_tasks():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "check_docs.py"),
            "--repo-root",
            str(ROOT),
            "--mode",
            "onboard",
            "--profile",
            "a2a3",
            "--dry-run",
            "--examples",
            "00_basic_matmul",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "00_basic_matmul" in result.stdout
    assert "jobs=" in result.stdout
