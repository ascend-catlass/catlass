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
"""On-board documentation check: run the commands a document tells the reader to run."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import discover as d

PROFILES_PATH = Path(__file__).resolve().parents[2] / "profiles.json"
OUTPUT_DIR = "output/bin"
ONBOARD_LOG_DIR = "build/docs-onboard"
PROFILE_MARKER = "build/.docs-onboard-profile"


@dataclass
class TaskResult:
    kind: str
    doc: str
    line: int
    status: str
    target: str = ""
    command: str = ""
    phase: str = ""
    exit_code: int | None = None
    duration: float = 0.0
    message: str = ""
    missing_output: list[str] = field(default_factory=list)

    def finding(self) -> d.Finding | None:
        if self.status != "fail":
            return None
        check = {
            "build": "onboard-build",
            "run": "onboard-run",
            "timeout": "onboard-timeout",
            "output": "onboard-output",
        }.get(self.phase, "onboard-run")
        line = self.line
        section = ""
        try:
            section = d.load_doc(self.doc).section_at(line)
        except OSError:
            section = ""
        return d.Finding(
            path=self.doc,
            line=line,
            check=check,
            message=self.message,
            section=section,
        )


def load_profiles(path: Path = PROFILES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_profile(profiles: dict, name: str) -> dict:
    profile = profiles.get(name)
    if profile is None:
        raise ValueError(
            f"unknown profile '{name}'; available: {', '.join(sorted(profiles))}"
        )
    if not profile.get("enabled", True):
        raise ValueError(
            f"profile '{name}' is reserved but not enabled yet; "
            f"fill its arch in {PROFILES_PATH.name} when the toolchain supports it"
        )
    return profile


def arch_status(job: d.BuildJob, profile: dict) -> str:
    """Return ``match``, ``mismatch`` or ``undeclared`` for one build job."""
    explicit = d.BUILD_ARCH_RE.search(job.line)
    if explicit:
        return "match" if explicit.group(1) == profile.get("arch") else "mismatch"
    declared = job.declared_products
    if not declared:
        return "undeclared"
    lowered = declared.lower()
    aliases = [alias.lower() for alias in profile.get("aliases", [])]
    return "match" if any(alias in lowered for alias in aliases) else "mismatch"


def inject_arch(line: str, profile: dict) -> str:
    if d.BUILD_ARCH_RE.search(line):
        return line
    flags = profile.get("build_flags") or [f"-DCATLASS_ARCH={profile.get('arch')}"]
    return f"{line} {' '.join(flags)}"


def _matches_examples(target: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(target, pattern) for pattern in patterns)


def select_jobs(
    jobs: list[d.BuildJob],
    profile: dict,
    include_undeclared: bool = False,
    examples: list[str] | None = None,
    kind: str = "all",
) -> tuple[list[d.BuildJob], dict[str, int]]:
    """Filter jobs by profile generation, example filter and document kind."""
    selected: list[d.BuildJob] = []
    skipped = {"arch-mismatch": 0, "undeclared": 0}
    for job in jobs:
        if kind != "all" and job.kind != kind:
            continue
        if not _matches_examples(job.target, examples or []):
            continue
        status = arch_status(job, profile)
        if status == "mismatch":
            skipped["arch-mismatch"] += 1
            continue
        if status == "undeclared" and not include_undeclared:
            skipped["undeclared"] += 1
            continue
        selected.append(job)
    return selected, skipped


def _environment(profile: dict) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in (profile.get("env") or {}).items():
        env[str(key)] = str(value)
    return env


def _documented_argv(line: str, build: bool = False) -> list[str] | None:
    """Turn a documented command line into an argv list without a shell."""
    if re.search(r"[|<>`]|\$\(", line):
        return None
    try:
        tokens = shlex.split(line, comments=True)
    except ValueError:
        return None
    if tokens and tokens[0] == "bash":
        tokens = tokens[1:]
    if not tokens:
        return None
    if build:
        script = d.repo_root() / tokens[0].lstrip("./")
        if not script.is_file():
            return None
        tokens[0] = str(script)
    return tokens


def _run_command(
    argv: list[str], cwd: Path, env: dict[str, str], timeout: float
) -> tuple[int, str, bool]:
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, output or "", False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate()
        return -1, output or "", True


def _tail(text: str, limit: int = 1200) -> str:
    text = text.strip()
    return text[-limit:]


def _preflight(profile: dict, build_only: bool, env: dict[str, str]) -> str | None:
    if not env.get("ASCEND_HOME_PATH"):
        return "ASCEND_HOME_PATH is not set"
    if build_only or not profile.get("preflight"):
        return None
    configured = profile["preflight"]
    argv = shlex.split(configured) if isinstance(configured, str) else list(configured)
    returncode, output, timed_out = _run_command(argv, d.repo_root(), env, 120)
    if timed_out:
        return "environment preflight timed out"
    if returncode != 0:
        return f"environment preflight failed (exit {returncode}): {_tail(output)}"
    return None


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def _verify_output(expected: list[str], output: str) -> list[str]:
    return [line for line in expected if line not in output]


def _profile_build_line(job: d.BuildJob, profile: dict, clean: bool) -> str:
    line = inject_arch(job.line, profile)
    if clean and "--clean" not in line:
        line = f"{line} --clean"
    return line


def _marker_path() -> Path:
    return d.repo_root() / PROFILE_MARKER


def _clean_required(profile_name: str) -> bool:
    marker = _marker_path()
    if not marker.exists():
        return False
    return marker.read_text(encoding="utf-8", errors="replace").strip() != profile_name


def _record_profile(profile_name: str) -> None:
    marker = _marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(profile_name + "\n", encoding="utf-8")


def _run_job(
    job: d.BuildJob,
    profile: dict,
    profile_name: str,
    options,
    env: dict[str, str],
    verify_kind: str,
    clean: bool,
) -> tuple[TaskResult, str]:
    root = d.repo_root()
    build_timeout = options.build_timeout or profile.get("build_timeout_sec", 1800)
    run_timeout = options.run_timeout or profile.get("run_timeout_sec", 300)

    build_line = _profile_build_line(job, profile, clean)
    build_argv = _documented_argv(build_line, build=True)
    if build_argv is None:
        return (
            TaskResult(
                kind=job.kind,
                doc=job.doc,
                line=job.line_no,
                status="fail",
                target=job.target,
                command=build_line,
                phase="build",
                message="documented build command cannot be executed directly",
            ),
            f"$ {build_line}\n",
        )
    started = time.monotonic()
    returncode, output, timed_out = _run_command(build_argv, root, env, build_timeout)
    duration = time.monotonic() - started
    log = [f"$ {build_line}", output]
    if timed_out:
        result = TaskResult(
            kind=job.kind,
            doc=job.doc,
            line=job.line_no,
            status="fail",
            target=job.target,
            command=build_line,
            phase="timeout",
            duration=duration,
            message=f"build timed out after {build_timeout:.0f}s",
        )
        return result, "\n".join(log)
    if returncode != 0:
        result = TaskResult(
            kind=job.kind,
            doc=job.doc,
            line=job.line_no,
            status="fail",
            target=job.target,
            command=build_line,
            phase="build",
            exit_code=returncode,
            duration=duration,
            message=f"documented build command failed (exit {returncode}): {_tail(output)}",
        )
        return result, "\n".join(log)
    _record_profile(profile_name)

    if options.build_only:
        return (
            TaskResult(
                kind=job.kind,
                doc=job.doc,
                line=job.line_no,
                status="pass",
                target=job.target,
                command=build_line,
                duration=duration,
                message="build ok",
            ),
            "\n".join(log),
        )

    missing: list[str] = []
    run_duration = 0.0
    run_command = ""
    for run in job.runs:
        run_cwd = root / run.workdir
        if not run_cwd.is_dir():
            missing.append(str(run_cwd))
            continue
        command = run.line
        argv = _documented_argv(command)
        if argv is None:
            return (
                TaskResult(
                    kind=job.kind,
                    doc=run.doc,
                    line=run.line_no,
                    status="fail",
                    target=job.target,
                    command=command,
                    phase="run",
                    message="documented run command cannot be executed directly",
                ),
                "\n".join([*log, f"$ {command}"]),
            )
        started = time.monotonic()
        returncode, output, timed_out = _run_command(argv, run_cwd, env, run_timeout)
        run_duration += time.monotonic() - started
        run_command = command
        log.extend([f"$ cd {run.workdir} && {command}", output])
        if timed_out:
            return (
                TaskResult(
                    kind=job.kind,
                    doc=run.doc,
                    line=run.line_no,
                    status="fail",
                    target=job.target,
                    command=command,
                    phase="timeout",
                    duration=run_duration,
                    message=f"run timed out after {run_timeout:.0f}s",
                ),
                "\n".join(log),
            )
        if returncode != 0:
            return (
                TaskResult(
                    kind=job.kind,
                    doc=run.doc,
                    line=run.line_no,
                    status="fail",
                    target=job.target,
                    command=command,
                    phase="run",
                    exit_code=returncode,
                    duration=run_duration,
                    message=f"documented run command failed (exit {returncode}): {_tail(output)}",
                ),
                "\n".join(log),
            )
        if verify_kind == "all" or (
            verify_kind in ("auto", "guide") and job.kind == "guide"
        ):
            absent = _verify_output(run.expected_output, output)
            if absent:
                return (
                    TaskResult(
                        kind=job.kind,
                        doc=run.doc,
                        line=run.line_no,
                        status="fail",
                        target=job.target,
                        command=command,
                        phase="output",
                        duration=run_duration,
                        message=f"documented output is missing: {absent}",
                        missing_output=absent,
                    ),
                    "\n".join(log),
                )

    if missing:
        return (
            TaskResult(
                kind=job.kind,
                doc=job.doc,
                line=job.line_no,
                status="fail",
                target=job.target,
                command=job.line,
                phase="run",
                duration=run_duration,
                message=f"run directory does not exist: {', '.join(missing)}",
            ),
            "\n".join(log),
        )

    return (
        TaskResult(
            kind=job.kind,
            doc=job.runs[0].doc,
            line=job.runs[0].line_no,
            status="pass",
            target=job.target,
            command=run_command,
            duration=duration + run_duration,
            message=f"{len(job.runs)} run(s) ok",
        ),
        "\n".join(log),
    )


def run_onboard(options, docs: list[str], index: d.RepoIndex) -> int:
    profiles_path = Path(options.profiles) if options.profiles else PROFILES_PATH
    try:
        profiles = load_profiles(profiles_path)
    except (OSError, json.JSONDecodeError) as error:
        print(f"error: cannot load profiles from {profiles_path}: {error}", flush=True)
        return 2
    try:
        profile = get_profile(profiles, options.profile)
    except ValueError as error:
        print(f"error: {error}", flush=True)
        return 2

    jobs = d.collect_build_jobs(docs, index)
    selected, skipped = select_jobs(
        jobs,
        profile,
        include_undeclared=options.include_undeclared,
        examples=[
            item.strip() for item in (options.examples or "").split(",") if item.strip()
        ],
        kind=options.kind,
    )

    if options.dry_run:
        for job in selected:
            print(
                f"{job.doc}:{job.line_no}: [{job.kind}] {job.target} "
                f"({len(job.runs)} run(s), products={job.declared_products or 'undeclared'})"
            )
            for run in job.runs:
                prefix = "" if run.workdir == "." else f"cd {run.workdir} && "
                print(f"    {prefix}{run.line}")
        print(
            f"\nprofile={options.profile} arch={profile.get('arch')} "
            f"jobs={len(selected)} skipped={skipped}"
        )
        return 0

    env = _environment(profile)
    if not options.skip_preflight:
        error = _preflight(profile, options.build_only, env)
        if error:
            print(f"error: {error}", flush=True)
            return 2

    results: list[TaskResult] = []
    started = time.monotonic()
    clean_first = options.clean or (
        not options.no_clean and _clean_required(options.profile)
    )
    for number, job in enumerate(selected, start=1):
        result, log = _run_job(
            job,
            profile,
            options.profile,
            options,
            env,
            options.verify_output,
            clean=clean_first and number == 1,
        )
        results.append(result)
        log_path = (
            d.repo_root() / ONBOARD_LOG_DIR / options.profile / f"{job.target}.log"
        )
        _write_log(log_path, log)
        marker = "PASS" if result.status == "pass" else "FAIL"
        print(
            f"[{number}/{len(selected)}] {job.kind:<6} {job.target}: {marker} "
            f"({result.duration:.1f}s)",
            flush=True,
        )

    findings = [result.finding() for result in results]
    findings = [item for item in findings if item is not None]
    findings.sort(key=lambda item: (item.path, item.line, item.check))

    totals = {
        "jobs": len(results),
        "pass": sum(1 for item in results if item.status == "pass"),
        "fail": len(findings),
        "guide_fail": sum(
            1 for item in results if item.status == "fail" and item.kind == "guide"
        ),
        "readme_fail": sum(
            1 for item in results if item.status == "fail" and item.kind == "readme"
        ),
    }
    duration = time.monotonic() - started

    if options.json:
        payload = {
            "profile": options.profile,
            "arch": profile.get("arch"),
            "duration_sec": round(duration, 1),
            "summary": {**totals, **skipped},
            "results": [
                {
                    "doc": item.doc,
                    "line": item.line,
                    "kind": item.kind,
                    "status": item.status,
                    "phase": item.phase,
                    "target": item.target,
                    "command": item.command,
                    "exit_code": item.exit_code,
                    "duration_sec": round(item.duration, 1),
                    "message": item.message,
                }
                for item in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for finding in findings:
            print(finding.render())
        print(
            f"\nprofile={options.profile} arch={profile.get('arch')} jobs={totals['jobs']} "
            f"pass={totals['pass']} fail={totals['fail']} "
            f"(guide={totals['guide_fail']}, readme={totals['readme_fail']}) "
            f"skipped={skipped} duration={duration:.1f}s"
        )

    failed = totals["guide_fail"] if options.fail_on == "guide" else totals["fail"]
    return 1 if failed else 0
