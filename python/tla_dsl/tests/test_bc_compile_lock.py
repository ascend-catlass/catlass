# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the BC cache lock and the import-time build."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from catlass import bc_compile


@pytest.fixture
def fake_build(tmp_path: Path, monkeypatch):
    """(cfg, compiled) built on stub compilers, so no CANN toolchain is needed."""
    stubs = tmp_path / "stubs"
    for subdir in ("Cube", "Vector"):
        (stubs / subdir).mkdir(parents=True)
        (stubs / subdir / "dma.cpp").write_text("dma")
    ccec = tmp_path / "ccec"
    ccec.write_bytes(b"test compiler")
    cfg = bc_compile.BCConfig(
        paths=bc_compile.BCPaths(stubs, tmp_path, stubs),
        toolchain=bc_compile.BCToolchain(ccec, tmp_path / "llvm-link", tmp_path),
        cce_arch="c310",
        catlass_arch=3510,
        cache_dir=tmp_path / "bc",
    )
    compiled: list[tuple[str, str]] = []

    def compile_stub(cpp, output, core_type, config):
        compiled.append((core_type, cpp.stem))
        output.write_text(cpp.stem + "\n")

    def link_bc(inputs, output, config):
        output.write_text("".join(p.read_text() for p in inputs))

    monkeypatch.setattr(bc_compile, "compile_stub", compile_stub)
    monkeypatch.setattr(bc_compile, "link_bc", link_bc)
    monkeypatch.setattr(bc_compile, "_resolve_paths", lambda: cfg.paths)
    monkeypatch.setattr(bc_compile, "_resolve_toolchain", lambda *a: cfg.toolchain)
    monkeypatch.setattr(bc_compile, "_resolve_cache_dir", lambda: cfg.cache_dir)
    return cfg, compiled


def _cache(cfg: bc_compile.BCConfig) -> Path:
    return cfg.cache_dir / bc_compile.compute_cache_key(cfg.toolchain.ccec)


def test_cache_lock_serializes_builds_until_release(fake_build, monkeypatch, capsys):
    """Two handles conflict as two processes do: flock is per open description."""
    cfg, _ = fake_build
    monkeypatch.setattr(bc_compile, "DEFAULT_LOCK_TIMEOUT_S", 0.05)

    with bc_compile._cache_lock(cfg.cache_dir):
        with pytest.raises(TimeoutError, match=r"bc$"):
            with bc_compile._cache_lock(cfg.cache_dir):
                pass
        # The import-time build reports a held lock instead of building anyway.
        assert bc_compile.autobuild_on_import() is None
        assert "BC prebuild skipped" in capsys.readouterr().err

    with bc_compile._cache_lock(cfg.cache_dir):  # released by the block above
        pass


def test_cache_lock_lives_beside_the_root_and_keeps_its_file(tmp_path: Path):
    """A lock inside the root would be unlinked by ``--clean``, and unlinking the
    lock file itself would let two processes lock two different inodes."""
    root = tmp_path / "bc"

    with bc_compile._cache_lock(root):
        lock = bc_compile._lock_path(root)
        assert lock.parent == root.resolve().parent and lock.exists()
        assert root.resolve() not in lock.parents

    assert lock.exists()
    assert not root.exists(), "locking is not building"


def test_cache_lock_unifies_real_and_symlinked_entries(tmp_path: Path):
    """A real path and a symlink to it must share one lock file.

    Otherwise a process using each spelling takes "the" lock independently, and a
    cold build races into the same tree (``--clean`` could also delete it while
    the other is building).  A child process proves the exclusion, not just the
    path equality.
    """
    real = tmp_path / "shared-bc"
    real.mkdir()
    alias = tmp_path / "alias-bc"
    alias.symlink_to(real, target_is_directory=True)

    assert bc_compile._lock_path(alias) == bc_compile._lock_path(real)

    child = (
        "import sys\n"
        "from pathlib import Path\n"
        "from catlass import bc_compile\n"
        "bc_compile.DEFAULT_LOCK_TIMEOUT_S = 0.5\n"
        "try:\n"
        "    with bc_compile._cache_lock(Path(sys.argv[1])):\n"
        "        print('ACQUIRED')\n"
        "except TimeoutError:\n"
        "    print('BLOCKED')\n"
    )
    # A bogus ASCEND_HOME_PATH keeps the child's import-time autobuild from
    # actually compiling; cwd=tmp_path keeps it off the source tree; PYTHONPATH
    # points at the very package the parent is using.
    package_root = str(Path(bc_compile.__file__).resolve().parents[1])
    child_env = {
        **os.environ,
        "ASCEND_HOME_PATH": str(tmp_path / "no-cann"),
        "CATLASS_DSL_CACHE_DIR": str(tmp_path / "no-cache"),
        "PYTHONPATH": os.pathsep.join(
            filter(None, [package_root, os.environ.get("PYTHONPATH", "")])
        ),
    }
    with bc_compile._cache_lock(alias):
        result = subprocess.run(
            [sys.executable, "-c", child, str(real)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
            env=child_env,
        )
    assert "BLOCKED" in result.stdout, result.stdout + result.stderr


def test_cache_lock_raises_on_a_permanent_failure(tmp_path: Path, monkeypatch):
    """A filesystem without flock must not look like a held lock: retrying it
    would stall for the whole budget and then blame a peer that does not exist."""

    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(bc_compile.fcntl, "flock", unsupported)

    with pytest.raises(OSError) as excinfo:
        with bc_compile._cache_lock(tmp_path / "bc"):
            pass
    assert excinfo.value.errno == errno.ENOLCK


def test_autobuild_builds_announces_and_reuses(fake_build, capsys):
    cfg, compiled = fake_build

    cache = bc_compile.autobuild_on_import()

    assert cache == _cache(cfg)
    assert (cache / "manifest.json").exists()
    assert compiled, "the first import must actually compile the stubs"
    assert "building the BC template cache" in capsys.readouterr().err

    compiled.clear()
    capsys.readouterr()
    assert bc_compile.autobuild_on_import() == cache
    assert compiled == []
    assert capsys.readouterr().err == ""


def test_autobuild_defers_to_the_bc_compile_cli(fake_build, monkeypatch):
    _, compiled = fake_build
    monkeypatch.setattr(
        sys, "orig_argv", ["python", "-m", "catlass.bc_compile", "--force"]
    )

    assert bc_compile.autobuild_on_import() is None
    assert compiled == [], "the CLI must own its own build"


def test_autobuild_is_fail_soft(fake_build, monkeypatch, capsys):
    cfg, compiled = fake_build

    def missing_toolchain(*args):
        raise RuntimeError("ASCEND_HOME_PATH is not set")

    monkeypatch.setattr(bc_compile, "_resolve_toolchain", missing_toolchain)
    # import catlass must keep working off-device; the compile step reports it.
    assert bc_compile.autobuild_on_import() is None
    assert compiled == []
    monkeypatch.setattr(bc_compile, "_resolve_toolchain", lambda *a: cfg.toolchain)

    def failing_compile(cpp, output, core_type, config):
        raise RuntimeError("ccec exploded")

    monkeypatch.setattr(bc_compile, "compile_stub", failing_compile)
    assert bc_compile.autobuild_on_import() is None
    assert "BC prebuild skipped" in capsys.readouterr().err
    assert not (_cache(cfg) / "manifest.json").exists()


def test_build_locked_rechecks_the_cache_inside_the_lock(fake_build):
    """A peer that finished while we waited must not trigger a second build."""
    cfg, compiled = fake_build
    cache = _cache(cfg)
    bc_compile._build_locked(cfg, cache)
    compiled.clear()

    bc_compile._build_locked(cfg, cache)

    assert compiled == []


def test_cache_is_current_rejects_another_version_or_arch(fake_build):
    """Judging by the defaults would reuse a cache built for another version or
    arch -- and ``init`` would then skip the build it was asked for."""
    cfg, compiled = fake_build
    cache = _cache(cfg)
    bc_compile._build_locked(cfg, cache)

    assert bc_compile._cache_is_current(cfg, cache)
    assert not bc_compile._cache_is_current(replace(cfg, cce_arch="c999"), cache)
    assert not bc_compile._cache_is_current(replace(cfg, catlass_arch=999), cache)

    manifest = json.loads((cache / "manifest.json").read_text())
    manifest["catlass_version"] = "0.0.0-other"
    (cache / "manifest.json").write_text(json.dumps(manifest))
    assert not bc_compile._cache_is_current(cfg, cache)

    compiled.clear()
    bc_compile.init(cce_arch="c999")
    assert compiled, "a different arch must be compiled, not skipped"
    assert (cache / "meta_op.aic.c999.bc").exists()


def test_compile_stub_surfaces_the_compiler_error(tmp_path: Path, monkeypatch, capsys):
    """``capture_output`` hides ccec's message unless it is re-printed."""
    stubs = tmp_path / "stubs"
    (stubs / "Cube").mkdir(parents=True)
    cpp = stubs / "Cube" / "dma.cpp"
    cpp.write_text("dma")
    cfg = bc_compile.BCConfig(
        paths=bc_compile.BCPaths(stubs, tmp_path, stubs),
        toolchain=bc_compile.BCToolchain(
            tmp_path / "ccec", tmp_path / "llvm-link", tmp_path
        ),
        cce_arch="c310",
        catlass_arch=3510,
        cache_dir=tmp_path / "bc",
    )

    def failing_run(args, **kwargs):
        raise subprocess.CalledProcessError(
            1, args, output="", stderr="error: unknown type name 'foo'\n"
        )

    monkeypatch.setattr(bc_compile.subprocess, "run", failing_run)

    with pytest.raises(subprocess.CalledProcessError):
        bc_compile.compile_stub(cpp, tmp_path / "dma.aic.c310.bc", "aic", cfg)
    assert "unknown type name" in capsys.readouterr().err


def test_clean_removes_the_root_and_keeps_the_lock_file(fake_build, monkeypatch):
    cfg, _ = fake_build
    bc_compile._build_locked(cfg, _cache(cfg))
    monkeypatch.setattr(sys, "argv", ["bc_compile", "--clean"])

    bc_compile.main()

    assert not cfg.cache_dir.exists()
    assert bc_compile._lock_path(cfg.cache_dir).exists()
