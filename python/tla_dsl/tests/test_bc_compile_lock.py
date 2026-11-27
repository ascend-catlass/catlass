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
import hashlib
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
        (stubs / subdir / "system_counter.cpp").write_text("counter")
    (stubs / "Vector" / "basic_arithmetic.cpp").write_text("arithmetic")
    (stubs / "Vector" / "log.cpp").write_text("log")
    for op in ("arange_reg", "sub_abs", "axpy", "mul_inplace_add", "muls_cast"):
        (stubs / "Vector" / f"{op}.cpp").write_text(op)
    (stubs / "Vector" / "lowp_cast.cpp").write_text("cast")
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
        output.write_text(cpp.read_text() + "\n")

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


def test_ar_stubs_invalidate_the_previous_bitcode_cache(fake_build, monkeypatch):
    cfg, compiled = fake_build
    monkeypatch.setattr(bc_compile, "_catlass_version_id", lambda: "2.2.2")

    def previous_key(ccec):
        return hashlib.sha256(
            f"2.2.2:{bc_compile._ccec_id(ccec)}:data-sync-barrier-v1".encode()
        ).hexdigest()[:16]

    with monkeypatch.context() as old_checkout:
        old_checkout.setattr(bc_compile, "compute_cache_key", previous_key)
        previous_cache = bc_compile.autobuild_on_import()
        assert previous_cache == _cache(cfg)
        assert bc_compile._cache_is_current(cfg, previous_cache)

    stub = cfg.paths.bc_stubs / "Vector" / "ar_spr.cpp"
    symbols = ("_mlir_ciface_tla_get_ar", "_mlir_ciface_tla_clear_ar")
    stub.write_text("\n".join(symbols))
    compiled.clear()

    with pytest.raises(RuntimeError, match="BC not compiled or version mismatch"):
        bc_compile._ensure_cache(cfg)

    current_cache = bc_compile.autobuild_on_import()
    assert current_cache == _cache(cfg)
    assert current_cache != previous_cache
    assert bc_compile._ensure_cache(cfg) == current_cache
    assert ("aiv", "ar_spr") in compiled
    assert ("aic", "ar_spr") not in compiled
    aggregate = bc_compile._meta_op_filename("aiv", cfg.cce_arch, "normal")
    for symbol in symbols:
        assert symbol in (current_cache / aggregate).read_text()
        assert symbol not in (previous_cache / aggregate).read_text()


@pytest.mark.parametrize("stub", ["lowp_cast", "arange_reg"])
def test_updated_stubs_rebuild_the_bitcode_cache(fake_build, stub):
    cfg, compiled = fake_build
    first = bc_compile.autobuild_on_import()
    compiled.clear()
    (cfg.paths.bc_stubs / "Vector" / f"{stub}.cpp").write_text(f"updated {stub}")

    second = bc_compile.autobuild_on_import()

    assert second == _cache(cfg)
    assert second != first
    assert ("aiv", stub) in compiled


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


def test_icache_aggregate_rebuilds_preexisting_counter_cache(fake_build):
    cfg, compiled = fake_build
    source_id = bc_compile._bc_source_id(bc_compile._CACHE_SOURCE_FILES)
    revision = f"ar-spr-v1:b32-carry-v1:{source_id}"
    previous_key = hashlib.sha256(
        f"{bc_compile._catlass_version_id()}:{bc_compile._ccec_id(cfg.toolchain.ccec)}:{revision}".encode()
    ).hexdigest()[:16]
    previous = cfg.cache_dir / previous_key
    previous.mkdir(parents=True)
    for core in ("aic", "aiv"):
        for variant in ("normal", "print"):
            (previous / bc_compile._meta_op_filename(core, "c310", variant)).touch()
    scalar_stubs = cfg.paths.bc_stubs / "Scalar"
    scalar_stubs.mkdir()
    for name in ("icache", "icache_status"):
        (scalar_stubs / f"{name}.cpp").write_text(name)

    bc_compile.init()
    selected = bc_compile.get_bc_paths(
        "mix", variant="print", icache_cores=frozenset({"aiv"})
    )
    aic, aiv = map(Path, selected.split(","))
    assert (aic.parent, aiv.parent) == (_cache(cfg), _cache(cfg))
    assert aic.parent != previous
    assert ("aic", "icache_status") in compiled
    assert ("aiv", "icache") in compiled
    assert set(aic.read_text().splitlines()) == {"dma", "counter", "icache_status"}
    assert set(aiv.read_text().splitlines()) == {
        "arange_reg",
        "sub_abs",
        "arithmetic",
        "axpy",
        "cast",
        "counter",
        "dma",
        "icache",
        "icache_status",
        "log",
        "mul_inplace_add",
        "muls_cast",
    }
