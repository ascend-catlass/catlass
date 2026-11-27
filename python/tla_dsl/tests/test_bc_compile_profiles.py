from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from catlass import bc_compile


@pytest.fixture
def helper_build(tmp_path: Path, monkeypatch):
    sources = {
        "Cube": (
            "printf",
            "mmad",
            "debug_tunnel",
            "dma",
            "print_tensor",
            "system_counter",
        ),
        "Vector": (
            "arange_reg",
            "vdiv",
            "printf",
            "dma",
            "print_tensor",
            "vf_print",
            "data_sync_barrier_ub",
            "sub_abs",
            "axpy",
            "lowp_cast",
            "muls_cast",
            "system_counter",
            "log",
        ),
        # Shared core-agnostic stubs, compiled once per core type and linked
        # into both meta-op bitcodes (see bc_compile._SHARED_SUBDIRS).
        "Scalar": ("data_sync_barrier",),
    }
    source_root = tmp_path / "stubs"
    for subdir, names in sources.items():
        directory = source_root / subdir
        directory.mkdir(parents=True)
        for name in names:
            (directory / f"{name}.cpp").write_text(name)
    ccec = tmp_path / "ccec"
    ccec.write_bytes(b"test compiler")
    cfg = bc_compile.BCConfig(
        paths=bc_compile.BCPaths(source_root, tmp_path, source_root),
        toolchain=bc_compile.BCToolchain(ccec, tmp_path / "llvm-link", tmp_path),
        cce_arch="c310",
        catlass_arch=3510,
        cache_dir=tmp_path / "bc",
    )
    compiled = []

    def compile_stub(cpp, output, core_type, config):
        compiled.append((core_type, cpp.stem))
        output.write_text(cpp.stem + "\n")

    def link_bc(inputs, output, config):
        output.write_text("".join(path.read_text() for path in inputs))

    monkeypatch.setattr(bc_compile, "compile_stub", compile_stub)
    monkeypatch.setattr(bc_compile, "link_bc", link_bc)
    monkeypatch.setattr(bc_compile, "_resolve_paths", lambda: cfg.paths)
    monkeypatch.setattr(bc_compile, "_resolve_toolchain", lambda *args: cfg.toolchain)
    monkeypatch.setattr(bc_compile, "_resolve_cache_dir", lambda: cfg.cache_dir)
    return cfg, compiled


def test_build_variants_preserves_print_order_and_compiles_components_once(
    helper_build,
):
    cfg, compiled = helper_build
    cache = bc_compile.init()

    # Link order follows the combined sorted glob: Cube/ < Scalar/ but
    # Scalar/ < Vector/, so the shared stub lands last for aic, first for aiv.
    assert (cache / "meta_op.aic.c310.bc").read_text() == (
        "dma\nmmad\nsystem_counter\ndata_sync_barrier\n"
    )
    assert (cache / "meta_op.aiv.c310.bc").read_text() == (
        "data_sync_barrier\narange_reg\naxpy\ndata_sync_barrier_ub\ndma\nlog\nlowp_cast\n"
        "muls_cast\nsub_abs\nsystem_counter\nvdiv\n"
    )
    assert (cache / "meta_op.aic.print.c310.bc").read_text() == (
        "debug_tunnel\ndma\nmmad\nprint_tensor\nprintf\nsystem_counter\ndata_sync_barrier\n"
    )
    assert (cache / "meta_op.aiv.print.c310.bc").read_text() == (
        "data_sync_barrier\narange_reg\naxpy\ndata_sync_barrier_ub\ndma\nlog\nlowp_cast\n"
        "muls_cast\nprint_tensor\nprintf\nsub_abs\nsystem_counter\nvdiv\nvf_print\n"
    )
    assert len(compiled) == len(set(compiled)) == 21
    # The shared scalar stub is compiled once per core type, not per variant;
    # the ub helper is AIV-only (same-name-different-body hazard, see
    # bc/Vector/data_sync_barrier_ub.cpp).
    assert {core for core, stem in compiled if stem == "data_sync_barrier"} == {
        "aic",
        "aiv",
    }
    assert {core for core, stem in compiled if stem == "data_sync_barrier_ub"} == {
        "aiv",
    }
    manifest = json.loads((cache / "manifest.json").read_text())
    assert {item["name"] for item in manifest["bc_files"]} == {
        path.name for path in cache.glob("*.bc")
    }

    # Both aggregates reuse their existing outputs during an ordinary rebuild.
    bc_compile.init()
    assert len(compiled) == 21


@pytest.mark.parametrize(
    "filename", ["system_counter.cpp", "sub_abs.cpp", "axpy.cpp", "muls_cast.cpp"]
)
def test_helper_edit_invalidates_helper_cache(helper_build, filename):
    cfg, compiled = helper_build
    first = bc_compile.init()
    source = cfg.paths.bc_stubs / "Vector" / filename
    source.write_text("changed helper implementation")
    second = bc_compile.init()
    assert first != second
    assert len(compiled) == 42
    assert (second / "meta_op.aiv.c310.bc").is_file()


@pytest.mark.parametrize("variant", ["normal", "print"])
@pytest.mark.parametrize(
    "mode,cores", [("aic", ["aic"]), ("aiv", ["aiv"]), ("mix", ["aic", "aiv"])]
)
def test_variant_lookup_selects_expected_cores_in_order(
    helper_build, variant, mode, cores
):
    cache = bc_compile.init()
    suffix = ".print" if variant == "print" else ""

    actual = bc_compile.get_bc_paths(mode, variant=variant, bc_dir=cache)

    assert actual.split(",") == [
        str(cache / f"meta_op.{core}{suffix}.c310.bc") for core in cores
    ]


def test_default_variant_is_print_and_unknown_variant_is_rejected(helper_build):
    cache = bc_compile.init()

    assert Path(bc_compile.get_bc_paths("aic", bc_dir=cache)).name == (
        "meta_op.aic.print.c310.bc"
    )
    with pytest.raises(ValueError, match="Unknown BC aggregate variant"):
        bc_compile.get_bc_paths("aic", variant="unknown", bc_dir=cache)


@pytest.mark.parametrize(
    "filename", ["meta_op.aic.c310.bc", "meta_op.aiv.print.c310.bc"]
)
def test_check_bc_requires_both_variants(helper_build, filename):
    cache = bc_compile.init()
    bc_compile.check_bc(bc_dir=cache)
    (cache / filename).unlink()

    with pytest.raises(RuntimeError, match=filename.replace(".", r"\.")):
        bc_compile.check_bc(bc_dir=cache)


def test_force_rebuild_replaces_legacy_aggregate_without_removing_kernel_cache(
    helper_build, monkeypatch
):
    cfg, compiled = helper_build
    cache = bc_compile.init()
    legacy_aggregate = cache / "meta_op.aic.c310.bc"
    legacy_aggregate.write_bytes((cache / "meta_op.aic.print.c310.bc").read_bytes())
    (cache / "stale_component.aic.c310.bc").write_text("stale")
    kernel_dir = cfg.cache_dir.parent / "kernel-key"
    kernel_dir.mkdir()
    kernel_binary = kernel_dir / "kernel.o"
    kernel_binary.write_bytes(b"existing compiled kernel")
    compiled.clear()
    monkeypatch.setattr(sys, "argv", ["bc_compile", "--force"])

    bc_compile.main()

    assert (
        legacy_aggregate.read_text() == "dma\nmmad\nsystem_counter\ndata_sync_barrier\n"
    )
    assert (
        (cache / "meta_op.aic.print.c310.bc").read_text().startswith("debug_tunnel\n")
    )
    assert not (cache / "stale_component.aic.c310.bc").exists()
    assert kernel_binary.read_bytes() == b"existing compiled kernel"
    assert len(compiled) == len(set(compiled)) == 21
    bc_compile.check_bc(bc_dir=cache)
