"""BC (LLVM bitcode) compiler for TLA DSL.

Compiles BC stub sources using the user's own ``ccec`` / ``llvm-link``
and caches the result.  Cache key = catlass version ID + ccec identity
(ELF build-id).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BCPaths:
    """Resolved paths for BC stubs and their header dependencies."""

    bc_stubs: Path
    catlass_include: Path
    bc_include: Path


@dataclass(frozen=True)
class BCToolchain:
    """Toolchain binaries for BC compilation."""

    ccec: Path
    llvm_link: Path
    ascend_home: Path


@dataclass(frozen=True)
class BCConfig:
    """Full BC build configuration."""

    paths: BCPaths
    toolchain: BCToolchain
    cce_arch: str
    catlass_arch: int
    cache_dir: Path


# ---------------------------------------------------------------------------
# CCE arch mapping
# ---------------------------------------------------------------------------

_CCE_AICORE_ARCH: dict[tuple[str, str], str] = {
    ("c310", "aic"): "dav-c310-cube",
    ("c310", "aiv"): "dav-c310-vec",
}

_SOURCE_SUBDIR: dict[str, str] = {
    "aic": "Cube",
    "aiv": "Vector",
}

# Core-agnostic stub directories, compiled once per core type and linked into
# both meta-op bitcodes. The scalar unit exists on AIC and AIV alike, so a
# scalar helper belongs in exactly one place rather than in a shared header with
# a one-line stub under each of Cube/ and Vector/.
_SHARED_SUBDIRS: tuple[str, ...] = ("Scalar",)

# Mix kernels are not compiled as a dedicated core type: they reuse the plain
# ``aic`` + ``aiv`` meta_op bitcode, linked together by hivmc at runtime.
_MODE_CORE_TYPES: dict[str, list[str]] = {
    "aic": ["aic"],
    "aiv": ["aiv"],
    "mix": ["aic", "aiv"],
}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_cache_dir() -> Path:
    """Resolve BC cache directory (mirrors execution.py logic)."""
    env = os.getenv("CATLASS_DSL_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve() / "bc"
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return (Path(xdg) / "catlass" / "bc").expanduser().resolve()
    return (Path.home() / ".cache" / "catlass" / "bc").resolve()


def _resolve_paths() -> BCPaths:
    """Resolve stub + header paths from installed package or source tree."""
    this_dir = Path(__file__).resolve().parent  # catlass/

    wheel_bc = this_dir / "csrc" / "mlir" / "bc"
    if wheel_bc.is_dir():
        return BCPaths(
            bc_stubs=wheel_bc,
            catlass_include=this_dir / "include",
            bc_include=wheel_bc,
        )

    dsl_root = this_dir.parent  # python/tla_dsl/
    bc_dir = dsl_root / "csrc" / "mlir" / "bc"
    catlass_inc = dsl_root.parents[1] / "include"  # <repo>/include
    return BCPaths(bc_stubs=bc_dir, catlass_include=catlass_inc, bc_include=bc_dir)


def _resolve_toolchain(ascend_home: Optional[Path] = None) -> BCToolchain:
    """Resolve ccec / llvm-link from ASCEND_HOME_PATH."""
    if ascend_home is None:
        env = os.environ.get("ASCEND_HOME_PATH", "")
        if not env:
            raise RuntimeError(
                "ASCEND_HOME_PATH is not set. "
                "Source set_env.sh from your CANN installation."
            )
        ascend_home = Path(env)

    ccec = ascend_home / "bin" / "ccec"
    llvm_link = ascend_home / "bin" / "llvm-link"
    for tool, name in [(ccec, "ccec"), (llvm_link, "llvm-link")]:
        if not tool.exists():
            raise RuntimeError(
                f"Cannot find {name} at {tool}. "
                "Check your ASCEND_HOME_PATH / CANN installation."
            )
    return BCToolchain(ccec=ccec, llvm_link=llvm_link, ascend_home=ascend_home)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def _catlass_version_id() -> str:
    """Get catlass version identifier for cache key."""
    try:
        from . import __version__

        if __version__ and __version__ != "0.0.0":
            return __version__
    except Exception:
        pass
    return "unknown"


def _ccec_id(ccec: Path) -> str:
    """ccec identity = ELF build-id (a few small reads, ~5 us).

    The linker computes it while building ccec, so any rebuilt ccec carries a
    new one -- no need to hash the 150 MB binary (~119 ms).  Falls back to
    size+mtime for binaries without a build-id (``--build-id=none``).
    """
    try:
        with ccec.open("rb") as f:
            header = f.read(64)
            if header[:4] == b"\x7fELF" and header[4:6] == b"\x02\x01":  # ELF64 LE
                e_phoff = struct.unpack_from("<Q", header, 32)[0]
                e_phentsize, e_phnum = struct.unpack_from("<HH", header, 54)
                f.seek(e_phoff)
                table = f.read(e_phentsize * e_phnum)
                for i in range(e_phnum):
                    base = i * e_phentsize
                    if struct.unpack_from("<I", table, base)[0] != 4:  # PT_NOTE
                        continue
                    f.seek(struct.unpack_from("<Q", table, base + 8)[0])
                    notes = f.read(struct.unpack_from("<Q", table, base + 32)[0])
                    pos = 0
                    while pos + 12 <= len(notes):  # Elf_Nhdr: namesz, descsz, type
                        namesz, descsz, ntype = struct.unpack_from("<III", notes, pos)
                        name = notes[pos + 12 : pos + 12 + namesz].rstrip(b"\0")
                        desc = notes[pos + 12 + ((namesz + 3) & ~3) :][:descsz]
                        if ntype == 3 and name == b"GNU":  # NT_GNU_BUILD_ID
                            return desc.hex()
                        pos += 12 + ((namesz + 3) & ~3) + ((descsz + 3) & ~3)
    except (OSError, struct.error):
        return "missing"
    stat = ccec.stat()
    return f"stat:{stat.st_size}:{stat.st_mtime_ns}"


def compute_cache_key(ccec: Path) -> str:
    """Cache key = catlass version ID + ccec identity."""
    return hashlib.sha256(
        f"{_catlass_version_id()}:{_ccec_id(ccec)}".encode()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Compile: .cpp -> .bc
# ---------------------------------------------------------------------------


def compile_stub(cpp: Path, out_bc: Path, core_type: str, cfg: BCConfig) -> None:
    """Compile a single stub ``.cpp`` to ``.bc`` via ``ccec``."""
    aicore_arch = _CCE_AICORE_ARCH.get((cfg.cce_arch, core_type))

    args = [
        str(cfg.toolchain.ccec),
        "-O2",
        "-x",
        "cce",
        "--cce-auto-sync=off",
        "--cce-aicore-only",
        "--cce-generic-addrspace=off",
        f"-DCATLASS_ARCH={cfg.catlass_arch}",
        "-DTILING_KEY_VAR",
        "-Wno-ignored-attributes",
        f"-I{cfg.toolchain.ascend_home}/include",
        f"-I{cfg.toolchain.ascend_home}/compiler/tikcpp",
        f"-I{cfg.toolchain.ascend_home}/compiler/tikcpp/tikcfw",
        f"-I{cfg.toolchain.ascend_home}/compiler/tikcpp/tikcfw/impl",
        f"-I{cfg.toolchain.ascend_home}/compiler/tikcpp/tikcfw/interface",
        f"-I{cfg.paths.catlass_include}",
        "-std=c++17",
    ]

    if cfg.paths.bc_include.is_dir():
        args.append(f"-I{cfg.paths.bc_include}")

    if aicore_arch is not None:
        args.extend(["--cce-enable-print", f"--cce-aicore-arch={aicore_arch}"])
        if cpp.name == "print_tensor.cpp":
            args.extend(["-DASCENDC_DUMP", "-DONE_CORE_DUMP_SIZE=1048576"])

    args.extend([str(cpp), "-emit-llvm", "-c", "-o", str(out_bc)])
    subprocess.run(args, check=True)


# ---------------------------------------------------------------------------
# Link: multiple .bc -> single .bc
# ---------------------------------------------------------------------------


def link_bc(inputs: list[Path], output: Path, cfg: BCConfig) -> None:
    """Link multiple ``.bc`` files into one via ``llvm-link``."""
    subprocess.run(
        [str(cfg.toolchain.llvm_link), *[str(p) for p in inputs], "-o", str(output)],
        check=True,
    )


# ---------------------------------------------------------------------------
# Build one meta_op
# ---------------------------------------------------------------------------


def build_meta_op(core_type: str, cache: Path, cfg: BCConfig) -> Path:
    """Compile all stubs for *core_type*, link into ``meta_op.*.bc``."""
    output_name = f"meta_op.{core_type}.{cfg.cce_arch}.bc"
    linked_path = cache / output_name
    if linked_path.exists():
        return linked_path.resolve()

    src_dir = cfg.paths.bc_stubs / _SOURCE_SUBDIR[core_type]
    if not src_dir.is_dir():
        raise RuntimeError(f"Stub source directory not found: {src_dir}")
    src_dirs = [src_dir]
    for shared in _SHARED_SUBDIRS:
        shared_dir = cfg.paths.bc_stubs / shared
        if shared_dir.is_dir():
            src_dirs.append(shared_dir)

    individual: list[Path] = []
    seen: dict[str, Path] = {}
    for cpp in sorted(p for d in src_dirs for p in d.glob("*.cpp")):
        # The per-file bitcode is named from the stem alone, so two stubs of the
        # same name in different directories would silently overwrite each
        # other's .bc. Fail loudly instead.
        if cpp.stem in seen:
            raise RuntimeError(
                f"duplicate bc stub name '{cpp.stem}.cpp': {seen[cpp.stem]} and {cpp}"
            )
        seen[cpp.stem] = cpp
        bc_name = cpp.stem + f".{core_type}.{cfg.cce_arch}.bc"
        bc_path = cache / bc_name
        if not bc_path.exists():
            compile_stub(cpp, bc_path, core_type, cfg)
        individual.append(bc_path)

    if individual:
        link_bc(individual, linked_path, cfg)

    return linked_path.resolve()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(cache: Path, cfg: BCConfig, key: str) -> None:
    """Write manifest.json with compilation metadata."""
    bc_files = sorted(cache.glob("*.bc"))
    manifest = {
        "cache_key": key,
        "catlass_version": _catlass_version_id(),
        "cce_arch": cfg.cce_arch,
        "catlass_arch": cfg.catlass_arch,
        "ascend_home": str(cfg.toolchain.ascend_home),
        "ccec": str(cfg.toolchain.ccec),
        "ccec_id": _ccec_id(cfg.toolchain.ccec),
        "llvm_link": str(cfg.toolchain.llvm_link),
        "bc_stubs": str(cfg.paths.bc_stubs),
        "compiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bc_files": [{"name": f.name, "size": f.stat().st_size} for f in bc_files],
    }
    (cache / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_cfg: Optional[BCConfig] = None
_cache: Optional[Path] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compile_all(cfg: BCConfig, cache: Path) -> None:
    """Compile the full artifact set (meta_op + print_tensor for aic/aiv)
    into *cache* and write the manifest. Existing files are skipped, so an
    up-to-date cache is not recompiled."""
    cache.mkdir(parents=True, exist_ok=True)

    for core_type in ("aic", "aiv"):
        build_meta_op(core_type, cache, cfg)

    for core_type in ("aic", "aiv"):
        subdir = _SOURCE_SUBDIR.get(core_type, "Cube")
        cpp = cfg.paths.bc_stubs / subdir / "print_tensor.cpp"
        bc_path = cache / f"print_tensor.{core_type}.{cfg.cce_arch}.bc"
        if not bc_path.exists():
            compile_stub(cpp, bc_path, core_type, cfg)

    _write_manifest(cache, cfg, compute_cache_key(cfg.toolchain.ccec))


def init(
    *,
    ascend_home: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    cce_arch: str = "c310",
    catlass_arch: int = 3510,
) -> Path:
    """Pre-compile the full BC artifact set and return the cache dir.

    Compiles into *cache_dir* (or the default cache root) without validating
    a previous cache. Used by install scripts and by the pytest session,
    which compiles into its own temporary directory.
    """
    global _cfg, _cache

    cfg = BCConfig(
        paths=_resolve_paths(),
        toolchain=_resolve_toolchain(ascend_home),
        cce_arch=cce_arch,
        catlass_arch=catlass_arch,
        cache_dir=cache_dir or _resolve_cache_dir(),
    )

    _cfg = cfg
    _cache = cfg.cache_dir / compute_cache_key(cfg.toolchain.ccec)
    _compile_all(cfg, _cache)
    return _cache


def _ensure_cache(cfg: BCConfig) -> Path:
    """Check manifest and return cache dir. Raises if BC needs recompilation."""
    key = compute_cache_key(cfg.toolchain.ccec)
    cache = cfg.cache_dir / key
    manifest_path = cache / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get(
                "catlass_version"
            ) == _catlass_version_id() and manifest.get("ccec_id") == _ccec_id(
                cfg.toolchain.ccec
            ):
                return cache
        except (json.JSONDecodeError, OSError):
            pass

    raise RuntimeError(
        "BC not compiled or version mismatch. Run: python -m catlass.bc_compile"
    )


def get_bc_paths(
    kernel_mode: str,
    *,
    bc_type: str = "meta_op",
    core_type: Optional[str] = None,
    bc_dir: Optional[Path] = None,
) -> str:
    """Get comma-joined BC paths from cache. Raises if not compiled.

    ``bc_type``: ``"meta_op"`` (default) or ``"print_tensor"``.
    ``core_type``: required when ``bc_type="print_tensor"`` (``"aic"`` or ``"aiv"``).
    ``bc_dir``: already-validated dir holding the ``.bc`` files; defaults to the
    cache dir resolved from the environment.
    """
    cfg = BCConfig(
        paths=_resolve_paths(),
        toolchain=_resolve_toolchain(),
        cce_arch="c310",
        catlass_arch=3510,
        cache_dir=_resolve_cache_dir(),
    )
    cache = bc_dir or _ensure_cache(cfg)

    if bc_type == "print_tensor":
        if core_type is None:
            raise ValueError("core_type is required when bc_type='print_tensor'")
        bc_name = f"print_tensor.{core_type}.{cfg.cce_arch}.bc"
        bc_path = cache / bc_name
        if not bc_path.exists():
            raise RuntimeError(
                f"{bc_name} not found. Run: python -m catlass.bc_compile"
            )
        return str(bc_path.resolve())

    core_types = _MODE_CORE_TYPES[kernel_mode]
    paths = []
    for ct in core_types:
        bc_path = cache / f"meta_op.{ct}.{cfg.cce_arch}.bc"
        if not bc_path.exists():
            raise RuntimeError(
                f"{bc_path.name} not found. Run: python -m catlass.bc_compile"
            )
        paths.append(str(bc_path.resolve()))
    return ",".join(paths)


def check_bc(bc_dir: Optional[Path] = None) -> None:
    """Check all runtime BC artifacts: meta_op aic/aiv/mix + print_tensor aic/aiv."""
    for kernel_mode in _MODE_CORE_TYPES:
        get_bc_paths(kernel_mode, bc_dir=bc_dir)
    for core_type in ("aic", "aiv"):
        get_bc_paths(
            core_type, bc_type="print_tensor", core_type=core_type, bc_dir=bc_dir
        )


# ---------------------------------------------------------------------------
# CLI: python -m catlass.bc_compile
# ---------------------------------------------------------------------------


def main() -> None:
    """Pre-compile all BC stubs. Useful for CI or offline environments."""
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Pre-compile CATLASS BC stubs (meta_op + print_tensor)."
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--cce-arch", type=str, default="c310")
    parser.add_argument("--catlass-arch", type=int, default=3510)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompile even when the cache looks up to date "
        "(e.g. after editing the .cpp stubs)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove all cached BC files and exit"
    )
    args = parser.parse_args()

    cache_root = (
        Path(args.cache_dir).expanduser() if args.cache_dir else _resolve_cache_dir()
    )

    if args.clean:
        if cache_root.exists():
            shutil.rmtree(cache_root)
            print(f"cleaned: {cache_root}")
        else:
            print(f"cache not found: {cache_root}")
        return

    cfg = BCConfig(
        paths=_resolve_paths(),
        toolchain=_resolve_toolchain(),
        cce_arch=args.cce_arch,
        catlass_arch=args.catlass_arch,
        cache_dir=cache_root,
    )
    key = compute_cache_key(cfg.toolchain.ccec)
    cache = cfg.cache_dir / key

    # A cache hit needs a matching manifest (catlass version + ccec identity)
    # and every artifact still on disk -- check_bc enforces both.
    cache_hit = False
    try:
        _ensure_cache(cfg)
        check_bc(bc_dir=cache)
        cache_hit = True
    except RuntimeError:
        pass
    cache_hit = cache_hit and not args.force

    # System info
    print(f"catlass  : {_catlass_version_id()}")
    print(f"ccec     : {cfg.toolchain.ccec}")
    print(f"cache    : {cache}")
    print(f"cached   : {cache_hit}")
    print()

    if cache_hit:
        print("cache hit, nothing to compile.")
        for f in sorted(cache.glob("*.bc")):
            print(f"  {f.name}: {f.stat().st_size // 1024} KB")
        return

    # Compile
    cache.mkdir(parents=True, exist_ok=True)
    # A forced rebuild must not leave stale bitcode behind (e.g. a stub that
    # was removed from the source tree), so drop existing artifacts first.
    if args.force:
        for stale in [*cache.glob("*.bc"), cache / "manifest.json"]:
            if stale.exists():
                stale.unlink()
    total_t0 = time.perf_counter()

    global _cfg, _cache
    _cfg = cfg
    _cache = cache
    _compile_all(cfg, cache)

    total_dt = time.perf_counter() - total_t0

    # Summary
    for f in sorted(cache.glob("*.bc")):
        print(f"  {f.name}: {f.stat().st_size // 1024} KB")
    print(f"\ncompiled {len(list(cache.glob('*.bc')))} files in {total_dt:.1f}s")


if __name__ == "__main__":
    main()
