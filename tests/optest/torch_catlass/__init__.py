# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
import os
import re

import torch
import torch_npu

# Version info — auto-derived from catlass git repo
from ._version import get_version as _get_version

_vers = _get_version()
__version__ = _vers["full"]
__catlass_version__ = _vers["full"]
__catlass_tag__ = _vers["tag"]
__catlass_commit__ = _vers["commit"]
# Expose to JIT compiler at runtime
os.environ["CATLASS_JIT_VERSION"] = _vers["full"]

__all__ = [
    "ops",
    "basic_matmul",
    "batched_matmul",
    "grouped_matmul_slice_m",
    "matmul_add",
    "padding_matmul",
    "grouped_matmul_slice_k",
    "optimized_matmul",
    "grouped_matmul_slice_m_per_token_dequant",
    "grouped_matmul_slice_m_per_token_dequant_multistage",
    "grouped_matmul_slice_k_per_token_dequant",
    "splitk_matmul",
    "quant_matmul",
    "basic_matmul_tla",
    "optimized_matmul_tla",
    "basic_matmul_preload_zN",
    "matmul_full_loadA",
    "padding_splitk_matmul",
    "a2_fp8_e4m3_matmul",
    "w8a16_matmul",
    "w4a8_matmul",
    "small_matmul",
    "single_core_splitk_matmul",
    "streamk_matmul",
    "sparse_matmul_tla",
    "quant_optimized_matmul_tla",
    "ascend950_basic_matmul",
    "quant_matmul_full_loadA_tla",
    "strided_batched_matmul_tla",
    "quant_multi_core_splitk_matmul_tla",
    "ascend950_fp8_mx_matmul_aswt",
    "ascend950_fp4_mx_matmul_aswt",
    "ascend950_matmul_fixpipe_opti",
    "ascend950_basic_matmul_gemv",
    "ascend950_quant_matmul_per_group_per_block_tla",
    "ascend950_matmul_full_dequant",
    "mla",
    "ascend950_matmul_evg",
    "EvgPostprocessMode",
    "ascend950_matmul_full_loadA",
    "ascend950_dual_matmul_silu_mul",
    "ascend950_matmul_gather_scatter",
    "ascend950_batched_matmul",
    "ascend950_streamk_matmul",
    "flash_attention_infer",
    "x_attention",
    "conv_bias",
    "flash_attention_infer_tla",
    "fag_tla",
    "ascend950_flash_attention_infer",
    "ascend950_fp8_mx_flash_attention_infer",
    "w4a4_matmul_per_token_per_channel_dequant",
    "broadcast_matmul_perblock_quant",
    "ascend950_fp8_mx_batch_matmul",
    "ascend950_dual_level_quant_mx_batch_matmul",
    "ascend950_multi_core_splitk_matmul",
    "ascend950_tail_multi_core_splitk_matmul",
    "ascend950_flash_attention_chunk_prefill",
    "ascend950_fp4_mx_quant_matmul",
    "ascend950_fp8_e4m3_quant_matmul",
    "ascend950_rain_fusion_attention",
    "clear_jit_cache",
    "symm",
    "ascend950_basic_syrk",
    "ascend950_syrk",
    "__version__",
    "__catlass_version__",
]

_catlass_loaded: bool = False


def enable_mssanitizer():
    """Enable Ascend memory sanitizer for subsequent JIT compilations."""
    os.environ["MS_SANITIZE_MEMORY"] = "1"


def clear_jit_cache():
    """Remove all JIT-compiled kernel cache files on disk."""
    import glob as _glob
    import shutil

    cache_dir = os.environ.get("CATLASS_JIT_CACHE_DIR", "")
    if cache_dir and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    home_cache = os.path.expanduser("~/.cache/catlass/jit_cache")
    if os.path.isdir(home_cache):
        shutil.rmtree(home_cache, ignore_errors=True)

    tmp_cache = "/tmp/catlass_jit"
    if os.path.isdir(tmp_cache):
        shutil.rmtree(tmp_cache, ignore_errors=True)


def get_npu_arch():
    """Return the CATLASS architecture id for the active TorchNPU device."""
    device_count = torch_npu.npu.device_count()
    if device_count <= 0:
        raise RuntimeError("No Ascend NPU device is available for torch-catlass")

    device_name = torch_npu.npu.get_device_name()
    if re.match(r"Ascend910B.+", device_name, re.I) or re.search(
        r"Ascend910_93", device_name, re.I
    ):
        return 2201
    elif re.search("Ascend950(PR|DT)", device_name, re.I):
        return 3510
    else:
        raise RuntimeError(f"Unsupported device name: {device_name}")


def _load_so_files(lib_dir):
    """Load every shared object in ``lib_dir`` with global symbol visibility."""
    for lib_file in sorted(os.listdir(lib_dir)):
        if lib_file.endswith(".so"):
            lib_path = os.path.join(lib_dir, lib_file)
            print(f"Loading library: {lib_path}")
            _dl_mode = getattr(os, "RTLD_NOW", 0x2)
            if lib_file.endswith("_ms.so"):
                # Sanitizer builds export the same kernel symbols; keep them local
                # so dlsym(RTLD_DEFAULT) resolves to the production prebuilt libs.
                _dl_mode |= getattr(os, "RTLD_LOCAL", 0)
            else:
                _dl_mode |= getattr(os, "RTLD_GLOBAL", 0x100)
            ctypes.CDLL(lib_path, mode=_dl_mode)


def _find_pkg_dir():
    """Return the directory that contains the compiled ``lib/`` tree.

    In a normal install this is simply the package directory (``__file__``).
    In an editable install the Python sources live in the source tree while
    the compiled artefacts are installed inside site-packages, so we fall
    back to the site-packages location.
    """
    src_base = os.path.dirname(__file__)
    if os.path.isdir(os.path.join(src_base, "lib")):
        return src_base

    from importlib.metadata import distribution

    try:
        dist = distribution("torch-catlass")
        sp_dir = dist.locate_file("torch_catlass")
        sp_dir_str = str(sp_dir)  # Convert PosixPath to string
        if os.path.isdir(os.path.join(sp_dir_str, "lib")):
            return sp_dir_str
    except Exception:
        pass

    return src_base


def _load_kernel_libs():
    """Load JIT and architecture-specific kernel libraries once per process."""
    global _catlass_loaded
    if not _catlass_loaded:
        print("Loading kernel libraries...")
        base = _find_pkg_dir()

        os.environ["CATLASS_JIT_PKG_DIR"] = base

        # 1. JIT 编译器 & JIT kernel 入口 — 先加载编译器，再加载统一入口库
        jit_dir = os.path.join(base, "lib", "jit")
        if os.path.exists(jit_dir):
            mode = getattr(os, "RTLD_NOW", 0x2) | getattr(os, "RTLD_GLOBAL", 0x100)
            compiler = os.path.join(jit_dir, "libcatlass_kernel_jit_compiler.so")
            if os.path.exists(compiler):
                print(f"Loading JIT compiler: {compiler}")
                ctypes.CDLL(compiler, mode=mode)
            jit_lib = os.path.join(jit_dir, "libcatlass_kernel_jit.so")
            if os.path.exists(jit_lib):
                print(f"Loading JIT library: {jit_lib}")
                ctypes.CDLL(jit_lib, mode=mode)

        # 2. 架构相关内核
        arch = get_npu_arch()
        arch_dir = os.path.join(base, "lib", str(arch))
        if os.path.exists(arch_dir):
            print(f"Loading architecture-specific kernels from: {arch_dir}")
            _load_so_files(arch_dir)

        if not os.path.exists(jit_dir) and not os.path.exists(arch_dir):
            raise RuntimeError(f"No library directory found: checked {jit_dir} and {arch_dir}")

        _catlass_loaded = True


def _load_main_lib():
    """Load the PyTorch extension that registers ``torch.ops.catlass`` ops."""
    base = _find_pkg_dir()
    lib_path = os.path.join(base, "lib", "libcatlass_torch.so")
    print(f"Loading main library: {lib_path}")
    torch.ops.load_library(lib_path)


_load_kernel_libs()
_load_main_lib()

print("Importing ops module")
from . import ops
from .ops import *
