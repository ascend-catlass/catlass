# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import inspect
from abc import ABC
from typing import List, Tuple, Union
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.common.utils import _get_cpp_value, _snake_to_camel, _get_cpp_type

MATMUL_GATHER_SCATTER_MAX_GATHER_K = 768


class MmadBase(ABC):
    """Base class for MMAD policies."""

    def __init__(self, arch_tag: Arch, async_: bool):
        self.arch_tag = arch_tag
        self.async_ = async_

    def to_cpp(self, const_mode: bool = False) -> Union[str, Tuple[List[str], str]]:
        """生成 C++ 代码字符串.

        根据类名和属性自动生成 C++ 模板代码：
        - 类名转换为 Gemm::ClassName 格式
        - 模板参数取自 __init__ 参数（排除self），按定义顺序排列
        - arch_tag 若存在则取其 value 作为模板参数
        - 布尔值自动转换为 C++ 风格的 true/false

        :param const_mode: 当为 True 时，返回常量声明列表和变量名形式的模板字符串；当为 False 时，返回硬编码的模板字符串
        :return: 当 const_mode=False 时返回 C++ 模板代码字符串；当 const_mode=True 时返回 (常量声明列表, 模板字符串) 元组
        """
        cls_name = f"Gemm::{self.__class__.__name__}"
        template_params: List[str] = []
        if const_mode:
            const_declarations: List[str] = []
            try:
                init_params = list(inspect.signature(self.__init__).parameters.keys())
                for param_name in init_params:
                    if param_name in {"self", "async_"} or not hasattr(
                        self, param_name
                    ):
                        continue
                    if param_name == "arch_tag":
                        template_params.append("ArchTag")
                        continue
                    param_value = getattr(self, param_name)
                    var_name = _snake_to_camel(param_name)
                    const_declarations.append(
                        f"constexpr {_get_cpp_type(param_value)} {var_name} = {_get_cpp_value(param_value)};"
                    )
                    template_params.append(var_name)
            except (ValueError, TypeError, AttributeError):
                pass
            params_str = ", ".join(template_params)
            return (
                const_declarations,
                f"{cls_name}<{params_str}>" if params_str else f"{cls_name}<>",
            )
        else:
            try:
                init_params = list(inspect.signature(self.__init__).parameters.keys())
                for param_name in init_params:
                    if param_name in {"self", "async_"} or not hasattr(
                        self, param_name
                    ):
                        continue
                    if param_name == "arch_tag":
                        param_value = getattr(self, param_name).value
                    else:
                        param_value = getattr(self, param_name)
                    template_params.append(_get_cpp_value(param_value))
            except (ValueError, TypeError, AttributeError):
                pass
            params_str = ", ".join(template_params)
            return f"{cls_name}<{params_str}>" if params_str else f"{cls_name}<>"


# Block Mmad Policies


class MmadAtlasA2(MmadBase):
    """MMAD policy for AtlasA2 architecture, synchronous."""

    def __init__(self):
        super().__init__(Arch.AtlasA2, False)


class MmadAtlasA2Async(MmadBase):
    """MMAD policy for AtlasA2 architecture, asynchronous."""

    def __init__(self):
        super().__init__(Arch.AtlasA2, True)


class MmadAtlasA2Pingpong(MmadAtlasA2):
    """MMAD policy with pingpong staging."""

    def __init__(self, enable_unit_flag: bool = False):
        super().__init__()
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag


class MmadAtlasA2PingpongSliceKWithPrologue(MmadAtlasA2):
    """MMAD policy with pingpong staging and sliced K dimension."""

    def __init__(self, enable_unit_flag: bool = False):
        super().__init__()
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag


class MmadAtlasA2PingPongWithPrologue(MmadAtlasA2):
    """MMAD policy with pingpong staging and prologue."""

    def __init__(self, enable_unit_flag: bool = False):
        super().__init__()
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag


class MmadAtlasA2Preload(MmadAtlasA2):
    """MMAD policy with preload capability."""

    def __init__(self, enable_unit_flag: bool = False, enable_shuffle_k: bool = False):
        super().__init__()
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag
        self.enable_shuffle_k = enable_shuffle_k


class MmadAtlasA2PreloadAsync(MmadAtlasA2Async):
    """MMAD policy with async preload capability."""

    def __init__(
        self,
        preload_stages: int,
        l1_stages: int,
        l0a_stages: int,
        l0b_stages: int,
        l0c_stages: int,
        enable_unit_flag: bool = False,
        enable_shuffle_k: bool = False,
    ):
        super().__init__()
        self.preload_stages = preload_stages
        self.l1_stages = l1_stages
        self.l0a_stages = l0a_stages
        self.l0b_stages = l0b_stages
        self.l0c_stages = l0c_stages
        self.enable_unit_flag = enable_unit_flag
        self.enable_shuffle_k = enable_shuffle_k


class MmadAtlasA2PreloadAsyncWithCallback(MmadAtlasA2PreloadAsync):
    """MMAD policy with async preload and callback capability."""

    def __init__(
        self,
        preload_stages: int,
        l1_stages: int,
        l0a_stages: int,
        l0b_stages: int,
        l0c_stages: int,
        enable_unit_flag: bool = False,
        enable_shuffle_k: bool = False,
    ):
        super().__init__(
            preload_stages,
            l1_stages,
            l0a_stages,
            l0b_stages,
            l0c_stages,
            enable_unit_flag,
            enable_shuffle_k,
        )


class GemmAtlasA2(MmadAtlasA2):
    """GEMM policy for AtlasA2 architecture."""

    def __init__(
        self,
        enable_unit_flag: bool = False,
        enable_shuffle_k: bool = False,
        enable_abba: bool = False,
    ):
        super().__init__()
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag
        self.enable_shuffle_k = enable_shuffle_k
        self.enable_abba = enable_abba


class GemvAtlasA2(MmadAtlasA2):
    """GEMV policy for AtlasA2 architecture."""

    def __init__(self):
        super().__init__()
        self.stages = 2


class MmadAtlasA2PingpongBias(MmadAtlasA2):
    """MMAD policy with pingpong staging and bias support."""

    def __init__(self, enable_unit_flag: bool = False):
        super().__init__()
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag


class MmadAtlasA2FullLoadA(MmadAtlasA2):
    """MMAD policy with full load of matrix A."""

    def __init__(self, enable_unit_flag: bool = False):
        super().__init__()
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag


class MmadAtlasA2W8A16(MmadAtlasA2):
    """MMAD policy with W8A16 configuration."""

    def __init__(self, enable_unit_flag: bool = False, enable_shuffle_k: bool = False):
        super().__init__()
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag
        self.enable_shuffle_k = enable_shuffle_k


class MmadAtlasA2DynamicCommon(MmadAtlasA2):
    """MMAD policy with dynamic common configuration."""

    def __init__(self, enable_unit_flag: bool = False, enable_shuffle_k: bool = False):
        super().__init__()
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag
        self.enable_shuffle_k = enable_shuffle_k


class MatmulGatherScatterDispatchPolicy(MmadBase):
    """Gather-mode and output-buffer policy for the MatmulGatherScatter MIX kernel."""

    def __init__(
        self,
        arch_tag: Arch,
        output_ub_stages: int = 1,
        aiv_gather: bool = True,
        max_gather_k: int = MATMUL_GATHER_SCATTER_MAX_GATHER_K,
    ):
        super().__init__(arch_tag, False)
        self.output_ub_stages = output_ub_stages
        self.aiv_gather = aiv_gather
        self.max_gather_k = max_gather_k


class MmadAtlasA2Small(MmadAtlasA2):
    """MMAD policy for small problem sizes."""

    def __init__(
        self,
        stages: int,
        enable_unit_flag: bool = False,
        enable_shuffle_k: bool = False,
    ):
        super().__init__()
        self.stages = stages
        self.enable_unit_flag = enable_unit_flag
        self.enable_shuffle_k = enable_shuffle_k


# Generic MMAD policies that work with different architectures


class MmadPingpong(MmadBase):
    """Generic MMAD policy with pingpong staging."""

    def __init__(
        self,
        arch_tag: Arch,
        enable_unit_flag: bool = False,
        use_hf32_mode: bool = False,
        l0c_stages: int = 1,
        enable_l1_resident: bool = False,
        l1a_stages: int = 2,
        l1b_stages: int = 2,
        l0a_stages: int = 2,
        l0b_stages: int = 2,
    ):
        super().__init__(arch_tag, False)
        self.stages = 2  # May be removed
        self.enable_unit_flag = enable_unit_flag
        self.use_hf32_mode = use_hf32_mode
        self.l0c_stages = l0c_stages
        self.enable_l1_resident = enable_l1_resident
        self.l1a_stages = l1a_stages
        self.l1b_stages = l1b_stages
        self.l0a_stages = l0a_stages
        self.l0b_stages = l0b_stages


class MmadPingpongTlaV2(MmadBase):
    """TLA pingpong policy whose tile views carry the runtime origin shape."""

    def __init__(self, arch_tag: Arch, enable_unit_flag: bool = False):
        super().__init__(arch_tag, False)
        self.stages = 2
        self.enable_unit_flag = enable_unit_flag


class MmadPreloadAsyncWithCallback(MmadBase):
    """Generic MMAD policy with async preload and callback."""

    def __init__(
        self,
        arch_tag: Arch,
        preload_stages: int,
        l1a_stages: int,
        l1b_stages: int,
        l0a_stages: int,
        l0b_stages: int,
        l0c_stages: int,
        enable_unit_flag: bool,
        enable_shuffle_k: bool,
        use_hf32_mode: bool = False,
        enable_l1_resident: bool = False,
    ):
        super().__init__(arch_tag, True)
        self.preload_stages = preload_stages
        self.l1a_stages = l1a_stages
        self.l1b_stages = l1b_stages
        self.l0a_stages = l0a_stages
        self.l0b_stages = l0b_stages
        self.l0c_stages = l0c_stages
        self.enable_unit_flag = enable_unit_flag
        self.enable_shuffle_k = enable_shuffle_k
        self.use_hf32_mode = use_hf32_mode
        self.enable_l1_resident = enable_l1_resident


class MmadMultiBatch(MmadBase):
    """Generic MMAD policy for multi-batch operations."""

    def __init__(
        self, arch_tag: Arch, use_hf32_mode: bool = False, l0c_stages: int = 2
    ):
        super().__init__(arch_tag, False)
        self.stages = 2
        self.use_hf32_mode = use_hf32_mode
        self.l0c_stages = l0c_stages
