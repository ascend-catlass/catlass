# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import unittest
from catlass_cppgen.catlass.gemm.dispatch_policy import (
    MmadAtlasA2Pingpong,
    MmadPingpong,
    MmadPingpongTlaV2,
    MmadPreloadAsyncWithCallback,
    MmadMultiBatch,
    MatmulGatherScatterDispatchPolicy,
)
from catlass_cppgen.catlass.arch.arch import Arch


class TestMmadAtlasA2Pingpong(unittest.TestCase):
    def test_mmad_atlas_a2_pingpong_default(self):
        mmad = MmadAtlasA2Pingpong()
        self.assertEqual(mmad.stages, 2)
        self.assertFalse(mmad.enable_unit_flag)

    def test_mmad_atlas_a2_pingpong_with_flag(self):
        mmad = MmadAtlasA2Pingpong(enable_unit_flag=True)
        self.assertEqual(mmad.stages, 2)
        self.assertTrue(mmad.enable_unit_flag)

    def test_mmad_atlas_a2_pingpong_to_cpp(self):
        mmad = MmadAtlasA2Pingpong(enable_unit_flag=True)
        cpp_str = mmad.to_cpp()
        self.assertIn("Gemm::MmadAtlasA2Pingpong", cpp_str)
        self.assertIn("true", cpp_str)


class TestMmadPingpong(unittest.TestCase):
    def test_mmad_pingpong(self):
        mmad = MmadPingpong(
            arch_tag=Arch.Ascend950,
            enable_unit_flag=True,
            use_hf32_mode=True,
            l0c_stages=2,
            enable_l1_resident=True,
        )
        self.assertEqual(mmad.arch_tag, Arch.Ascend950)
        self.assertFalse(mmad.async_)
        self.assertEqual(mmad.stages, 2)
        self.assertTrue(mmad.enable_unit_flag)
        self.assertTrue(mmad.use_hf32_mode)
        self.assertEqual(mmad.l0c_stages, 2)
        self.assertTrue(mmad.enable_l1_resident)

    def test_mmad_pingpong_to_cpp(self):
        mmad = MmadPingpong(arch_tag=Arch.Ascend950, enable_unit_flag=True)
        cpp_str = mmad.to_cpp()
        self.assertIn("Gemm::MmadPingpong", cpp_str)
        self.assertIn("Arch::Ascend950", cpp_str)


class TestMmadPingpongTlaV2(unittest.TestCase):
    def test_mmad_pingpong_tla_v2_to_cpp(self):
        policy = MmadPingpongTlaV2(
            arch_tag=Arch.AtlasA2,
            enable_unit_flag=True,
        )
        self.assertEqual(
            policy.to_cpp(),
            "Gemm::MmadPingpongTlaV2<Arch::AtlasA2, true>",
        )


class TestMmadPreloadAsyncWithCallback(unittest.TestCase):
    def test_mmad_preload_async_with_callback(self):
        mmad = MmadPreloadAsyncWithCallback(
            arch_tag=Arch.Ascend950,
            preload_stages=3,
            l1a_stages=2,
            l1b_stages=2,
            l0a_stages=1,
            l0b_stages=1,
            l0c_stages=1,
            enable_unit_flag=True,
            enable_shuffle_k=True,
            use_hf32_mode=True,
        )
        self.assertEqual(mmad.arch_tag, Arch.Ascend950)
        self.assertTrue(mmad.async_)
        self.assertEqual(mmad.preload_stages, 3)
        self.assertTrue(mmad.use_hf32_mode)


class TestMmadMultiBatch(unittest.TestCase):
    def test_mmad_multi_batch(self):
        mmad = MmadMultiBatch(arch_tag=Arch.Ascend950, use_hf32_mode=True, l0c_stages=3)
        self.assertEqual(mmad.arch_tag, Arch.Ascend950)
        self.assertFalse(mmad.async_)
        self.assertEqual(mmad.stages, 2)
        self.assertTrue(mmad.use_hf32_mode)
        self.assertEqual(mmad.l0c_stages, 3)


class TestMatmulGatherScatterDispatchPolicy(unittest.TestCase):
    def test_to_cpp(self):
        policy = MatmulGatherScatterDispatchPolicy(
            arch_tag=Arch.Ascend950,
            output_ub_stages=2,
            aiv_gather=False,
            max_gather_k=768,
        )
        self.assertEqual(
            policy.to_cpp(),
            "Gemm::MatmulGatherScatterDispatchPolicy<Arch::Ascend950, 2, false, 768>",
        )


if __name__ == "__main__":
    unittest.main()
