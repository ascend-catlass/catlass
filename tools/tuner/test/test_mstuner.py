
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import subprocess
import shutil
import unittest


class MsTunerCatlassTest(unittest.TestCase):

    CATLASS_OUTPUT_BINARY_PATH = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "..", "..", "output", "bin"
    )
    CATLASS_OUTPUT_LIB_PATH = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "..", "..", "output", "lib64"
    )
    MSTUNER_TEST_TEMP_PATH = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "mstuner_test_temp"
    )

    @classmethod
    def setUpClass(cls):
        # create temp dir for mstuner_test
        if not os.path.exists(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH):
            os.mkdir(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH)

        # add ./output/lib64 to LD_LIBRARY_PATH
        if 'LD_LIBRARY_PATH' in os.environ:
            os.environ['LD_LIBRARY_PATH'] = MsTunerCatlassTest.CATLASS_OUTPUT_LIB_PATH + \
                ':' + os.environ['LD_LIBRARY_PATH']
        else:
            os.environ['LD_LIBRARY_PATH'] = MsTunerCatlassTest.CATLASS_OUTPUT_LIB_PATH

    @classmethod
    def tearDownClass(cls):
        # remove temp dir for mstuner_test
        if os.path.exists(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH):
            shutil.rmtree(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH, ignore_errors=True)


    def compile_lib_catlass_kernels(self, kernel_desc: str):
        macro_str = '-DCATLASS_LIBRARY_KERNELS=' + kernel_desc
        compile_cmd = [
            'bash', 'scripts/build.sh', '--clean', macro_str, 'mstuner_catlass'
        ]
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
        return result


    def run_one_case(self, case):
        case_name = case[0]
        case_args = case[1:]
        csv_file_name = case_name + '.csv'
        csv_file_path = os.path.join(MsTunerCatlassTest.MSTUNER_TEST_TEMP_PATH, csv_file_name)

        result = self.compile_lib_catlass_kernels(self.KERNEL_INSTANCES[case_name])
        self.assertEqual(
            result.returncode, 0,
            f'build libcatlass_kernels.so for {case_name} failed: {result.stderr}'
        )

        mstuner_path = os.path.join(MsTunerCatlassTest.CATLASS_OUTPUT_BINARY_PATH, 'mstuner_catlass')
        mstuner_cmd = [
            mstuner_path,
            f'--output={csv_file_path}'
        ]
        mstuner_cmd.extend(case_args)
        result = subprocess.run(mstuner_cmd, capture_output=True, text=True, timeout=600)
        self.assertEqual(
            result.returncode, 0,
            f'running mstuner_catlass for {case_name} with {case_args} failed: {result.stderr}'
        )

        # The single-instance filter is matched by substring. A stale or
        # mismatched CATLASS_LIBRARY_KERNELS description would still build and
        # run "successfully" while tuning 0 operations, yielding a header-only
        # CSV. Assert the intended kernel was actually tuned to avoid a silent
        # pass that loses regression coverage.
        self.assertIn(
            'Initializing 1 operations', result.stdout,
            f'expected exactly 1 operation to be tuned for {case_name}, got: {result.stdout}'
        )

        self.assertEqual(
            os.path.exists(csv_file_path), True,
            f'csv file not generated for {case_name} with {case_args} failed: {result.stdout}'
        )

        with open(csv_file_path, encoding='utf-8') as csv_file:
            csv_lines = csv_file.read().splitlines()
        self.assertGreaterEqual(
            len(csv_lines) - 1, 1,
            f'no tuning result rows generated for {case_name}: {csv_file_path} is header-only'
        )
        self.assertIn(
            self.KERNEL_INSTANCES[case_name], '\n'.join(csv_lines[1:]),
            f'expected kernel instance for {case_name} not found in tuning result: {csv_file_path}'
        )



    # Build only the largest search-space instance of each operator (the last
    # valid tile-shape combination, i.e. max L1/L0 tile shape) instead of the
    # whole search space, so each case compiles a single kernel. See
    # tools/tuner/README.md for the single-instance CATLASS_LIBRARY_KERNELS usage.
    KERNEL_INSTANCES = {
        '00_basic_matmul':
            'catlass_gemm_00_basic_matmul_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_128x256x256_128x256x64_swizzle3x0',
        '02_grouped_matmul_slice_m':
            'catlass_gemm_02_grouped_matmul_slice_m_fp16xColumnMajor_fp16xRowMajor_fp16xRowMajor_256x128x256_256x128x32_swizzle3x1',
        '06_optimized_matmul_padding_ab':
            'catlass_gemm_06_optimized_matmul_padding_ab_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_256x128x256_256x128x64_swizzle3x0',
        '06_optimized_matmul_padding_a_only':
            'catlass_gemm_06_optimized_matmul_padding_a_only_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_256x128x256_256x128x64_swizzle3x0',
        '06_optimized_matmul_padding_b_only':
            'catlass_gemm_06_optimized_matmul_padding_b_only_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_256x128x256_256x128x64_swizzle3x0',
        '06_optimized_matmul_without_padding':
            'catlass_gemm_06_optimized_matmul_without_padding_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_256x128x256_256x128x64_swizzle3x0',
        '08_grouped_matmul':
            'catlass_gemm_08_grouped_matmul_fp16xColumnMajor_fp16xRowMajor_fp16xRowMajor_256x128x256_256x128x32_swizzle3x1',
        '12_quant_matmul':
            'catlass_gemm_12_quant_matmul_int8xColumnMajor_int8xColumnMajor_fp16xRowMajor_256x128x512_256x128x64_swizzle3x1',
        '27_matmul_gelu':
            'catlass_gemm_27_matmul_gelu_fp16xRowMajor_fp16xRowMajor_fp16xRowMajor_256x128x256_256x128x64_swizzle3x0',
    }

    # add custom test cases below
    mstuner_cases = [
        ['00_basic_matmul', '--m=256', '--n=512', '--k=1024'],
        ['02_grouped_matmul_slice_m', '--m=512', '--n=1024', '--k=2048', '--group_count=128'],
        ['06_optimized_matmul_padding_ab', '--m=555', '--n=322', '--k=1111'],
        ['06_optimized_matmul_padding_a_only', '--m=655', '--n=256', '--k=1111'],
        ['06_optimized_matmul_padding_b_only', '--m=555', '--n=322', '--k=1024'],
        ['06_optimized_matmul_without_padding', '--m=512', '--n=256', '--k=1024'],
        ['08_grouped_matmul', '--m=512', '--n=1024', '--k=2048', '--group_count=128'],
        ['12_quant_matmul', '--m=256', '--n=512', '--k=1024'],
        ['27_matmul_gelu', '--m=256', '--n=512', '--k=1024'], # Add matmul gelu mstuner (m, n, k)
    ]


    def test_all_cases(self):
        for case in self.mstuner_cases:
            self.run_one_case(case)


if __name__ == '__main__':
    unittest.main()