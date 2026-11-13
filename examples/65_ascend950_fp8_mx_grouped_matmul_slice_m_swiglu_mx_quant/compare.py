#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import argparse
import torch
import numpy as np
from ml_dtypes import float8_e4m3fn, float8_e5m2, float8_e8m0fnu as float8_e8m0

WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def ceilDiv(m, n):
    return (m + n - 1) // n


def swiglu(gmm_out):
    act, gate = gmm_out.chunk(2, dim=-1)
    temp1 = -act
    temp2 = torch.exp(temp1)
    temp3 = temp2 + 1
    temp4 = act / temp3
    temp5 = temp4 * gate
    return temp5


def quant(swiglu_out, mx_quant_dtype):
    MX_BLOCK_SIZE = 32
    MAX_EXP_FOR_BF16 = 0x7F80
    BF16_EXP_BIAS = 0x7F00
    SHR_NUM_FOR_BF16 = 7
    EMAX_SHIFTED = 0x0400
    NAN_CUSTOMIZATION = 0x7F81
    MAX_EXP_FOR_FP8 = 0x00FF
    SPECIAL_EXP_THRESHOLD = 0x0040

    if mx_quant_dtype == 36:
        EMAX_SHIFTED = 0x0400
    elif mx_quant_dtype == 35:
        EMAX_SHIFTED = 0x0780
    else:
        raise ValueError(f"Unsupported mx_quant_dtype: {mx_quant_dtype}")

    data_bf16 = swiglu_out.to(torch.bfloat16)
    M, N = data_bf16.shape

    data_uint16 = data_bf16.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exp_field = data_uint16 & MAX_EXP_FOR_BF16

    n_blocks = ceilDiv(N, MX_BLOCK_SIZE)
    N_padded = n_blocks * MX_BLOCK_SIZE

    if N_padded > N:
        exp_padded = torch.zeros(M, N_padded, dtype=torch.int32)
        exp_padded[:, :N] = exp_field
        data_padded = torch.zeros(M, N_padded, dtype=torch.bfloat16)
        data_padded[:, :N] = data_bf16
    else:
        exp_padded = exp_field
        data_padded = data_bf16

    exp_blocks = exp_padded.reshape(M, n_blocks, MX_BLOCK_SIZE)
    max_exp = exp_blocks.max(dim=-1).values

    cmp_result = max_exp != MAX_EXP_FOR_BF16
    zero_mask = max_exp != 0
    invalid_data_mask = max_exp <= EMAX_SHIFTED

    t_emax_shifted = torch.tensor(EMAX_SHIFTED, dtype=torch.int32)
    t_max_exp_fp8 = torch.tensor(MAX_EXP_FOR_FP8, dtype=torch.int32)
    t_zero = torch.tensor(0, dtype=torch.int32)
    t_nan_cust = torch.tensor(NAN_CUSTOMIZATION, dtype=torch.int32)
    t_special = torch.tensor(SPECIAL_EXP_THRESHOLD, dtype=torch.int32)

    max_exp_clamped = torch.where(invalid_data_mask, t_emax_shifted, max_exp)
    shared_exp = max_exp_clamped - EMAX_SHIFTED
    scale_value = shared_exp >> SHR_NUM_FOR_BF16

    scale_value = torch.where(cmp_result, scale_value, t_max_exp_fp8)
    scale_value = torch.where(zero_mask, scale_value, t_zero)

    special_data_mask = shared_exp == BF16_EXP_BIAS
    half_scale_val = BF16_EXP_BIAS - shared_exp
    half_scale_val = torch.where(cmp_result, half_scale_val, t_nan_cust)
    half_scale_val = torch.where(zero_mask, half_scale_val, t_zero)
    half_scale_val = torch.where(special_data_mask, t_special, half_scale_val)

    half_scale_expanded = (
        half_scale_val.unsqueeze(-1).expand(-1, -1, MX_BLOCK_SIZE).reshape(M, N_padded)
    )
    half_scale_bf16 = (
        half_scale_expanded.to(torch.int16).contiguous().view(torch.bfloat16)
    )

    scaled_data = data_padded * half_scale_bf16
    scaled_data_f32 = scaled_data.to(torch.float32)[:, :N]

    if mx_quant_dtype == 36:
        scaled_data_f32 = torch.clamp(scaled_data_f32, -448.0, 448.0)
        quant_out = scaled_data_f32.to(torch.float8_e4m3fn)
    elif mx_quant_dtype == 35:
        scaled_data_f32 = torch.clamp(scaled_data_f32, -57344.0, 57344.0)
        quant_out = scaled_data_f32.to(torch.float8_e5m2)
    else:
        raise ValueError(f"Unsupported mx_quant_dtype: {mx_quant_dtype}")

    quant_out_np = quant_out.contiguous().view(torch.int8).numpy()
    scale_out_np = scale_value.to(torch.uint8).numpy()

    return quant_out_np, scale_out_np


def golden_compute(group, m, n, k, quant_type_str):
    data_dir = os.path.join(WORKSPACE, "data")
    input_dir = os.path.join(data_dir, "input")
    golden_dir = os.path.join(data_dir, "golden")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    if quant_type_str == "float8_e4m3fn":
        dtype = float8_e4m3fn
        mx_quant_dtype = 36
    elif quant_type_str == "float8_e5m2":
        dtype = float8_e5m2
        mx_quant_dtype = 35
    else:
        raise ValueError(f"不支持的量化类型: {quant_type_str}")

    a = np.fromfile(os.path.join(input_dir, "a_8.bin"), dtype=dtype)
    a_scale = np.fromfile(os.path.join(input_dir, "a_scale.bin"), dtype=float8_e8m0)
    b = np.fromfile(os.path.join(input_dir, "b_8_trans.bin"), dtype=dtype)
    b_scale = np.fromfile(
        os.path.join(input_dir, "b_scale_trans.bin"), dtype=float8_e8m0
    )

    group_list = np.fromfile(os.path.join(input_dir, "group_list.bin"), dtype=np.int64)
    group_list = np.cumsum(group_list)

    k0 = ceilDiv(k, 64)
    x = a.reshape(m, k).astype(np.float32)
    x_s = np.repeat(a_scale.reshape(m, k0 * 2), 32, axis=-1).astype(np.float32)
    w = b.reshape(group, n, k).transpose(0, 2, 1).astype(np.float32)
    w_s = (
        np.repeat(b_scale.reshape(group, n, k0 * 2), 32, axis=-1)
        .transpose(0, 2, 1)
        .astype(np.float32)
    )

    x_pad_len = x_s.shape[-1] - x.shape[-1]
    x = np.pad(
        x,
        [(0, 0)] * (len(x.shape) - 1) + [(0, x_pad_len)],
        mode="constant",
        constant_values=0,
    )
    x = torch.from_numpy(x)
    x_s = torch.from_numpy(x_s)

    x1 = x * x_s
    grouped_quant_output = []
    grouped_quant_scale_output = []
    for i in range(group):
        w_temp = w[i]
        w_s_temp = w_s[i]
        w_pad_len = w_s_temp.shape[-2] - w_temp.shape[-2]
        w_temp = np.pad(
            w_temp, [(0, w_pad_len)] + [(0, 0)], mode="constant", constant_values=0
        )
        w_temp = torch.from_numpy(w_temp)
        w_s_temp = torch.from_numpy(w_s_temp)
        x2 = w_temp * w_s_temp
        if i == 0:
            x1_temp = x1[: group_list[i], :]
        else:
            x1_temp = x1[group_list[i - 1] : group_list[i], :]

        gmm_out = torch.matmul(x1_temp, x2)
        torch.set_printoptions(threshold=torch.inf)
        _has_inf = torch.isinf(gmm_out).any()
        _has_nan = torch.isnan(gmm_out).any()
        swiglu_out = swiglu(gmm_out)
        swiglu_out = swiglu_out.to(torch.bfloat16)
        swiglu_out = swiglu_out.to(torch.float32)
        quant_output, quant_scale_output = quant(swiglu_out, mx_quant_dtype)

        grouped_quant_output.append(quant_output)
        grouped_quant_scale_output.append(quant_scale_output)

    result_np = np.concatenate(grouped_quant_output, axis=0)
    result_np.tofile(os.path.join(golden_dir, "golden_result.bin"))

    result_scale_np = np.concatenate(grouped_quant_scale_output, axis=0)
    result_scale_np.tofile(os.path.join(golden_dir, "golden_result_scale.bin"))


def golden_high_precision_compute(group, m, n, k, quant_type_str):
    data_dir = os.path.join(WORKSPACE, "data")
    input_dir = os.path.join(data_dir, "input")
    golden_dir = os.path.join(data_dir, "golden")
    os.makedirs(golden_dir, exist_ok=True)

    if quant_type_str == "float8_e4m3fn":
        mx_quant_dtype = 36
    elif quant_type_str == "float8_e5m2":
        mx_quant_dtype = 35
    else:
        raise ValueError(f"不支持的量化类型: {quant_type_str}")

    a_fp32 = np.fromfile(os.path.join(input_dir, "a_fp32.bin"), dtype=np.float32)
    a_fp32 = torch.from_numpy(a_fp32).reshape(m, k)

    b_fp32 = np.fromfile(os.path.join(input_dir, "b_fp32.bin"), dtype=np.float32)
    b_fp32 = torch.from_numpy(b_fp32).reshape(group, k, n)

    group_list = np.fromfile(os.path.join(input_dir, "group_list.bin"), dtype=np.int64)
    group_list = np.cumsum(group_list)

    grouped_quant_output = []
    grouped_quant_scale_output = []
    for i in range(group):
        if i == 0:
            a_slice = a_fp32[: group_list[i], :]
        else:
            a_slice = a_fp32[group_list[i - 1] : group_list[i], :]

        gmm_out = torch.matmul(a_slice, b_fp32[i])
        swiglu_out = swiglu(gmm_out)
        quant_output, quant_scale_output = quant(swiglu_out, mx_quant_dtype)

        grouped_quant_output.append(quant_output)
        grouped_quant_scale_output.append(quant_scale_output)

    result_np = np.concatenate(grouped_quant_output, axis=0)
    result_np.tofile(os.path.join(golden_dir, "golden_high_precision_result.bin"))

    result_scale_np = np.concatenate(grouped_quant_scale_output, axis=0)
    result_scale_np.tofile(
        os.path.join(golden_dir, "golden_high_precision_result_scale.bin")
    )


def compare_results(quant_type_str, m, n):
    data_dir = os.path.join(WORKSPACE, "data")
    golden_dir = os.path.join(data_dir, "golden")

    if quant_type_str == "float8_e4m3fn":
        dtype = float8_e4m3fn
        err = 2**-4
    elif quant_type_str == "float8_e5m2":
        dtype = float8_e5m2
        err = 2**-3
    else:
        raise ValueError(f"不支持的量化类型: {quant_type_str}")

    input_dir = os.path.join(data_dir, "input")
    group_list = np.fromfile(os.path.join(input_dir, "group_list.bin"), dtype=np.int64)
    valid_m = int(np.sum(group_list))
    N_half = n // 2
    q_scale_row_stride = ceilDiv(N_half, 32)

    example_result = np.fromfile(os.path.join(golden_dir, "result.bin"), dtype=dtype)
    example_result = (
        example_result.reshape(m, N_half)[:valid_m, :].flatten().astype(np.float32)
    )
    example_scale = np.fromfile(
        os.path.join(golden_dir, "result_scale.bin"), dtype=float8_e8m0
    )
    example_scale = (
        example_scale.reshape(m, q_scale_row_stride)[:valid_m, :]
        .flatten()
        .astype(np.float32)
    )
    golden_result = np.fromfile(
        os.path.join(golden_dir, "golden_result.bin"), dtype=dtype
    ).astype(np.float32)
    golden_scale = np.fromfile(
        os.path.join(golden_dir, "golden_result_scale.bin"), dtype=float8_e8m0
    ).astype(np.float32)
    hp_result = np.fromfile(
        os.path.join(golden_dir, "golden_high_precision_result.bin"), dtype=dtype
    ).astype(np.float32)
    hp_scale = np.fromfile(
        os.path.join(golden_dir, "golden_high_precision_result_scale.bin"),
        dtype=float8_e8m0,
    ).astype(np.float32)

    if example_result.size != hp_result.size or golden_result.size != hp_result.size:
        print(
            f"Compare failed. Size mismatch: example={example_result.size}, golden={golden_result.size}, hp={hp_result.size}"
        )
        sys.exit(1)
    if example_scale.size != hp_scale.size or golden_scale.size != hp_scale.size:
        print(
            f"Compare failed. Scale size mismatch: example={example_scale.size}, golden={golden_scale.size}, hp={hp_scale.size}"
        )
        sys.exit(1)

    error_example = np.abs(hp_result - example_result) / np.maximum(
        np.abs(hp_result), 1.0
    )
    error_golden = np.abs(hp_result - golden_result) / np.maximum(
        np.abs(hp_result), 1.0
    )

    failed_mask_result = error_example > np.maximum(error_golden, err) * 2
    failed_count_result = int(np.sum(failed_mask_result))

    error_example_scale = np.abs(hp_scale - example_scale) / np.maximum(
        np.abs(hp_scale), 1.0
    )
    error_golden_scale = np.abs(hp_scale - golden_scale) / np.maximum(
        np.abs(hp_scale), 1.0
    )

    failed_mask_scale = error_example_scale > np.maximum(error_golden_scale, err) * 2
    failed_count_scale = int(np.sum(failed_mask_scale))

    if failed_count_result == 0 and failed_count_scale == 0:
        print("Compare success.")
    else:
        print(
            f"Compare failed. Error count: Q: {failed_count_result}, QScale: {failed_count_scale}"
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_count", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    args = parser.parse_args()

    quant_type = "float8_e4m3fn"
    golden_compute(args.problem_count, args.m, args.n, args.k, quant_type)
    golden_high_precision_compute(
        args.problem_count, args.m, args.n, args.k, quant_type
    )
    compare_results(quant_type, args.m, args.n)
