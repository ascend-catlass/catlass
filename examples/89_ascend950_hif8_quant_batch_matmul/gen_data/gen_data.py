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
"""Offline data generator for the 89_ascend950_hif8_quant_batch_matmul hifp8 example.

Mirrors the host-side data preparation that used to live in
``quant_batch_matmul.cpp`` (107-158): uniform hif8 bytes in ``[8, 30)``,
Fixpipe-truncated per-tensor / per-channel scales, and the ``(n,)`` float bias.

Usage:
    python gen_data/gen_data.py batch M K N bias x1quantmode x2quantmode [seed]

Outputs (under ``gen_data/data/``):
    tensor_a.bin / tensor_b.bin : uint8 hifp8 bytes, ``(batch, m, k)`` / ``(batch, k, n)``
    dequant_scale.bin            : float32 effective scales, ``(n,)`` for per-channel,
                                   1-element combined scale for per-tensor
    quant_scale.bin              : uint64 Fixpipe encoded per-channel words, ``(n,)``;
                                   empty for per-tensor
    bias.bin                     : float32 ``(n,)``
    golden_out.bin               : float32 ``(batch, m, n)`` reference
"""

import os
import struct
import sys

import numpy as np

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE, "data")

# Fixpipe scale-word format: truncate float low 13 bits, set bit 46 for per-channel words.
DEQ_SCALE_MASK = 0xFFFFE000
DEQ_SCALE_FLAG = 1 << 46


def truncate(v):
    bits = struct.unpack("I", struct.pack("f", v))[0] & DEQ_SCALE_MASK
    return struct.unpack("f", struct.pack("I", bits))[0]


def encode_word(v):
    bits = struct.unpack("I", struct.pack("f", v))[0]
    return (bits & DEQ_SCALE_MASK) | DEQ_SCALE_FLAG


def hif8_decode_table():
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = b >> 7
        t2 = (b >> 5) & 0x3
        if t2 == 3:
            d, dl = 4, 2
        elif t2 == 2:
            d, dl = 3, 2
        elif t2 == 1:
            d, dl = 2, 2
        elif (b >> 4) & 1:
            d, dl = 1, 3
        elif (b >> 3) & 1:
            d, dl = 0, 4
        else:
            m = b & 0x7
            if s and m == 0:
                table[b] = np.nan
            elif m:
                table[b] = np.copysign(1.0, -1.0 if s else 1.0) * (2.0 ** (m - 23))
            else:
                table[b] = 0.0
            continue
        sh = 7 - dl
        em = ((b >> (sh - d)) & ((1 << d) - 1)) if d else 0
        w = 3 if d <= 2 else (2 if d == 3 else 1)
        m = b & ((1 << w) - 1)
        if d == 0:
            e = 0
        else:
            se = (em >> (d - 1)) & 1
            stored = (em & ((1 << (d - 1)) - 1)) if d > 1 else 0
            mag = (1 << (d - 1)) + stored
            e = -mag if se else mag
        sg = -1.0 if s else 1.0
        if d == 4 and mag == 15 and m == 1:
            table[b] = sg * 32768.0
        else:
            table[b] = sg * (1.0 + m / (1 << w)) * (2.0 ** e)
    return table


TABLE = hif8_decode_table()


def run(batch, m, k, n, bias_len, x1_mode, x2_mode, seed):
    rng = np.random.default_rng(seed)

    host_a = rng.integers(8, 30, size=(batch, m, k)).astype(np.uint8)
    host_b = rng.integers(8, 30, size=(batch, k, n)).astype(np.uint8)

    scale_a = float(rng.uniform(1.0, 2.0)) if x1_mode == 1 else 1.0
    if x2_mode == 1:
        scale_b = rng.uniform(1.0, 2.0, size=(n,)).astype(np.float32)
        per_channel = False
    elif x2_mode == 2:
        scale_b = rng.uniform(0.01, 2.0, size=(n,)).astype(np.float32)
        per_channel = True
    else:
        scale_b = np.ones((n,), dtype=np.float32)
        per_channel = False

    if per_channel:
        words = np.array([encode_word(float(v)) for v in scale_b], dtype=np.uint64)
        eff = np.array([truncate(float(v)) for v in scale_b], dtype=np.float32)
        quant = words
        dequant = eff
    else:
        combined = np.float32(truncate(float(scale_a * float(scale_b[0]))))
        quant = np.zeros((0,), dtype=np.uint64)
        dequant = np.array([combined], dtype=np.float32)
        eff = np.full((n,), combined, dtype=np.float32)

    host_bias = np.zeros((n,), dtype=np.float32)
    if bias_len == n:
        host_bias[::2] = 0.5
        host_bias[1::2] = -1.0

    float_a = TABLE[host_a].astype(np.float32)
    float_b = TABLE[host_b].astype(np.float32)
    acc = np.matmul(float_a, float_b)
    golden = (acc + host_bias.reshape(1, 1, n)) * eff.reshape(1, 1, n)

    host_a.tofile(os.path.join(DATA_DIR, "tensor_a.bin"))
    host_b.tofile(os.path.join(DATA_DIR, "tensor_b.bin"))
    dequant.astype(np.float32).tofile(os.path.join(DATA_DIR, "dequant_scale.bin"))
    quant.astype(np.uint64).tofile(os.path.join(DATA_DIR, "quant_scale.bin"))
    host_bias.astype(np.float32).tofile(os.path.join(DATA_DIR, "bias.bin"))
    golden.astype(np.float32).tofile(os.path.join(DATA_DIR, "golden_out.bin"))
    print(f"batch={batch}, m={m}, k={k}, n={n}, per_channel={per_channel}, seed={seed}")


if __name__ == "__main__":
    batch = int(sys.argv[1])
    m = int(sys.argv[2])
    k = int(sys.argv[3])
    n = int(sys.argv[4])
    bias_len = int(sys.argv[5])
    x1_mode = int(sys.argv[6])
    x2_mode = int(sys.argv[7])
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 20260803
    os.makedirs(DATA_DIR, exist_ok=True)
    run(batch, m, k, n, bias_len, x1_mode, x2_mode, seed)
