/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mla_tiling.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "catlass/detail/alignment.hpp"
using namespace std;
namespace MLATiling {
using AddrOffsets = struct AddressOffsetInfo {
    uint64_t addrQSeqOffset = 0;
    uint64_t addrQSeqRopeOffset = 0;
    uint64_t addrMaskBatchOffset = 0;
    uint64_t addrOFdSeqOffset = 0;
    uint64_t addrLSeqOffset = 0;
};

struct AddrOffsetPerBatch {
    uint64_t qSeqOffset = 0;
    uint64_t qSeqRopeOffset = 0;
    uint64_t maskBatchOffset = 0;
};

inline uint32_t GetHigh32Bit(uint64_t v)
{
    return static_cast<uint32_t>(v >> NUM32);
}
inline uint32_t GetLow32Bit(uint64_t v)
{
    return static_cast<uint32_t>(v);
}

void GetAddrOffsetMLA(uint32_t* tilingHost, const AddrOffsets& addrOffsets, const int32_t tilingOffset)
{
    // Calculate address offset
    tilingHost[tilingOffset + NUM4] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
    tilingHost[tilingOffset + NUM5] = GetLow32Bit(addrOffsets.addrQSeqOffset);
    tilingHost[tilingOffset + NUM6] = GetHigh32Bit(addrOffsets.addrQSeqRopeOffset);
    tilingHost[tilingOffset + NUM7] = GetLow32Bit(addrOffsets.addrQSeqRopeOffset);
    tilingHost[tilingOffset + NUM8] = GetHigh32Bit(addrOffsets.addrMaskBatchOffset);
    tilingHost[tilingOffset + NUM9] = GetLow32Bit(addrOffsets.addrMaskBatchOffset);
}

void GetMLATilingCommon(
    const MLAInfo& mlaInfo, uint32_t& blockDim, uint32_t* tilingHost, const std::vector<uint32_t>& sortedIndices,
    const std::vector<AddrOffsetPerBatch>& addrOffsetsPerBatch)
{
    int32_t maxKVSeqlen = 0;
    int32_t maxQSeqlen = 0;
    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        uint32_t sortSeqIdx = sortedIndices[seqIdx];
        int32_t qSeqLen = *(mlaInfo.qSeqLen + sortSeqIdx);
        int32_t kvSeqlen = *(mlaInfo.kvSeqLen + sortSeqIdx);

        qSeqLen = (kvSeqlen == 0) ? 0 : qSeqLen;
        maxQSeqlen = std::max(maxQSeqlen, qSeqLen);
        maxKVSeqlen = std::max(maxKVSeqlen, kvSeqlen);

        int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE * seqIdx;
        tilingHost[tilingOffset] = static_cast<uint32_t>(qSeqLen);
        tilingHost[tilingOffset + NUM1] = static_cast<uint32_t>(kvSeqlen);
        tilingHost[tilingOffset + NUM2] = static_cast<uint32_t>(sortSeqIdx);
        tilingHost[tilingOffset + NUM3] = static_cast<uint32_t>(mlaInfo.blockSize);

        const auto& batchOffsets = addrOffsetsPerBatch[sortSeqIdx];
        AddrOffsets addrOffsets{};
        addrOffsets.addrQSeqOffset = batchOffsets.qSeqOffset;
        addrOffsets.addrQSeqRopeOffset = batchOffsets.qSeqRopeOffset;
        addrOffsets.addrMaskBatchOffset = batchOffsets.maskBatchOffset;
        GetAddrOffsetMLA(tilingHost, addrOffsets, tilingOffset);
    }

    tilingHost[TILING_MAX_KVSEQLEN] = maxKVSeqlen;
    tilingHost[TILING_MAX_QSEQLEN] = maxQSeqlen;
}

void GetMLATilingSpec(
    const MLAInfo& mlaInfo, uint32_t& blockDim, uint32_t* tilingHost, const std::vector<uint32_t>& sortedIndices,
    const std::vector<std::vector<uint32_t>>& realPreTaskNums)
{
    // TP1 scenario specialization.
    // Treat every Q token with 128 heads as one process, regardless of the mtp depth
    int32_t prevTaskNum = 0;
    int32_t maxKVSeqlen = 0;

    for (int32_t i = 0; i < mlaInfo.batch; i++) {
        uint32_t sortSeqIdx = sortedIndices[i];
        int32_t qSeqLen = *(mlaInfo.qSeqLen + sortSeqIdx);
        int32_t kvSeqlen = *(mlaInfo.kvSeqLen + sortSeqIdx);
        const std::vector<uint32_t>& currentRealPreTaskNum = realPreTaskNums[sortSeqIdx];

        maxKVSeqlen = (maxKVSeqlen > kvSeqlen) ? maxKVSeqlen : kvSeqlen;

        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq++) {
            int32_t tilingOffset = TILING_HEAD_SIZE + PARA_TILING_ELENUM_SPEC * prevTaskNum;
            tilingHost[tilingOffset] = sortSeqIdx;
            tilingHost[tilingOffset + NUM1] = currentRealPreTaskNum[qSeq];
            tilingHost[tilingOffset + NUM2] = kvSeqlen;
            prevTaskNum++;
        }
    }
    tilingHost[TILING_MAX_KVSEQLEN] = maxKVSeqlen;
}

int32_t GetQNBlockTile(const MLAInfo& mlaInfo, int32_t qSeqLen, uint32_t specStrategyFlag)
{
    int32_t tokenNum = qSeqLen;
    if (specStrategyFlag) {
        tokenNum = NUM1;
    }
    int32_t tileListIdx = static_cast<int32_t>(std::ceil(std::log2(tokenNum)));
    tileListIdx = (tileListIdx > NUM5) ? NUM5 : tileListIdx;
    int32_t qNBlockTile = QN_TILE_LIST[tileListIdx];
    int32_t group = mlaInfo.numHeads / mlaInfo.kvHeads;
    qNBlockTile = (qNBlockTile > group) ? group : qNBlockTile;
    return qNBlockTile;
}

void GetTilingHead(
    const MLAInfo& mlaInfo, uint32_t* tilingHost, const uint32_t* torPtr, int32_t maxQseqlen, uint32_t specStrategyFlag)
{
    // Calculating tiling parameters
    tilingHost[TILING_BATCH] = static_cast<uint32_t>(mlaInfo.batch);
    tilingHost[TILING_HEADSIZE] = static_cast<uint32_t>(TILING_HEAD_SIZE);
    if (specStrategyFlag) {
        tilingHost[TILING_PARASIZE] = static_cast<uint32_t>(PARA_TILING_ELENUM_SPEC);
    } else {
        tilingHost[TILING_PARASIZE] = static_cast<uint32_t>(TILING_PARA_SIZE);
    }
    tilingHost[TILING_NUMHEADS] = static_cast<uint32_t>(mlaInfo.numHeads);
    tilingHost[TILING_HEADDIM] = static_cast<uint32_t>(mlaInfo.embeddingSize);
    tilingHost[TILING_NUMBLOKS] = static_cast<uint32_t>(mlaInfo.numBlocks);
    tilingHost[TILING_BLOCKSIZE] = static_cast<uint32_t>(mlaInfo.blockSize);
    int32_t maxNumBlocksPerQuery = (mlaInfo.maxKvSeqlen + mlaInfo.blockSize - 1) / mlaInfo.blockSize;
    tilingHost[TILING_MAXBLOCKS] = static_cast<uint32_t>(maxNumBlocksPerQuery);
    tilingHost[TILING_TOR] = *torPtr;
    tilingHost[TILING_KVHEADS] = mlaInfo.kvHeads;
    int32_t curQNBlockTile = GetQNBlockTile(mlaInfo, maxQseqlen, specStrategyFlag);
    int32_t curQNBlockNum = (mlaInfo.numHeads + curQNBlockTile - 1) / curQNBlockTile;
    tilingHost[TILING_HEAD_SPLIT_SIZE] = static_cast<uint32_t>(curQNBlockTile);
    tilingHost[TILING_HEAD_SPLIT_NUM] = static_cast<uint32_t>(curQNBlockNum);
    tilingHost[TILING_MASKTYPE] = static_cast<uint32_t>(mlaInfo.maskType);
    tilingHost[TILING_HEADDIM_ROPE] = static_cast<uint32_t>(mlaInfo.embeddingSizeRope);
    tilingHost[TILING_TOTAL_QTOKENS] = static_cast<uint32_t>(mlaInfo.numTokens);
}

uint32_t GetKVSplitParam(
    const MLAInfo& mlaInfo, uint32_t& blockDim, uint32_t* tilingHost, const std::vector<uint32_t>& sortedIndices)
{
    // Only split KV when the longest sequence is long enough to be worth distributing across
    // cores (>= blockDim * KV_SEQLEN_SLICE * 2). For short sequences, splitting adds extra
    // flash-decoding accumulation that degrades the max/mean relative error (MARE/MERE) on
    // near-zero output elements without any performance benefit. This matches the reference
    // implementation's split decision, using the runtime AIC core count in blockDim.
    bool isKVSplit = (tilingHost[TILING_MAX_KVSEQLEN] >= blockDim * KV_SEQLEN_SLICE * NUM2) &&
                     (tilingHost[TILING_BATCH] <= blockDim * SPLITKV_RATION && tilingHost[TILING_MAX_QSEQLEN] == 1);

    if (!isKVSplit) {
        tilingHost[TILING_KVCORENUM] = 1;
        tilingHost[TILING_PROCESSNUM] = 0;
        for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
            uint32_t sortSeqIdx = sortedIndices[seqIdx];
            uint32_t kvSeqLen = *(mlaInfo.kvSeqLen + sortSeqIdx);
            int32_t tilingOffset = seqIdx * TILING_PARA_SIZE + TILING_HEAD_SIZE;
            tilingHost[tilingOffset + NUM15] = kvSeqLen;
            tilingHost[tilingOffset + NUM16] = 1;
        }
        return tilingHost[TILING_BATCH];
    }

    // T+1-style balanced KV split: distribute available cores among the (q==1) decode tasks
    // by repeatedly splitting the currently heaviest task, with a minimum tokens-per-split floor.
    // Each batch is one task here (split path requires maxQseqlen == 1).
    const uint32_t MIN_TOKENS_PER_SPLIT = static_cast<uint32_t>(mlaInfo.blockSize * NUM4);
    // Cap the per-task split count to a proven-safe value (the original common path always
    // used 8). This keeps each task's kvSplitCoreNum within the range the flash-decoding
    // reduction handles correctly, and together with totalAllocated <= blockDim guarantees
    // sum(kvSplitCoreNum) <= blockDim (i.e. at most one task per core).
    const uint32_t MAX_SPLIT_PER_TASK = 8;
    uint32_t taskCount = static_cast<uint32_t>(mlaInfo.batch);

    std::vector<uint32_t> alignedKvLens(taskCount);
    for (uint32_t seqIdx = 0; seqIdx < taskCount; seqIdx++) {
        uint32_t sortSeqIdx = sortedIndices[seqIdx];
        uint32_t kvSeqLen = *(mlaInfo.kvSeqLen + sortSeqIdx);
        alignedKvLens[seqIdx] = RoundUp(kvSeqLen, static_cast<uint32_t>(mlaInfo.blockSize));
    }

    std::vector<uint32_t> batchSplitNum(taskCount, 1);
    uint32_t totalAllocated = taskCount;
    while (totalAllocated < blockDim) {
        int32_t maxLoadIdx = -1;
        uint32_t maxLoad = 0;
        for (uint32_t i = 0; i < taskCount; i++) {
            uint32_t currentLoad = alignedKvLens[i] / batchSplitNum[i];
            uint32_t nextLoad = alignedKvLens[i] / (batchSplitNum[i] + 1);
            if (batchSplitNum[i] < MAX_SPLIT_PER_TASK && nextLoad >= MIN_TOKENS_PER_SPLIT && currentLoad > maxLoad) {
                maxLoad = currentLoad;
                maxLoadIdx = static_cast<int32_t>(i);
            }
        }
        if (maxLoadIdx == -1) {
            break;
        }
        batchSplitNum[maxLoadIdx]++;
        totalAllocated++;
    }

    uint32_t MAX_KV_SPLIT_NUM = 1;
    for (uint32_t num : batchSplitNum) {
        if (num > MAX_KV_SPLIT_NUM) {
            MAX_KV_SPLIT_NUM = num;
        }
    }

    // Build the cumulative-task prefix sum so the kernel can map process -> (task, kvIdx) tightly.
    uint32_t cuTaskVal = 0;
    int32_t cuTaskIdx = 0;
    tilingHost[CUTASK_START_OFFSET + cuTaskIdx++] = cuTaskVal;
    for (uint32_t seqIdx = 0; seqIdx < taskCount; seqIdx++) {
        int32_t tilingOffset = seqIdx * TILING_PARA_SIZE + TILING_HEAD_SIZE;
        uint32_t sortSeqIdx = sortedIndices[seqIdx];
        uint32_t kvSeqLen = *(mlaInfo.kvSeqLen + sortSeqIdx);

        uint32_t kvSplitPerCore;
        uint32_t kvSplitCoreNum;
        if (kvSeqLen == 0) {
            kvSplitPerCore = static_cast<uint32_t>(mlaInfo.blockSize);
            kvSplitCoreNum = 1;
        } else {
            uint32_t allocSplit = batchSplitNum[seqIdx];
            uint32_t kvSeqAlign = RoundUp(kvSeqLen, static_cast<uint32_t>(mlaInfo.blockSize));
            uint32_t kvSeqBlockNum = kvSeqAlign / mlaInfo.blockSize;
            uint32_t kvBlockPerCore = CeilDiv(kvSeqBlockNum, allocSplit);
            kvSplitPerCore = kvBlockPerCore * mlaInfo.blockSize;
            kvSplitCoreNum = CeilDiv(kvSeqLen, kvSplitPerCore);
        }
        tilingHost[tilingOffset + NUM15] = kvSplitPerCore;
        tilingHost[tilingOffset + NUM16] = kvSplitCoreNum;
        cuTaskVal += kvSplitCoreNum;
        tilingHost[CUTASK_START_OFFSET + cuTaskIdx++] = cuTaskVal;
    }
    tilingHost[TILING_PROCESSNUM] = cuTaskVal;
    tilingHost[TILING_KVCORENUM] = MAX_KV_SPLIT_NUM;

    // Set lOffsetInfo and OfdOffsetInfo
    AddrOffsets addrOffsets;
    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        uint32_t sortSeqIdx = sortedIndices[seqIdx];
        uint32_t qSeqlen = *(mlaInfo.qSeqLen + sortSeqIdx);
        uint32_t kvSeqLen = *(mlaInfo.kvSeqLen + sortSeqIdx);
        qSeqlen = (kvSeqLen == 0) ? 0 : qSeqlen;
        int32_t tilingOffset = seqIdx * TILING_PARA_SIZE + TILING_HEAD_SIZE;
        tilingHost[tilingOffset + NUM11] = GetHigh32Bit(addrOffsets.addrLSeqOffset);
        tilingHost[tilingOffset + NUM12] = GetLow32Bit(addrOffsets.addrLSeqOffset);
        tilingHost[tilingOffset + NUM13] = GetHigh32Bit(addrOffsets.addrOFdSeqOffset);
        tilingHost[tilingOffset + NUM14] = GetLow32Bit(addrOffsets.addrOFdSeqOffset);
        addrOffsets.addrLSeqOffset += static_cast<uint64_t>(mlaInfo.numHeads * qSeqlen * MAX_KV_SPLIT_NUM);
        addrOffsets.addrOFdSeqOffset += static_cast<uint64_t>(mlaInfo.numHeads * qSeqlen * mlaInfo.embeddingSize);
    }

    return cuTaskVal;
}

uint32_t GetKVSplitParamSpec(
    const MLAInfo& mlaInfo, uint32_t& blockDim, uint32_t* tilingHost, const std::vector<uint32_t>& sortedIndices)
{
    // TP1 scenario specialization.
    // Calculate the tiling parameters related to flash decoding
    uint32_t totalTaskNumSpec = tilingHost[TILING_TOTAL_QTOKENS];
    uint32_t formerTaskNum = totalTaskNumSpec;
    uint32_t tailTaskNum = 0;
    // Use the runtime AIC core count (blockDim) consistently so the former/tail split stays
    // valid on any core count. When numTokens <= blockDim all tasks are former (one per core,
    // no KV split); only align former tasks to full core waves when a wave is actually filled.
    if (totalTaskNumSpec >= blockDim && mlaInfo.numTokens % blockDim <= blockDim / 2 && mlaInfo.batch <= blockDim * 2) {
        uint32_t processLoop = totalTaskNumSpec / blockDim;
        formerTaskNum = processLoop * blockDim;
        tailTaskNum = totalTaskNumSpec - formerTaskNum;
    }

    tilingHost[TILING_FORMERTASKNUM] = formerTaskNum;
    tilingHost[TILING_TAILTASKNUM] = tailTaskNum;
    if (tailTaskNum == 0) {
        tilingHost[TILING_KVCORENUM] = 1;
        tilingHost[TILING_KVSPLIT] = tilingHost[TILING_MAX_KVSEQLEN];
        tilingHost[TILING_PROCESSNUM] = 0;
        return blockDim;
    }

    const uint32_t MIN_TOKENS_PER_SPLIT = static_cast<uint32_t>(mlaInfo.blockSize * NUM4);

    std::vector<uint32_t> alignedKvLens;
    int32_t prevTaskNumTemp = 0;
    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        uint32_t sortSeqIdx = sortedIndices[seqIdx];
        uint32_t qSeqLen = *(mlaInfo.qSeqLen + sortSeqIdx);
        if (prevTaskNumTemp + qSeqLen < formerTaskNum) {
            prevTaskNumTemp += qSeqLen;
            continue;
        }
        uint32_t kvSeqLen = *(mlaInfo.kvSeqLen + sortSeqIdx);
        alignedKvLens.push_back(RoundUp(kvSeqLen, static_cast<uint32_t>(mlaInfo.blockSize)));
    }

    uint32_t tailSeqCount = alignedKvLens.size();
    std::vector<uint32_t> batchSplitNum(tailSeqCount, 1);
    uint32_t totalAllocated = tailTaskNum;
    while (totalAllocated < blockDim) {
        int32_t maxLoadIdx = -1;
        uint32_t maxLoad = 0;

        for (int32_t i = 0; i < tailSeqCount; i++) {
            uint32_t currentLoad = alignedKvLens[i] / batchSplitNum[i];
            uint32_t nextLoad = alignedKvLens[i] / (batchSplitNum[i] + 1);
            if (nextLoad >= MIN_TOKENS_PER_SPLIT && currentLoad > maxLoad) {
                maxLoad = currentLoad;
                maxLoadIdx = i;
            }
        }

        if (maxLoadIdx != -1) {
            batchSplitNum[maxLoadIdx]++;
        } else if (tailSeqCount > 0) {
            int32_t longestIdx = 0;
            for (int32_t i = 1; i < tailSeqCount; i++) {
                if (alignedKvLens[i] > alignedKvLens[longestIdx]) {
                    longestIdx = i;
                }
            }
            batchSplitNum[longestIdx]++;
        }
        totalAllocated++;
    }
    uint32_t MAX_KV_SPLIT_NUM = 1;
    for (uint32_t num : batchSplitNum) {
        if (num > MAX_KV_SPLIT_NUM) {
            MAX_KV_SPLIT_NUM = num;
        }
    }
    uint32_t cuTaskVal = 0;
    int32_t cuTaskIdx = 0;
    tilingHost[CUTASK_START_OFFSET + cuTaskIdx++] = cuTaskVal;

    int32_t prevTaskNum = 0;
    int32_t tailSeqIdx = 0;
    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        uint32_t sortSeqIdx = sortedIndices[seqIdx];
        uint32_t qSeqLen = *(mlaInfo.qSeqLen + sortSeqIdx);
        uint32_t kvSeqLen = *(mlaInfo.kvSeqLen + sortSeqIdx);

        if (prevTaskNum + qSeqLen < formerTaskNum) {
            prevTaskNum += qSeqLen;
            continue;
        }

        uint32_t allocSplit = 1;
        if (tailSeqIdx < tailSeqCount) {
            allocSplit = batchSplitNum[tailSeqIdx++];
        }

        uint32_t kvSeqAlign = RoundUp(kvSeqLen, static_cast<uint32_t>(mlaInfo.blockSize));
        uint32_t kvSeqBlockNum = kvSeqAlign / mlaInfo.blockSize;
        uint32_t kvBlockPerCore = CeilDiv(kvSeqBlockNum, allocSplit);
        uint32_t kvSplitPerCore = kvBlockPerCore * mlaInfo.blockSize;
        uint32_t kvSplitCoreNum = CeilDiv(kvSeqLen, kvSplitPerCore);

        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq++) {
            if (prevTaskNum >= formerTaskNum) {
                cuTaskVal += kvSplitCoreNum;
                tilingHost[CUTASK_START_OFFSET + cuTaskIdx++] = cuTaskVal;
                int32_t tilingOffset = TILING_HEAD_SIZE + PARA_TILING_ELENUM_SPEC * prevTaskNum;
                tilingHost[tilingOffset + NUM15] = kvSplitPerCore;
                tilingHost[tilingOffset + NUM16] = kvSplitCoreNum;
            }
            prevTaskNum++;
        }
    }
    tilingHost[TILING_PROCESSNUM] = cuTaskVal;
    tilingHost[CUTASK_START_OFFSET + cuTaskIdx++] = TILING_HEAD_SIZE;
    tilingHost[TILING_KVCORENUM] = MAX_KV_SPLIT_NUM;

    AddrOffsets addrOffsets{};
    prevTaskNum = 0;

    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        uint32_t sortSeqIdx = sortedIndices[seqIdx];
        uint32_t qSeqLen = *(mlaInfo.qSeqLen + sortSeqIdx);
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq++) {
            int32_t tilingOffset = TILING_HEAD_SIZE + PARA_TILING_ELENUM_SPEC * prevTaskNum;
            tilingHost[tilingOffset + NUM11] = GetHigh32Bit(addrOffsets.addrLSeqOffset);
            tilingHost[tilingOffset + NUM12] = GetLow32Bit(addrOffsets.addrLSeqOffset);
            tilingHost[tilingOffset + NUM13] = GetHigh32Bit(addrOffsets.addrOFdSeqOffset);
            tilingHost[tilingOffset + NUM14] = GetLow32Bit(addrOffsets.addrOFdSeqOffset);
            addrOffsets.addrLSeqOffset += static_cast<uint64_t>(mlaInfo.numHeads * MAX_KV_SPLIT_NUM);
            addrOffsets.addrOFdSeqOffset += static_cast<uint64_t>(mlaInfo.numHeads * mlaInfo.embeddingSize);
            prevTaskNum++;
        }
    }

    return tailTaskNum * MAX_KV_SPLIT_NUM;
}

void swapIndices(std::vector<uint32_t>& indices, int i, int j)
{
    uint32_t temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
}

int partitionIndices(std::vector<uint32_t>& indices, const int32_t* kvSeqLen, const int32_t* qSeqLen, int low, int high)
{
    uint32_t pivotIndex = indices[high];
    int32_t pivotKv = kvSeqLen[pivotIndex];
    int32_t pivotQ = qSeqLen[pivotIndex];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        uint32_t currIndex = indices[j];
        int32_t currKv = kvSeqLen[currIndex];
        int32_t currQ = qSeqLen[currIndex];

        bool needSwap = (currKv > pivotKv) || (currKv == pivotKv && currQ > pivotQ);
        if (needSwap) {
            i++;
            swapIndices(indices, i, j);
        }
    }
    swapIndices(indices, i + 1, high);
    return (i + 1);
}

void quickSortIndices(
    std::vector<uint32_t>& indices, const int32_t* kvSeqLen, const int32_t* qSeqLen, int low, int high)
{
    if (low < high) {
        int pi = partitionIndices(indices, kvSeqLen, qSeqLen, low, high);
        quickSortIndices(indices, kvSeqLen, qSeqLen, low, pi - 1);
        quickSortIndices(indices, kvSeqLen, qSeqLen, pi + 1, high);
    }
}

int32_t GetMLATilingParam(const MLAInfo& mlaInfo, uint32_t& blockDim, uint32_t* tilingHost)
{
    if (tilingHost == nullptr || mlaInfo.qSeqLen == nullptr || mlaInfo.kvSeqLen == nullptr) {
        cerr << "[ERROR] pointer tilingHost or seq is nullptr." << endl;
        return -1;
    }
    if (mlaInfo.blockSize != NUM128) {
        cerr << "[ERROR] blockSize != 128 is not supported." << endl;
        return -1;
    }

    std::vector<AddrOffsetPerBatch> addrOffsetsPerBatch(mlaInfo.batch);
    uint64_t currentQSeqOffset = 0;
    uint64_t currentQSeqRopeOffset = 0;
    uint64_t currentMaskBatchOffset = 0;

    for (int32_t i = 0; i < mlaInfo.batch; ++i) {
        int32_t qSeqLen = *(mlaInfo.qSeqLen + i);

        uint64_t deltaQSeq = static_cast<uint64_t>(mlaInfo.numHeads * mlaInfo.embeddingSize * qSeqLen);
        uint64_t deltaQSeqRope = static_cast<uint64_t>(mlaInfo.numHeads * mlaInfo.embeddingSizeRope * qSeqLen);
        uint64_t deltaMaskBatch = static_cast<uint64_t>(mlaInfo.maxKvSeqlen * qSeqLen);

        addrOffsetsPerBatch[i].qSeqOffset = currentQSeqOffset;
        addrOffsetsPerBatch[i].qSeqRopeOffset = currentQSeqRopeOffset;
        addrOffsetsPerBatch[i].maskBatchOffset = currentMaskBatchOffset;

        currentQSeqOffset += deltaQSeq;
        currentQSeqRopeOffset += deltaQSeqRope;
        currentMaskBatchOffset += deltaMaskBatch;
    }

    std::vector<uint32_t> sortedIndices(mlaInfo.batch);
    for (uint32_t i = 0; i < mlaInfo.batch; ++i) {
        sortedIndices[i] = i;
    }

    std::vector<std::vector<uint32_t>> realPreTaskNums(mlaInfo.batch, std::vector<uint32_t>(NUM512));
    int32_t preTaskNum = 0;
    for (int32_t i = 0; i < mlaInfo.batch; ++i) {
        uint32_t qSeqlen = static_cast<uint32_t>(*(mlaInfo.qSeqLen + i));
        for (int32_t token = 0; token < qSeqlen; token++) {
            realPreTaskNums[i][token] = preTaskNum;
            preTaskNum++;
        }
    }

    if (mlaInfo.numTokens > 0) {
        quickSortIndices(sortedIndices, mlaInfo.kvSeqLen, mlaInfo.qSeqLen, 0, mlaInfo.batch - 1);
    }

    int32_t maxQseqlen = 0;
    int64_t totalRequiredBlocks = 0;
    for (int32_t seqIdx = 0; seqIdx < mlaInfo.batch; seqIdx++) {
        int32_t qSeqLen = *(mlaInfo.qSeqLen + seqIdx);
        if (qSeqLen > NUM4) {
            cerr << "[ERROR] qSeqLen > 4 is not supported." << endl;
            return -1;
        }
        int32_t kvSeqLen = *(mlaInfo.kvSeqLen + seqIdx);
        qSeqLen = (kvSeqLen == 0) ? 0 : qSeqLen;
        maxQseqlen = std::max(qSeqLen, maxQseqlen);
        totalRequiredBlocks += (static_cast<int64_t>(kvSeqLen) + mlaInfo.blockSize - 1) / mlaInfo.blockSize;
    }
    if (totalRequiredBlocks > mlaInfo.numBlocks) {
        cerr << "[ERROR] the number of K and V blocks (" << totalRequiredBlocks
             << ") is too big to fit in the paged cache (numBlocks = " << mlaInfo.numBlocks << ")." << endl;
        return -1;
    }
    float tor = static_cast<float>(1.0 / sqrt(1.0 * (mlaInfo.embeddingSize + mlaInfo.embeddingSizeRope)));
    uint32_t* torPtr = reinterpret_cast<uint32_t*>(&tor);
    uint32_t specStrategyFlag = (mlaInfo.numHeads == NUM128) ? 1 : 0;

    if (specStrategyFlag) {
        GetMLATilingSpec(mlaInfo, blockDim, tilingHost, sortedIndices, realPreTaskNums);
    } else {
        GetMLATilingCommon(mlaInfo, blockDim, tilingHost, sortedIndices, addrOffsetsPerBatch);
    }
    GetTilingHead(mlaInfo, tilingHost, torPtr, maxQseqlen, specStrategyFlag);
    if (specStrategyFlag) {
        GetKVSplitParamSpec(mlaInfo, blockDim, tilingHost, sortedIndices);
    } else {
        GetKVSplitParam(mlaInfo, blockDim, tilingHost, sortedIndices);
    }
    return 0;
}
} // namespace MLATiling
