# CATLASS TRMM 样例介绍

## 原型设计

TRMM（Triangular Matrix-Matrix Multiplication）用于计算三角矩阵与普通稠密矩阵的矩阵乘：

```text
side = 0, Left : C = alpha * op(A) * B，A 为三角矩阵
side = 1, Right: C = alpha * A * op(B)，B 为三角矩阵
```

当前样例使用 `float32` 输入输出，输出矩阵 C 固定为 RowMajor 排布。

| 名称/Name | 类型/Class | 数据类型/Dtype | 维度/Dims | 格式/Format | 描述/Description |
| --------- | ---------- | -------------- | --------- | ----------- | ---------------- |
| matA      | inTensor   | float          | [m, k]    | ND          | 左输入矩阵，`side=left` 时为三角矩阵 |
| matB      | inTensor   | float          | [k, n]    | ND          | 右输入矩阵，`side=right` 时为三角矩阵 |
| matC      | outTensor  | float          | [m, n]    | ND          | 输出矩阵 |

功能参数如下：

| 参数 | 作用 | 当前支持 |
| ---- | ---- | -------- |
| `M` / `N` | 输出矩阵 C 的行数和列数 | 正整数 |
| `side` | 指定三角矩阵位于左输入还是右输入 | `0=left`，`1=right` |
| `uplo` | 指定三角矩阵有效区域 | `0=lower`，`1=upper` |
| `trans` | 指定是否按转置语义访问三角矩阵 | `0=no transpose`，`1=transpose triangular input` |
| `diag` | 指定三角矩阵对角线语义 | 当前仅支持 `0=non-unit diagonal` |
| `alpha` | TRMM 输出缩放系数 | `float` |

`K` 不是独立功能参数，而是由 `side` 推导：`side=left` 时 `K=M`，`side=right` 时 `K=N`。

## 样例实现

CATLASS [`trmm` 样例](./README.md)算子是基于 CATLASS Gemm API 实现的三角矩阵乘算子。实现主体复用通用矩阵乘组件，TRMM 自身负责三角矩阵语义参数、有效 K 范围裁剪以及 `alpha` 后处理。

关键组件包括：

- **Example 组装**：[trmm.cpp](./trmm.cpp)
- **Kernel 实现**：[trmm.hpp](../../include/catlass/gemm/kernel/trmm.hpp)
- **Block 组件**：
  - 通用 MMAD 组件 [block_mmad_pingpong_tla.hpp](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp)
  - 基本块分发策略 [block_swizzle.hpp](../../include/catlass/gemm/block/block_swizzle.hpp)
- **Tile 组件**：
  - GM/L1/L0 搬运组件 [tile_copy_tla.hpp](../../include/catlass/gemm/tile/tile_copy_tla.hpp)
  - AIV 后处理搬运组件 [copy_gm_to_ub_tla.hpp](../../include/catlass/epilogue/tile/copy_gm_to_ub_tla.hpp) 和 [copy_ub_to_gm_tla.hpp](../../include/catlass/epilogue/tile/copy_ub_to_gm_tla.hpp)
- **CMake 组装**：[CMakeLists.txt](./CMakeLists.txt)

## Example 组装

### Host 侧参数组织

`trmm` 的命令格式为：

```bash
./output/bin/76_trmm m n side uplo trans diag alpha [device_id]
```

Host 侧通过 `TrmmOptions` 解析命令行参数，并根据 `side` 推导 `K`：

```cpp
problemShape.k() = (side == 0) ? problemShape.m() : problemShape.n();
```

入口会检查 `m/n > 0`、`side/uplo/trans` 取值合法，并拒绝当前不支持的 `diag != 0`。Kernel 侧 `Arguments` 只保存执行所需的矩阵形状、A/B/C GM 地址、layout、TRMM 功能参数和 `alpha`。

### 输入构造

Host 示例在 CPU 侧构造输入：

- `side=left` 时，matA 为三角矩阵，matB 为 dense 矩阵。
- `side=right` 时，matA 为 dense 矩阵，matB 为三角矩阵。
- 三角矩阵按原始 `uplo` 保留 active half，inactive half 写 0。
- dense 矩阵使用 [FillRandomData](../../examples/common/golden/fill_data.hpp)，三角矩阵使用 [FillTriangularData](../../examples/common/golden/fill_data.hpp)。

inactive half 置零是当前 kernel 满足 TRMM 语义的前置条件。Kernel 做 tile 级 K 范围裁剪，不做逐元素三角 mask；同一个输出 tile 内仍可能覆盖到三角矩阵 inactive half，因此 inactive half 需要由调用侧保证为 0。

### Layout 选择

Host 示例用 layout 表达三角矩阵转置语义，避免在 kernel 主体中增加转置分支：

| 条件 | LayoutA | LayoutB |
| ---- | ------- | ------- |
| `trans=0` | RowMajor | RowMajor |
| `trans=1 && side=left` | ColumnMajor | RowMajor |
| `trans=1 && side=right` | RowMajor | ColumnMajor |

输出 C 固定为 RowMajor。

### alpha 处理

Host 示例根据三角矩阵元素量和输出元素量决定是否把 `alpha` 预融合进三角矩阵：

```cpp
uint64_t triElements = static_cast<uint64_t>(k) * static_cast<uint64_t>(k);
uint64_t outputElements = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
bool fuseAlphaInPrepare = (triElements <= outputElements);
float prepareAlpha = fuseAlphaInPrepare ? alpha : 1.0f;
float kernelAlpha = fuseAlphaInPrepare ? 1.0f : alpha;
```

- 如果三角矩阵元素数不大于输出元素数，把 `alpha` 乘到三角输入上，kernel 接收 `alpha=1`，无需 AIV 后处理。
- 否则保留 `kernelAlpha=alpha`，AIC 先写回矩阵乘结果，AIV 再对 C 做元素级乘法。

当前 Host 示例没有 `alpha == 0` 的 early return。`alpha=0` 会通过上述预融合或 AIV 乘零路径得到全零输出。

## Kernel 方案设计

### 模板组装

TRMM kernel 复用 GEMM 的 `BlockMmadTla` 完成矩阵乘主体，TRMM 自身只处理三角语义相关逻辑：

| 模块 | 作用 |
| ---- | ---- |
| `DispatchPolicy` | 选择 Atlas A2 上的 MMAD 执行策略 |
| `PackedTileCopyTla` | 负责 A/B/C 的 TLA tile 搬运 |
| `BlockMmadTla` | 完成裁剪后的 `A_tile x B_tile -> C_tile` |
| `BlockScheduler` | 按 C 的 M/N tile 分配 AIC 工作 |
| `Trmm::operator()<AIC>` | 计算有效 K 范围并调用 `BlockMmadTla` |
| `Trmm::operator()<AIV>` | 仅在 `alpha != 1` 时对 C 做后处理 |

当前样例的关键实例化点：

- `ElementA/B/C = float`
- `ArchTag = Arch::AtlasA2`
- `useHF32 = false`，AIC 路径显式关闭 HF32
- `BlockEpilogue = void`，不走通用 epilogue
- A/B/C 都通过 `MakeTensor` 和 `GetTile` 形成 TLA 视图后传入 `BlockMmadTla`

### AIC 计算流程

AIC 的计算粒度是 C 矩阵上的二维 tile，整体流程如下：

```text
BlockScheduler 生成 C tile
  -> 计算当前 tile 的 M/N 起止位置
  -> 按 TRMM 规则裁剪 K 起止位置
  -> K 范围非空时切出 A/B/C 的 TLA tile
  -> 调用 BlockMmadTla 完成当前 C tile
```

`actualBlockShape` 用于处理 M/N 边界 tile；TRMM 的核心差异是随后根据三角矩阵所在侧和上下三角方向裁剪 K 维。这样 `BlockMmadTla` 看到的仍是普通 GEMM 形态，不需要在 MMAD 内部加入三角判断。

### 有效 K 范围裁剪

三角矩阵转置后，上三角和下三角的有效区域互换，因此 kernel 先计算：

```cpp
uint32_t effectiveUplo = params.uplo ^ params.trans;
```

随后根据当前 C tile 的 M/N 范围裁剪 K：

| 场景 | 有效三角方向 | 裁剪行为 | 说明 |
| ---- | ------------ | -------- | ---- |
| `side=left` | lower | `kEnd = min(kEnd, mEnd)` | 当前 M tile 只需要累加到本 tile 最大行号 |
| `side=left` | upper | `kStart = min(mStart, kEnd)` | 跳过当前 M tile 之前的无效 K |
| `side=right` | lower | `kStart = min(nStart, kEnd)` | 跳过当前 N tile 之前的无效 K |
| `side=right` | upper | `kEnd = min(kEnd, nEnd)` | 当前 N tile 只需要累加到本 tile 最大列号 |

如果 `kStart >= kEnd`，说明当前 C tile 没有有效累加区间，AIC 直接跳过；否则更新 `actualBlockShape.k() = kEnd - kStart`，并用裁剪后的 K 范围切出 A/B/C tile。

### AIV alpha 后处理

AIV 路径只在 `params.alpha != 1.0f` 时参与，核心操作是：

```cpp
copyGmToUbC(tensorUbC, tensorBlockC);
AscendC::Muls(tensorUbC.data(), tensorUbC.data(), static_cast<ElementC>(params.alpha), elementCount);
copyUbToGmC(tensorBlockC, tensorUbC);
```

AIC 在写回有效 C tile 后设置 cross-core flag，AIV 等待该 flag 后处理同一块有效 C 区域。AIV 复用 AIC 的 K 裁剪规则，因此不会处理被 AIC 跳过的空 tile。

## 空间分配

### 全局空间

| 类别 | 分配方式 | 说明 |
| ---- | -------- | ---- |
| GM 输入 A/B | 调用侧传入 | 三角矩阵仍按稠密矩阵存储，inactive half 由 Host 置零 |
| GM 输出 C | 调用侧传入 | RowMajor 输出 |
| Workspace | 不分配 | `GetWorkspaceSize()` 固定返回 0 |
| 三角元数据 | 不分配 | 不做压缩存储，也不维护三角索引表 |

### L1/L0 空间

当前 tile 以 `M x N x K` 表示，`float32` 按 4 Byte 估算，单份 tile 数据量如下：

| L1 K 配置 | L1 A tile | L1 B tile | L1 合计 | L0A | L0B | L0C |
| --------- | --------- | --------- | ------- | --- | --- | --- |
| `128x128x256` | 128 KB | 128 KB | 256 KB | 32 KB | 32 KB | 64 KB |
| `128x128x128` | 64 KB | 64 KB | 128 KB | 32 KB | 32 KB | 64 KB |
| `128x128x64` | 32 KB | 32 KB | 64 KB | 32 KB | 32 KB | 64 KB |

表中 L1 是单份 A/B tile 的理论数据量；实际预取和流水由 `BlockMmadTla` 与 `DispatchPolicy` 管理。L0 tile 当前固定为 `128x128x64`，因此 L0A/L0B/L0C 的单份数据量保持一致。

### UB 与同步

UB 只在 AIV 后处理路径使用，用于暂存 C 的子块并执行 `Muls(alpha)`。当 Host 已经预融合 `alpha`，或者 `kernelAlpha == 1.0f` 时，AIV 直接返回，不消耗这部分 UB 路径。

AIC 与 AIV 之间通过 cross-core flag 同步，保证 AIV 只读取已经由 AIC 写回的 C tile。除 C 子块临时 UB 外，当前 kernel 没有额外常驻缓冲、partial sum 缓冲或全局归约空间。

## Tile Variant 与 Dispatch

Host 根据矩阵形状和三角方向选择 L1 K 与 swizzle：

| Variant | 选择条件 | L1 Tile | L0 Tile | Swizzle |
| ------- | -------- | ------- | ------- | ------- |
| `TILE_DEFAULT` | 其他通用场景 | `128x128x256` | `128x128x64` | `<3,0>` |
| `TILE_DEFAULT_SWIZZLE31` | `m >= 2048 && m < n` | `128x128x256` | `128x128x64` | `<3,1>` |
| `TILE_SMALL_RIGHT_LOWER` | `side=right && effective lower && n<=512` | `128x128x64` | `128x128x64` | `<1,0>` |
| `TILE_SMALL_RIGHT_UPPER` | `side=right && effective upper && 256<n<=512` | `128x128x64` | `128x128x64` | `<4,1>` |
| `TILE_SMALL_RIGHT_K128` | `side=right && effective upper && n<=256` | `128x128x128` | `128x128x64` | `<1,1>` |
| `TILE_SMALL_LEFT_K128` | `side=left && effective upper && m<=256` | `128x128x128` | `128x128x64` | `<1,1>` |
| `TILE_SMALL_LEFT_UPPER` | `side=left && effective upper && 256<m<=512` | 实际回落 default | 实际回落 default | `<3,0>` |
| `TILE_SMALL_LEFT_LOWER` | `side=left && effective lower && m<=512` | 实际回落 default | 实际回落 default | `<3,0>` |

选择逻辑先计算 `effectiveUplo = uplo ^ trans`，再区分三角矩阵位于左侧还是右侧。Small Left Lower/Upper 仍有 enum 和选择分支，但在当前 Host dispatch 中回落到 default tile；只有 Small Left K128 使用专门的 `128x128x128` L1 K。

## 编译运行

CMake 写法与其他 matmul 类样例一致：

```cmake
set_source_files_properties(trmm.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(76_trmm mix trmm.cpp)
target_link_libraries(trmm PRIVATE pthread)
```

在 CATLASS 仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh 76_trmm
```

编译成功后会生成：

```text
output/bin/76_trmm
```

运行命令格式为：

```bash
./output/bin/76_trmm m n side uplo trans diag alpha [device_id]
```

运行示例：

```bash
./output/bin/76_trmm 128 96 0 0 0 0 1.0 0
./output/bin/76_trmm 128 96 0 1 1 0 0.5 0
./output/bin/76_trmm 96 128 1 0 0 0 1.0 0
./output/bin/76_trmm 96 128 1 1 1 0 0.5 0
```

Profiling 示例：

```bash
WARMUP=5 REPEAT=20 SKIP_OUTPUT=1 ./output/bin/76_trmm 4608 256 1 1 1 0 1.0 0
```

## 约束说明

- 当前公开路径只支持 `float32` 输入输出。
- `diag` 仅支持 `0`，Host 示例和 kernel `CanImplement` 都会拒绝 `diag != 0`。
- `side=left` 时 `K=M`，`side=right` 时 `K=N`。
- 输出 C 固定为 RowMajor。
- 三角矩阵 inactive half 必须由调用侧置零；kernel 不做逐元素三角 mask。
- 不使用 workspace，不支持 SplitK、StreamK、partial sum 或全局归约。
- 不支持 `beta`，语义固定为 `C = alpha * TRMM(...)`。
