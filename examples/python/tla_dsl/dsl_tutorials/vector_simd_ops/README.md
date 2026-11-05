# 基础向量运算端到端示例

本目录下提供的系列样例是 **CATLASS Python** 下的多种向量（Vector）操作示例。

## 功能说明

向量运算是 NPU 上基础的计算原语，在 AIV 物理核上执行，整体执行流程包括：

1. 构造 GM （Global memory）上的输入/输出 `tla.Tensor`；
2. 启动MTE2，将数据搬运至 UB （Unified Buffer）；
3. 加载至寄存器上，以 `VL`（向量寄存器位宽， 256 字节）为粒度分块执行各类向量指令；
4. 输出回 UB，然后启动 MTE3，将数据搬运回 GM。

以下是一个最小示例，以 `tla.abs` 为例，一个 Vector 类 op 从 GM 搬运到 UB、加载到寄存器计算、再写回 GM 的完整过程：

```python
VECTOR_ELE = 400                              # 输入/输出 tensor 的元素个数
VL_ELE = 64                                   # 单条向量指令处理的元素数（256B / 4B）
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE   # 分块数
_DTYPE = tla.Float32

@tla.kernel
def vector_op(mem_in: tla.Tensor, mem_out: tla.Tensor) -> None:
    # 跨流水同步 flag：MTE2->VECTOR 表示“UB 数据已就绪”，
    # VECTOR->MTE3 表示“寄存器计算已完成”。
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    # 在 GM 上建立输入/输出视图：形状 VECTOR_ELE，坐标从 0 开始。
    gm_in = tla.tile_view(mem_in, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    gm_out = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    # 在 UB 上申请缓冲区，并构造与 GM 视图同布局的 tensor。
    # allocate(元素个数, dtype, 地址空间, 对齐字节数)
    ub_in = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256),
        gm_in,
        tla.arch.RowMajor,
    )
    ub_out = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256),
        gm_out,
        tla.arch.RowMajor,
    )

    # 向量相关的搬运与计算统一放在 tla.vector() 上下文中。
    with tla.vector():
        # 1) GM -> UB（MTE2）；用 flag 保证搬运完成后才进入寄存器计算。
        tla.copy(ub_in, gm_in)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        # 2) 寄存器级计算：以 VL 为粒度分块，load -> op -> store。
        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                in_tile = tla.tile_view(ub_in, tla.make_shape(VL_ELE), tla.make_coord(i))
                out_tile = tla.tile_view(ub_out, tla.make_shape(VL_ELE), tla.make_coord(i))

                # vector op 操作
                out_tile.store(tla.abs(in_tile.load()))

        # 3) UB -> GM（MTE3）；用 flag 保证计算完成后才回写。
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_out, ub_out)

        # 4) 流水收尾同步。
        tla.pipe_barrier(tla.pipes.ALL)
```

## 代码组织

本目录组织结构及文件概述如下：

```plain
./vector_ops
├── arange_op.py
├── binary_op.py
├── bitwise_ops.py
├── cast_multi.py
├── compare_mask.py
├── gather_reg_op.py
├── gather_op.py
├── interleave_op.py
├── load_and_store_scalar_after_reduction.py
├── load_dintlv_op.py
├── load_mask_dist.py
├── load_store_mask.py
├── load_store_update.py
├── load_store_block_update.py
├── load_store_unalign.py
├── load_us_b8_op.py
├── mask_select_op.py
├── masked_binary.py
├── multi_binary.py
├── pack_op.py
├── reduction_ops.py
├── register_control_flow.py
├── histograms_op.py
├── squeeze_op.py
├── store_pack.py
├── unary_ops.py
├── vector_op_harness.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`vector_op_harness.py`**](vector_op_harness.py) | 公共测试框架，用于执行批量测试，封装命令行参数解析、dtype/shape 配置、输入构造、golden 校验与批量/扫描执行逻辑。 |
| [**`binary_op.py`**](binary_op.py) | 二元运算示例，支持 `add`/`sub`/`mul`/`div`/`max`/`min`，以及非对齐搬运（`add_unalign`）与广播（`add_brc_b32`）变体。 |
| [**`unary_ops.py`**](unary_ops.py) | 一元运算示例，支持 `exp`/`log`/`sqrt`/`abs`/`neg`，以及带掩码的变体。 |
| [**`multi_binary.py`**](multi_binary.py) | 多算子二元计算示例，演示多级混合计算过程。 |
| [**`masked_binary.py`**](masked_binary.py) | 多算子掩码二元运算示例，演示不同掩码模式下 `add`/`sub`/`mul`/`div` 向量计算。 |
| [**`bitwise_ops.py`**](bitwise_ops.py) | 按位运算示例，支持 `bitwise_and`/`bitwise_or`/`bitwise_xor`/`bitwise_not`。 |
| [**`compare_mask.py`**](compare_mask.py) | 比较掩码示例，覆盖 `tla.cmp` 的六种比较模式（lt/le/gt/ge/eq/ne）、向量-标量比较；另含 u8/u16/u32（`VCmpU`，操作数越过有符号边界）。 |
| [**`cast_multi.py`**](cast_multi.py) | `tla.cast` 类型转换示例，覆盖 `f32` / `f16` / `bf16` / `i32` / `i16` / `i8` 之间的多级类型转换；另含 unsigned `f16↔u8`（值 ≥128 咬住 signed twin）。 |
| [**`arange_op.py`**](arange_op.py) | `tla.arange` 序列生成示例，在 UB 上生成递增/递减（`increase`/`decrease`）序列。 |
| [**`gather_reg_op.py`**](gather_reg_op.py) | `tla.gather` Reg→Reg 独立上板样例：`vselr`（源为寄存器）；`--mixed` 走 AIC/AIV mix（回归 #822）。 |
| [**`gather_op.py`**](gather_op.py) | `tla.gather` UB→Reg 按索引收集示例（独立脚本），依据索引向量从 UB tile 收集元素。 |
| [**`interleave_op.py`**](interleave_op.py) | 交错运算示例，基于两路输入分别产出交错后的两路输出。 |
| [**`load_dintlv_op.py`**](load_dintlv_op.py) |双目的交织加载示例，单次加载拆出偶数/奇数两路寄存器。 |
| [**`load_us_b8_op.py`**](load_us_b8_op.py) | 上采样加载示例，将 i8 元素上采样至 VL 寄存器（仅 i8）。 |
| [**`load_mask_dist.py`**](load_mask_dist.py) | `MaskLoadParams`：`ds`/`us` * `--mask-dtype i8\|i16\|i32` → `tla.where`. |
| [**`load_store_mask.py`**](load_store_mask.py) | 掩码搬运往返示例，掩码 UB 与伴随向量同 dtype；另含 u8 GM↔UB（含 128..255）。 |
| [**`pack_op.py`**](pack_op.py) | `tla.pack` 示例，覆盖 MaskSSA 宽度扩展，并验证低半/高半结果。 |
| [**`mask_select_op.py`**](mask_select_op.py) | MaskSSA 选择示例，使用 `H` 在 `Q` 和 `M4` 两个 predicate 之间逐 bit 选择。 |
| [**`load_store_update.py`**](load_store_update.py) | 对齐 `PostMode.UPDATE`：`tile.load` / `tile.store`（`add`，可选 `--dynamic-extent`）。 |
| [**`load_store_block_update.py`**](load_store_block_update.py) | Block `PostMode.UPDATE`：`BlockLoadParams` / `BlockStoreParams` 滑动窗口（`add`，`block_stride=1`）。 |
| [**`load_store_unalign.py`**](load_store_unalign.py) | 非对齐 PostMode UPDATE：dense `tile_extent`（`origin_shape`），每趟自含双指令（`Unalign*Params(post_mode=update)`，`add`；支持 `--tile-extent 1`）。 |
| [**`store_pack.py`**](store_pack.py) | 压缩存储示例，取得低半有效数位做紧凑存储。 |
| [**`reduction_ops.py`**](reduction_ops.py) | `tla.reduce` 归约示例，支持 `add`/`max`/`min` 三种模式，覆盖对齐与非对齐场景。 |
| [**`histograms_op.py`**](histograms_op.py) | `tla.histograms` 独立上板样例：FREQUENCY/ACCUMULATE*BIN0/BIN1（`full(0)` 清零 / 跨调用累加 / 全零 CDF）。 |
| [**`squeeze_op.py`**](squeeze_op.py) | `tla.squeeze` 示例，按掩码（M4）压缩有效 lane 并紧凑写回。 |
| [**`register_control_flow.py`**](register_control_flow.py) | 控制流示例，演示 for-loop 控制流及 `VectorSSA` 与 `MaskSSA` 跨循环边界存活。 |
| [**`load_and_store_scalar_after_reduction.py`**](load_and_store_scalar_after_reduction.py) | UB 标量访问示例，对归约结果做标量 `load` / `store` 操作。 |

 - 各样例支持的数据类型及其他约束见 [子测试项说明](#子测试项说明) 中各子节的 "约束说明"。

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../python/tla_dsl/docs/zh/quick_start.md)完成部署。

### 命令行参数

各样例通过公共测试组件提供统一的命令行接口：

```text
<script.py> [op] [--sweep] [--batch-run {op ...}]
            [--device DEVICE] [--block-num BLOCK_NUM]
            [--dtype {f32,f16,bf16,i8,i16,i32}]
            [--shape N] [--shapes N ...] [--sizes N ...]
            [--dtypes ...] [--all-dtypes] [--batch-size {1..4}]
            [--sentinel SENTINEL] [--atol ATOL] [--fail-fast]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `op` | -（位置参数，必填或与 `--batch-run` 二选一） | 待执行的算子名，取值见「各子测试项说明」。 |
| `--sweep` | `None` | 全量扫描模式，对 `--dtypes` / `--shapes` / `--sizes` 的组合逐一执行，默认不启用。 |
| `--batch-run` | - | 批量模式，将多个算子打包作多核并发执行。 |
| `--device` | `2` | 上板执行使用的 NPU 设备号。 |
| `--block-num` | 各脚本默认值 | 所启用的 AI Core 核数。 |
| `--dtype` | `f32` | 操作数数据类型。 |
| `--shape` | 各脚本默认向量长度 | 一维扁平向量的元素个数（例如 `--shape 400`）。 |
| `--shapes` / `--sizes` | - | 供 `--sweep` 使用的一维形状列表。 |
| `--dtypes` | 全部支持类型 | 供 `--sweep` 使用的数据类型列表。 |
| `--all-dtypes` | `None` | 对全部支持的数据类型逐一执行，默认不启用。 |
| `--batch-size` | `4` | 批量模式下每个 kernel 打包的算子数（1~4）。 |
| `--sentinel` | 各算子输出的初始默认值 | 初始化输出的值。 |
| `--atol` | 各算子默认绝对误差 | 浮点精度校验的绝对误差阈值。 |
| `--fail-fast` | `None` | 扫描/批量模式遇到失败即停止，默认不启用。 |

> `gather_op.py`、`load_mask_dist.py`、`reduction_ops.py`、`load_and_store_scalar_after_reduction.py` 为独立脚本，使用各自的命令行参数（如 `--run`、位置参数 `add/max/min`/`ds`、`--device` 等）。

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
# 基础测试（默认 dtype=f32，shape 使用脚本默认值）
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/binary_op.py add

# 指定 NPU 设备、数据类型与向量长度
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/binary_op.py add --device 1 --dtype f16 --shape 1024

# 全量扫描（多 dtype、多 shape）
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/unary_ops.py exp --sweep --dtypes f16 f32 --sizes 31 64 255 1024

# 批量模式（多个算子打包、多核并发）
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/binary_op.py --batch-run add sub mul max

# `gather_reg_op.py` 执行（需显式 `--run` ；可选 `--mixed`）
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/gather_reg_op.py --run --device 0
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/gather_reg_op.py --run --mixed --device 0

# `gather_op.py` 执行（需显式 `--run` ）
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/gather_op.py --run --device 0

# `histograms_op.py` 执行（需显式 `--run` ）
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/histograms_op.py --run --device 0

# MaskLoadDist 独立脚本（需显式 `--run` ）
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_mask_dist.py ds --run --device 0
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_mask_dist.py us --run --device 0

# 无符号覆盖（并入现有脚本）
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/compare_mask.py vector_vector_gt --dtype u8 --device 0
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/cast_multi.py cast_multi --dtype f32 --device 0
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_store_mask.py load_store_mask --dtype u8 --device 0
```

执行测试后，预期输出（以统一框架样例为例）：

```plain
compile_ok=True host=torch_npu op=<op> dtype=<dtype> shape=<shape> layout=row
kernel.o path=<cache_dir>/<cache_key>/kernel.o
launch_ok=True
outputs equal expected <op>? True
first mismatch=None
```

其中 `compile_ok`/`launch_ok` 表示编译与上板启动是否成功；`outputs equal expected <op>?` 为 `True` 或 `False`，表明 NPU 计算结果与 golden 参考值的精度校验是否通过；`kernel.o path` 为编译产物路径。

 - 浮点类型（`f32`/`f16`/`bf16`）与整型（`i8`/`i16`/`i32`）在精度校验上采用不同的策略：浮点类型使用容差法进行精度判断（绝对误差可由 `--atol` 给定），整型则进行逐元素比对。

## 子测试项说明

以下是针对各子类向量运算测试脚本的具体说明。

### 二元运算

- 文件位置：
  - `binary_op.py`：基础二元运算。
  - `masked_binary.py`：带掩码的多算子二元运算。
  - `multi_binary.py`：多级混合二元运算。

- 功能说明：
  - **`binary_op.py`** 统一二元运算示例，按 `op` 选择算子与搬运模式：

    | Op 模式 | 语义 | 说明 |
    |------|------|------|
    | `add` / `sub` / `mul` / `div` | `z = x + y` / `x - y` / `x * y` / `x / y` | 逐元素四则运算 |
    | `max` / `min` | `z = max(x, y)` / `min(x, y)` | 逐元素最大 / 最小 |
    | `add_unalign` | `z = x + y` | 非对齐 load/store（`UnalignLoadParams` / `UnalignStoreParams`），逐元素精确比对 |
    | `add_brc_b32` | `z = x + y` | 广播加载 + 加（仅 f32） |

  - **`masked_binary.py`** 演示不同掩码下的带掩码计算与选择：

    | 输出 | 计算 | 掩码 / 语义 |
    |------|------|------|
    | `radd` | `add(a, b)` | pattern `H`（前半），未命中通道置 0 |
    | `rsub` | `sub(a, b)` | pattern `Q`（前 1/4），未命中通道置 0 |
    | `rmul` | `mul(a, b)` | pattern `M4`（4 的倍数），未命中通道置 0 |
    | `rdiv` | `div(a, b)` | pattern `H`（前半），未命中通道置 0 |
    | `rsel` | `where(H, a+b, a-b)` | 前半取 `a+b`、后半取 `a-b`，所有 lane 均有定义（`ave.hir.vsel`） |

  - **`multi_binary.py`**（i32, f32）——op `multi_binary`，多级混合计算 + 多种控制流结构：
    - 每块中间量：`inter = b * c / d`；
    - 按 `block_idx` 分流：`block_idx == 0` 写 `x = a + inter - e`，否则写 `t = a * inter + e`；
    - `NUM_CHUNKS = 7` 个分块分别落在 `for` 循环与直线代码等不同控制流位置，覆盖跨 `tla.range` 边界的寄存器值处理。

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `binary_op.py` | i8, i16, i32, f16, f32, bf16 |
    | `masked_binary.py` | i16, i32, f16, f32 |
    | `multi_binary.py` | i32, f32 |

  - `binary_op.py`：`mul`/`div` 不支持 `i8`；`div` 不支持 `bf16`；`add_brc_b32` 仅支持 `f32`；`div` 要求 rhs 非零。
  - `masked_binary.py`：掩码未命中的 lane 被置 0；末块越界由循环携带的 `tla.update_mask` 尾掩码约束。
  - `multi_binary.py`：仅支持 4 字节 dtype（`i32`/`f32`，64 lane）；固定 7 块恰好覆盖 400 元素，不随 shape 泛化。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 基础二元：逐元素加（默认 dtype=f32, shape=400）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/binary_op.py add --dtype f16 --shape 1024

  # 非对齐搬运 / B32 广播变体（均仅支持 f32）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/binary_op.py add_unalign --dtype f32
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/binary_op.py add_brc_b32 --dtype f32

  # 批量多算子：一次打包 add/sub/mul/max 多核并发执行
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/binary_op.py --batch-run add sub mul max

  # 带掩码的多算子二元 / 多级混合二元
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/masked_binary.py masked_binary --dtype f32
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/multi_binary.py multi_binary --dtype f32
  ```

### 一元运算

- 文件位置：
  - `unary_ops.py`：向量一元运算与带掩码一元运算。

- 功能说明：
  - **`unary_ops.py`** 一元运算示例：

    | op | 语义 |
    |------|------|
    | `exp` / `log` / `sqrt` | 指数 / 对数 / 开方（仅 f16/f32） |
    | `abs` / `neg` | 绝对值 / 取负 |
    | `masked_unary` | 一次同时计算 exp/log/sqrt/abs 并施加不同掩码 |
    | `masked_abs` / `masked_neg` | 带掩码的绝对值 / 取负 |

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `unary_ops.py` | i8, i16, i32, f16, f32, bf16（浮点类运算仅 f16/f32，bf16 不支持） |

  - `exp`/`log`/`sqrt` 仅支持 f16/f32（bf16 与整数跳过）。
  - `abs`/`neg` 支持整数与 f16/f32（bf16 跳过）；`masked_unary` 仅 f16/f32；`masked_abs` 仅 i8/i16/i32；`masked_neg` 支持整数与 f16/f32。
  - 输出 4 个（非 `masked_unary` 时其余用 sentinel 补齐）。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 指数运算（浮点类运算仅支持 f16/f32）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/unary_ops.py exp --dtype f32 --device 0

  # 带掩码的一元运算组合：一次计算 exp/log/sqrt/abs（仅 f16/f32）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/unary_ops.py masked_unary --dtype f16 --device 0

  # 带掩码的绝对值（仅整数 i8/i16/i32）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/unary_ops.py masked_abs --dtype i32 --device 0
  ```

### 位运算与移位

- 文件位置：
  - `bitwise_ops.py`：MaskSSA 与 VectorSSA 的位运算。
  - `shift_op.py`：寄存器/标量移位量的左移与右移。

- 功能说明：
  - **`bitwise_ops.py`** 按位运算示例。
  - **`shift_op.py`** 移位运算示例。

    | op | 语义 |
    |------|------|
    | `shift_left` | 寄存器移位量的左移（`tla.shift_left`，reg-reg） |
    | `shift_right` | 寄存器移位量的右移（reg-reg） |
    | `shift_lefts` | 标量移位量的左移（`tla.shift_left(..., int)`，reg-scalar） |
    | `shift_rights` | 标量移位量的右移（reg-scalar） |

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `bitwise_ops.py` | f16, bf16, f32, i32, i16, i8 |
    | `shift_op.py` | i8, i16, i32 |

  - `bitwise_ops.py`：全部列出的 dtype 均支持；`bf16` 下寄存器 `not` 路径退化（`_SUPPORTS_REG_NOT=False`）。
  - `shift_op.py`：源为 signed `i8`/`i16`/`i32`；移位量须非负；右移为算术右移。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 位运算：掩码/寄存器的 not/and/or/xor（共 8 路输出）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/bitwise_ops.py bitwise_ops --dtype i32 --device 0

  # 移位：reg-reg 左移（移位量来自寄存器）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/shift_op.py shift_left --dtype i32 --device 0

  # 移位：reg-scalar 右移（移位量为标量，算术右移）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/shift_op.py shift_rights --dtype i16 --device 0
  ```

### 比较与掩码搬运

- 文件位置：
  - `compare_mask.py`：比较掩码与 `tla.where` 选择。
  - `mask_select_op.py`：MaskSSA 之间的 `tla.where` 选择。
  - `load_store_mask.py`：MaskSSA 的 UB 往返搬运。
  - `load_mask_dist.py`（独立脚本）：`MaskLoadParams` 的 `ds` / `us` 两种模式。

- 功能说明：
  - **`compare_mask.py`**（f32）：

    | op | 语义 |
    |------|------|
    | `vector_vector_lt` / `le` / `gt` / `ge` / `eq` / `ne` | 向量-向量比较后用 `tla.where` 选 lhs/rhs |
    | `vector_scalar_gt` / `ge` | 向量与 0.0 比较 |
    | `masked_vector_vector_lt` | 带 `H` 掩码的向量-向量比较 |
    | `cmp_masked_fused` | 比较 + masked add + 掩码 store 融合 |
    | `static_dynamic_lt` | 静态首 chunk 参考与动态 chunk 混用 |

  - **`load_store_mask.py`**（f32, f16, i8）：op `load_store_mask`，`create_mask(H)` → `store(MaskStoreParams)` → `load(MaskLoadParams)` → 以该掩码做 masked store 的往返。
  - **`load_mask_dist.py`**（独立脚本，位置参数 `ds` / `us`，可选 `--mask-dtype`）：
    - `ds`：`DIST_DS` / `mask_load_ds_b*` → `tla.where`
    - `us`：`DIST_US`/`PACK` / `mask_load_us_b*`；f32 下 byte `i` → lane `4*i`
    - `--mask-dtype i8|i16|i32`：mask UB 寻址类型（b8/b16/b32；IR 亦接受 b64）；首 64 mask 字节布局跨 dtype 相同
      （`make_tensor_like`/GM↔UB 暂无 i64，故 e2e 未覆盖 b64）

  - **`mask_select_op.py`**（f32, f16, i8）：op `mask_select`，使用 `H` 作 selector，在 `Q` 和 `M4` 两个 MaskSSA 之间选择；结果掩码再驱动数据版 `tla.where` 写回 GM，便于观察 predicate bit 结果。

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `compare_mask.py` | f32；u8/u16/u32（仅 `vector_vector_*`） |
    | `mask_select_op.py` | f32, f16, i8 |
    | `load_store_mask.py` | f32, f16, i8, u8 |
    | `load_mask_dist.py ds/us` | f32 伴随 + mask `i8/i16/i32`（IR 另支持 b64） |

  - `compare_mask.py`：f32 全 op；u8/u16/u32 仅 `vector_vector_*`（越过有符号边界）；输入 2、输出 1；支持 batch。
  - `mask_select_op.py`：shape 必须是完整向量寄存器 lane 数的整数倍；分别覆盖 `mask<64/128/256>`。
  - `load_store_mask.py`：shape 随 dtype 变化（f32→64、f16→128、i8/u8→256），输出仅前 `ELE/2` lane 有效；掩码与伴随向量 dtype 一致。
  - `load_mask_dist.py`：必须显式传 `op` 与 `--run`；默认 `--mask-dtype i8`。
  - **字节偏移对齐**（作用于 UB 字节地址 `base + offset*sizeof(T)`；DSL **不**做静态校验，
    不对齐可编译但运行可能 fault）：

    | `MaskLoadDist` | 对齐约束（Byte） | VL=256 时 |
    |----------------|------------------|-----------|
    | `DIST_NORM` | `VL/8` | 32B |
    | `DIST_US` | `VL/16` | 16B |
    | `DIST_DS` | `min(32, VL/4)` | 32B |

    硬件路径为 `plds(uint32_t*)`，故绝对下限亦为 4B；正式契约以表中模式约束为准。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 向量-向量比较（lt）后用 tla.where 选择 lhs/rhs
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/compare_mask.py vector_vector_lt --dtype f32 --device 0
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/compare_mask.py vector_vector_gt --dtype u8 --device 0

  # 比较 + masked add + 掩码 store 的融合
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/compare_mask.py cmp_masked_fused --dtype f32 --device 0

  # MaskSSA 的 UB 存储 / 加载往返（含 unsigned u8）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_store_mask.py load_store_mask --dtype f32 --device 0
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_store_mask.py load_store_mask --dtype u8 --device 0

  # MaskSSA Select：H ? Q : M4
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/mask_select_op.py mask_select --dtype f32 --shape 512 --device 0

  # MaskLoadDist：统一脚本 load_mask_dist.py <ds|us> --run
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_mask_dist.py ds --run --device 0
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_mask_dist.py us --run --device 0
  # b16/b32 mask 寻址类型（首 64 mask 字节与 i8 相同）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_mask_dist.py ds --run --mask-dtype i32 --device 0
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_mask_dist.py us --run --mask-dtype i16 --device 0
    ```

### 类型转换

- 文件位置：
  - `cast_multi.py`：`tla.cast` 多级类型转换。

- 功能说明：**`cast_multi.py`**（f32）——op `cast_multi`，单个内核在 4 个 64-lane chunk 上跑多条 cast 路径并汇总为 f32 输出，覆盖 f16/bf16/i32/i16/i8 与 f32 之间的 2x/4x 宽度转换；另含 `(a+200)` 的 `f16↔u8` unsigned 往返（咬住 signed twin）。

- `CastParams` 模式说明：

  | 字段 | 取值 | 含义 |
  |------|------|------|
  | `reg_slot`（`tla.params.RegSlot`） | `ZERO` / `ONE` | 2x 宽度转换（如 f32→f16、i32↔i16）落到的寄存器位置：`ZERO`=`part_even`（偶位）、`ONE`=`part_odd`（奇位） |
  | | `ZERO` / `ONE` / `TWO` / `THREE` | 4x 宽度转换（如 i32↔i8）的 pack 象限：`ZERO`=pp0、`ONE`=pp1、`TWO`=pp2、`THREE`=pp3 |
  | `sat_mode`（`tla.params.SatMode`） | `NOSAT` / `SAT` / `UNKNOWN` | 溢出行为（AVE `sat`）：不饱和 / 饱和 / 未指定（当前 lowering 按不饱和处理） |
  | `round_mode`（`tla.params.RoundMode`） | `CAST_ROUND` | 就近取整，平局远离 0 |
  | | `CAST_FLOOR` | 向 -∞ 取整 |
  | | `CAST_CEIL` | 向 +∞ 取整 |
  | | `CAST_TRUNC` | 向 0 取整（截断） |

  默认组合为 `CastParams(reg_slot=RegSlot.ZERO, sat_mode=SatMode.NOSAT, round_mode=RoundMode.CAST_ROUND)`。

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `cast_multi.py` | f32 |

  - `cast_multi.py`：输入/输出均为 f32，转换在内核内部进行；固定 256 元素 / 4 chunk；默认 atol 1e-2。
  - 浮点→浮点收窄（f32→f16/bf16）的 `round_mode` 为亚 ULP 舍入；本示例输入可精确表示，故该处舍入不改变结果。
  - 整数舍入（trunc/floor）在浮点→整数转换上可观测；`sat_mode=UNKNOWN` 被放在浮点→浮点转换上以避免影响数值。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 多级类型转换（输入/输出 f32，内核内部转换）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/cast_multi.py cast_multi --dtype f32 --device 0
  ```

### 序列生成

- 文件位置：
  - `arange_op.py`：`tla.arange` 递增/递减序列生成。

- 功能说明：**`arange_op.py`**（i8, i16, i32）

  | op | 语义 |
  |------|------|
  | `increase` | 每个 chunk 生成递增序列 |
  | `decrease` | 每个 chunk 生成递减序列 |

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `arange_op.py` | i8, i16, i32 |

  - `arange_op.py`：无 GM 输入（`input_count=0`）；输出 1 个；支持 batch。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 递增序列
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/arange_op.py increase --dtype i32 --device 0

  # 递减序列
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/arange_op.py decrease --dtype i16 --device 0
  ```

### 数据重排

- 文件位置：
  - `interleave_op.py`：交织/解交织。
  - `pack_op.py`：谓词压缩放置到目标寄存器低/高半区。

- 功能说明：**`interleave_op.py`**（i8, i16, i32, f16, f32, bf16）：

  | op | 语义 |
  |------|------|
  | `interleave` | 两路输入交错为两路输出：`dst0` 的偶、奇位分别取自 `src0`、`src1` 的前半段，`dst1` 的偶、奇位分别取自两者的后半段 |
  | `deinterleave` | 两路输入解交错为两路输出：`dst0` 收集两路输入的偶数位（前半段取自 `src0`、后半段取自 `src1`），`dst1` 收集奇数位 |

- 功能说明：**`pack_op.py`**：

  | part | 语义 |
  |------|------|
  | `low` | MaskSSA predicate 写入目标寄存器低半 |
  | `high` | MaskSSA predicate 写入目标寄存器高半 |

  支持 mask<64>→mask<128>、mask<128>→mask<256>。未选中的半区由 `ppack.z` 指令置零；`LOW/HIGH` 表示目标寄存器半区。

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `interleave_op.py` | i8, i16, i32, f16, f32, bf16 |
    | `pack_op.py` | mask: mask<64>, mask<128> |

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # interleave 模式（两路输入 两路输出）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/interleave_op.py interleave --dtype f32 --device 0

  # deinterleave 模式 （interleave 的逆操作）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/interleave_op.py deinterleave --dtype f32 --device 0

  # mask pack 到目标寄存器高半（mask<128> -> mask<256>）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/pack_op.py --mask-width 128 --part high --device 0
  ```

### 数据搬运

- 文件位置：
  - `load_dintlv_op.py`：双目标交织加载。
  - `load_us_b8_op.py`：b8 上采样加载。
  - `load_store_update.py`：对齐 `PostMode.UPDATE`（`tile.load` / `tile.store`，`add`）。
  - `load_store_block_update.py`：Block `PostMode.UPDATE`（`BlockLoadParams` / `BlockStoreParams`，`add`）。
  - `load_store_unalign.py`：非对齐 PostMode UPDATE，dense `tile_extent` / 每趟自含双指令（`add`；`--tile-extent 1` 为逐元素搬入搬出）。
  - `store_pack.py`：紧凑 PACK 存储。

- 功能说明：
  - **`load_dintlv_op.py`**（f32）：op `dintlv_b32`，按 `2*i*VL` 偏移读 `2*VL`，拆出偶数/奇数两路寄存器。
  - **`load_us_b8_op.py`**（i8）：op `us_b8`，每 loop 读 `VL/2` 个 b8 并上采样为 VL 寄存器。
  - **`load_store_update.py`**（f32）：`add`，`tile.load` / `tile.store` 的 `post_mode=update` 滑动窗口；`--dynamic-extent` 测动态 tile extent。循环 `yield` 的是按指针造好的 tile。
  - **`load_store_block_update.py`**（f32 / f16）：`add`，`Block*Params(post_mode=update, block_stride=1)` 滑动窗口；`repeat_stride` 以 32B DataBlock 计（一整 VL = 8）。
  - **`load_store_unalign.py`**（f32 / f16 / bf16）：`add`，环内仅 `Unalign*Params(post_mode=update)`（dense：禁止显式 `post_update_stride`；每趟自含双指令）；`--offset 1` 起点非 32 字节对齐；`--mode once` 只走一圈 UPDATE。
  - **`store_pack.py`**（输入 i32/i16）：op `store_pack`，i32→i16（`DIST_PACK_B32`）、i16→i8（`DIST_PACK_B16`），抽取低半有效数位做紧凑存储。

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `load_dintlv_op.py` | f32 |
    | `load_us_b8_op.py` | i8 |
    | `load_store_update.py` | f16, bf16, f32 |
    | `load_store_block_update.py` | f16, f32 |
    | `store_pack.py` | i32, i16 |

  - `load_dintlv_op.py`：仅 f32（i32/u32 不支持）；`VECTOR_ELE` 须为 `2*VL` 的倍数；仅写 `LOOPS*VL` lane，其余保留 sentinel。
  - `load_us_b8_op.py`：仅 i8；仅写 `LOOPS*VL` lane，其余保留 sentinel。
  - `load_store_update.py`：`post_mode=update` 要求元素类型为 `f16`/`bf16`/`f32`、rank-1 tile、`DIST_NORM`、`post_update_stride > 0`；`unalign` 的 `--unalign-offset` 须 `N*sizeof(T)%32==0`。
  - `load_store_block_update.py`：`post_mode=update` 要求 origin 窗口（`coord=0`）、`repeat_stride > 0`（DataBlock 单位）、`block_stride` 在 `[0, 65535]`。
  - `store_pack.py`：输入 i32/i16，输出对应 i16/i8；固定 256 元素。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 双目标交织加载（仅 f32）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_dintlv_op.py dintlv_b32 --dtype f32 --device 0

  # b8 元素上采样加载（仅 i8）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_us_b8_op.py us_b8 --dtype i8 --device 0

  # aligned PostMode.UPDATE
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_store_update.py add --shape 400 --device 0

  # Block PostMode.UPDATE
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_store_block_update.py add --shape 400 --device 0

  # 紧凑 PACK 存储（i32->i16，取低半有效数位）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/store_pack.py store_pack --dtype i32 --device 0
  ```

### 归约、压缩与控制流

- 文件位置：
  - `reduction_ops.py`（独立脚本）：向量归约。
  - `squeeze_op.py`：按掩码压缩 lane。
  - `register_control_flow.py`：寄存器值跨循环控制流。

- 功能说明：
  - **`reduction_ops.py`**（f32，独立脚本）：

    | op | 语义 |
    |------|------|
    | `add` / `max` / `min` | 求和 / 最大 / 最小归约 |

    将 128 个 f32 归约为两个 1 元素结果：tile0（前 64）对齐存储、tile1（后 64）非对齐存储（`UnalignStoreParams`）。

  - **`squeeze_op.py`**（f32, f16, i32）：op `squeeze`，按 `M4` 掩码压缩有效 lane 并紧凑写回。
  - **`register_control_flow.py`**（f32）：op `register_carriers`，演示 `VectorSSA`/`MaskSSA` 作为 `scf.for` 的 iter_args/results 跨循环边界携带。

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `reduction_ops.py` | f32 |
    | `squeeze_op.py` | f32, f16, i32 |
    | `register_control_flow.py` | f32 |

  - `reduction_ops.py`：仅 f32；单 op 模式须 `--run`；`--batch-run` 与位置 op 互斥；默认 `--device 0`。
  - `squeeze_op.py`：仅单次 VL（`VECTOR_ELE = VL_ELE = 64`）。
  - `register_control_flow.py`：shape 必须恰好一个物理向量寄存器（f32 下为 64 元素）。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 求和归约（独立脚本，单 op 模式需显式 --run）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/reduction_ops.py add --run --dtype f32 --device 0

  # 批量归约：一次打包 add/max/min 多核并发
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/reduction_ops.py --batch-run --device 0

  # 按 M4 掩码压缩有效 lane 并紧凑写回
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/squeeze_op.py squeeze --dtype f32 --device 0

  # VectorSSA/MaskSSA 跨 scf.for 循环边界携带
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/register_control_flow.py register_carriers --dtype f32 --device 0
  ```

### 标量访问

- 文件位置：
  - `load_and_store_scalar_after_reduction.py`：UB 标量访问。

- 功能说明：**`load_and_store_scalar_after_reduction.py`**（f32）：对 f32 UB 向量归约后，分别在 `tla.vec.func` 内与 `tla.vector` 内直接做 UB 标量 `load`/`store`。

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `load_and_store_scalar_after_reduction.py` | f32 |

  - `load_and_store_scalar_after_reduction.py`：dtype 固定 f32，无位置 op、无 `--run`，唯一参数 `--device`；固定写入索引 7（`vec.func` 内）与 8（`vector` 内）。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # UB 标量 load/store（无位置 op、无 --run，直接执行）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/load_and_store_scalar_after_reduction.py --device 0
  ```

### 离散与聚合

- 文件位置：
  - `gather_op.py`：`tla.gather` 按索引收集（UB→Reg）。
  - `gather_reg_op.py`：`tla.gather` Reg→Reg（`vselr`）。
  - `histograms_op.py`：Reg 直方图（BIN0/BIN1）。

- 功能说明：
  - **`gather_op.py`**（f32，独立脚本）——`tla.gather` 依据 int32 索引向量从源向量收集元素并写回目标向量；源向量、索引向量与目标向量均为 64 lane。
  - **`gather_reg_op.py`**（f16，独立脚本）——`tla.gather` Reg→Reg 置换，128-lane f16 + i16 索引位型。
  - **`histograms_op.py`**（i8→i16，独立脚本）——`tla.histograms` 全组合：FREQUENCY/ACCUMULATE*BIN0/BIN1；`dst` 必传（`tla.full(0)` 清零）；另测 FREQUENCY 跨调用累加与 ACCUMULATE 全零（BIN1=256）。

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `gather_op.py` | f32 |
    | `gather_reg_op.py` | f16 + i16 索引 |
    | `histograms_op.py` | i8 输入（u8 位型）/ i16 输出 |

  - `gather_op.py`：仅 f32；必须显式传 `--run`，否则退出；索引为 i32；`VECTOR_ELE = 64`；默认 `--device 0`；精度校验 `atol=1e-4`。
  - `gather_reg_op.py`：必须 `--run`；固定 `VL_ELE=128`；默认 `--device 0`。
  - `histograms_op.py`：必须 `--run`；`--mode frequency|accumulate|frequency_into|all`（默认 `all`）；固定 `SRC_ELE=256`、`DST_ELE=128`；默认 `--device 0`。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 按索引收集（独立脚本，必须显式 --run）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/gather_op.py --run --dtype f32 --device 0

  # Reg→Reg Gather 上板
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/gather_reg_op.py --run --device 0

  # Reg 直方图上板（ACCUMULATE + FREQUENCY into）
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/histograms_op.py --run --device 0
  ```

### 广播（独立脚本）

- 文件位置：
  - `full_op.py`：`tla.full` 的标量填充或单通道广播。

- 功能说明：**`full_op.py`** ——演示 `tla.full` 的两种取值形式以及可选 lane 掩码，一次 kernel 产出 4 路广播结果：

  | 输出 | 计算 | 语义 |
  |------|------|------|
  | `out_vec_nomask` | `full(reduced, data_type)` | 单 lane `fragment`（`tla.reduce` 后的结果）广播到所有 lane |
  | `out_scl_nomask` | `full(SCALAR_FILL, data_type)` | Python 标量填充到所有 lane |
  | `out_vec_mask` | `full(reduced, data_type, mask=mask)` | 仅 lane < `MASK_LANES` 写归约值 |
  | `out_scl_mask` | `full(SCALAR_FILL, data_type, mask=mask)` | 仅 lane < `MASK_LANES` 写标量 |

  其中 `fragment` 形式是：`tla.reduce` 的结果位于向量寄存器的一个 lane，SIMD 后端没有把 lane 读回标量寄存器的指令；直接广播 fragment 无需经过标量寄存器即可完成数据相关的填充。

- 约束说明：
  - 支持的数据类型：

    | 文件 | 支持的数据类型 |
    |------|---------------|
    | `full_op.py` | f32 |

  - `full_op.py`：固定 `VECTOR_ELE = 64`（恰好一个物理向量寄存器）、`MASK_LANES = 40`、`SCALAR_FILL = 3.0`；输入 1 个（`x`），输出 4 个（`vn` / `sn` / `vm` / `sm`）。
  - 掩码未命中的 lane（`vm[40:]` / `sm[40:]`）不写入，保持初始设定值 `-999.0`，故 golden 校验只比较前 `MASK_LANES` 个元素。

- 执行示例：

  ```bash
  # 在仓库根目录下执行
  cd /path/to/catlass
  # 标量填充 / 单 lane fragment 广播，含带掩码变体
  python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/full_op.py --device 0
  ```
