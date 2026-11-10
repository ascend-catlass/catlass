# 外部 Ascend C 算子端到端示例

本目录下样例演示如何使用 `@tla.extern` 将用户提供的 Ascend C 函数嵌入 **CATLASS Python** Kernel，包括单 Kernel 调用多个外部算子，以及同一个外部符号同时被 AIC 和 AIV 调用。

## VecAdd 功能说明

目录包含以下示例：

| 文件 | 概述 |
|------|------|
| [**`extern_dual_core.py`**](extern_dual_core.py) | 单份 Ascend C 源码、单个 extern op 同时在 AIC 与 AIV 区域调用。 |
| [**`extern_vecadd.py`**](extern_vecadd.py) | 外部 Ascend C 函数与 TLA DSL 混合编程示例，包含外部函数源码、ABI 声明和 VecAdd Kernel。 |
| [**`extern_multi_ops.py`**](extern_multi_ops.py) | 两个独立外部 Ascend C 算子组成 GM→UB→GM round-trip。 |
| [**`extern_custom_include.py`**](extern_custom_include.py) | 使用 `include_dirs` 引用用户头文件，并将头文件常量写回 Host 验证。 |
| [**`extern_simd_vf_add.py`**](extern_simd_vf_add.py) | 在 SIMD VF 内向 extern 传入 UB 指针，由外部 Reg API 完成加法。 |
| [**`extern_simd_reg_add.py`**](extern_simd_reg_add.py) | 在 SIMD VF 内按值传入两个 Float32 寄存器，将返回的寄存器交给 DSL store。 |
| [**`extern_simd_reg_types.py`**](extern_simd_reg_types.py) | 验证 FP16、BF16、FP32 和 8/16/32/64 位整数寄存器的传参、返回及跨类型返回，对全部 256 字节按位取反。 |

向量加法示例实现两个一维向量的逐元素加法，计算公式为：

$$
\begin{aligned}
C &= A + B
\end{aligned}
$$

`extern_vecadd.py` 仅使用外部 Ascend C 函数 `tla_user_gm_to_ub_f32` 替换输入 A、B 的 GM（Global Memory）到 UB（Unified Buffer）数据搬运，向量加法和结果写回仍由 TLA DSL 完成。整体流程如下：

1. 通过 `@tla.extern` 声明外部函数的 C ABI，并以内联字符串形式提供 Ascend C 源码；
2. 在 `tla.vector()` 区域内调用外部函数，将 A、B 从 GM 搬运到 UB；
3. 使用 TLA DSL 的 `load`、`tla.add` 和 `store` 完成分块向量加法；
4. 使用 `tla.copy` 将结果从 UB 写回 GM。

## AIC/AIV 共享外部函数

`extern_dual_core.py` 只提供一份 `OP_SOURCE_CODES` 和一个 `tla_user_store_i32` 外部函数声明。TLA Kernel 在 `tla.cube()` 与 `tla.vector()` 区域中复用这个声明。输出包含三个 64-byte cache line：AIC 写入索引 0，两个 AIV sub-block 分别写入索引 16、32，使三个执行单元写入不同的 cache line。对应位置的期望值为 `[101, 202, 202]`，其余元素为 0。

## 单 Kernel 多外部算子

`extern_multi_ops.py` 的流程为：

1. 使用两份独立 source 声明 `tla_multi_gm_to_ub_f32` 和 `tla_multi_ub_to_gm_f32`；
2. 在同一个 `tla.vector()` 区域依次调用两个外部算子；
3. 使用 MTE2→MTE3 flag 保证 GM→UB 完成后才开始 UB→GM；
4. 编译时分别生成 `extern.0.aiv.c310.bc`、`extern.1.aiv.c310.bc`，并全部加入 HIVMC 链接输入。

## 约束说明

- VecAdd 和 multi-op 样例的向量长度固定为 256，输入和输出数据类型固定为 `float32`，启动核数固定为 1，编译目标为 `--npu-arch 3510`。
- `@tla.extern` 的 `source` 必须是非空的 Ascend C 源码字符串；`name` 必须是合法的 C 标识符，省略时使用被装饰的 Python 函数名。
- 相同 symbol 可以为不同 Kernel 分别声明；同一 Kernel 内只能使用其中一个声明对象，相同声明可以调用多次。
- 同一 Kernel 内，相同 source 只能使用同一组规范化、有序的 `include_dirs`；不同 Kernel 可以分别配置。
- `include_dirs` 可按顺序指定用户头文件搜索目录；相对路径按 extern 声明文件所在目录解析。
- 外部函数声明只支持位置固定、无默认值的参数。参数类型支持 `tla.Pointer[dtype, address_space]`、具体的 TLA 数值类型（如 `tla.Int32`）和 `tla.VectorSSA[dtype]`。返回类型为 `None` 或一个 `tla.VectorSSA[dtype]`。
- 寄存器 dtype 支持 `Float16`、`BFloat16`、`Float32` 和 `Int8/16/32/64`、`UInt8/16/32/64`，有效 lane 数必须为 `256 / sizeof(dtype)`。每个参数和返回值独立声明 dtype；外部实现须填满返回寄存器。Int64/UInt64 寄存器可由 extern 返回，现有 tensor API 不支持直接 load。
- 指针参数传入显式指针（例如 `tensor.ptr`）。参数个数、数据类型和指针地址空间须与声明匹配。
- 寄存器参数和结果在同一 SIMD VF 内使用；该 VF 必须有显式 tensor 访存或 extern 指针参数。外部入口使用对应的 `vector_half`、`vector_bfloat16_t`、`vector_float` 或 `vector_int8_t` / `vector_int16_t` / `vector_int32_t` / `vector_int64_t`、`vector_uint8_t` / `vector_uint16_t` / `vector_uint32_t` / `vector_uint64_t` 按值传参和返回，并标注 `always_inline`。整数 signedness 在 TLA 类型检查时保留，LLVM 向量 ABI 使用 signless 整数。
- 外部函数必须在一个 `tla.vector()` 或 `tla.cube()` 区域内调用，可以在 `tla.vec.func(mode="simd")` 中调用，不可在simt vf中调用。
- 单个 Kernel 可以调用多个外部函数；相同 source 对每个实际使用的 core target 只编译一次，不同 source 按首次调用顺序编号并分别编译。
- 外部调用当前需要显式编写流水同步，不能与 `@tla.kernel(auto_sync="v0")` 组合使用。样例通过 `flag` 的 `set` / `wait` 保证外部 GM→UB 搬运完成后，再进入后续 VECTOR 或 MTE3 流水。

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../python/tla_dsl/docs/zh/quick_start.md)完成部署。

### 命令行参数

```text
extern_vecadd.py [-h] [--device DEVICE]
extern_dual_core.py [-h] [--device DEVICE]
extern_multi_ops.py [-h] [--device DEVICE]
extern_custom_include.py [-h] [--device DEVICE]
extern_simd_vf_add.py [-h] [--device DEVICE]
extern_simd_reg_add.py [-h] [--device DEVICE]
extern_simd_reg_types.py [-h] [--device DEVICE] [--dtype {all,f16,bf16,f32,i8,u8,i16,u16,i32,u32,i64,u64}]
```

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
# 使用 0 号 NPU 执行
python examples/python/tla_dsl/dsl_tutorials/extern_op/extern_vecadd.py --device 0

# 同一个 extern op 分别在 AIC 和 AIV 中调用
python examples/python/tla_dsl/dsl_tutorials/extern_op/extern_dual_core.py --device 0

# 执行 GM→UB→GM round-trip
python examples/python/tla_dsl/dsl_tutorials/extern_op/extern_multi_ops.py --device 0

# 执行 extern op，验证从用户头文件读取并写回的值
python examples/python/tla_dsl/dsl_tutorials/extern_op/extern_custom_include.py --device 0

# 在 SIMD VF 内按值传入寄存器，验证外部 Reg API 的计算结果
python examples/python/tla_dsl/dsl_tutorials/extern_op/extern_simd_reg_add.py --device 0

# 检查全部 11 种寄存器类型，也可用 --dtype bf16 单独检查一种类型
python examples/python/tla_dsl/dsl_tutorials/extern_op/extern_simd_reg_types.py --device 0

# 忽略进程内缓存和磁盘缓存，强制重新编译后执行
CATLASS_DSL_FORCE_RECOMPILE=1 \
  python examples/python/tla_dsl/dsl_tutorials/extern_op/extern_vecadd.py --device 0
```

执行测试后，预期输出：

```plain
passed; kernel=<cache_dir>/<cache_key>/kernel.o
```

执行成功后输出 `passed` 和编译产物路径，失败则抛出异常。`cache_dir` 是编译缓存目录，`cache_key` 是编译缓存的哈希值。

---

## 特性介绍

### 声明外部 Ascend C 函数

外部函数源码保存在 `OP_SOURCE_CODES` 字符串中，并使用 `extern "C"` 导出与 `@tla.extern` 声明一致的符号。示例中的 Ascend C 函数通过 `AscendC::DataCopy` 完成 GM 到 UB 的数据搬运：

```cpp
extern "C" {

__aicore__ __attribute__((always_inline)) void tla_user_gm_to_ub_f32(
    uint64_t src_gm_addr, uint64_t dst_ub_addr, int32_t count) {
  AscendC::GlobalTensor<float> src;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src_gm_addr),
                      static_cast<uint32_t>(count));
  AscendC::LocalTensor<float> dst(AscendC::TPosition::VECCALC,
                                  static_cast<uint32_t>(dst_ub_addr),
                                  static_cast<uint32_t>(count));
  AscendC::DataCopy(dst, src, static_cast<uint32_t>(count));
}

} // extern "C"
```

`@tla.extern` 根据 Python 函数注解描述 C ABI。声明函数的 Python 函数体不会执行：

```python
@tla.extern(
    name="tla_user_gm_to_ub_f32",
    source=OP_SOURCE_CODES,
)
def tla_user_gm_to_ub_f32(
    gm_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
    ub_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    ele_num: tla.Int32,
) -> None: ...
```

### 在 TLA Kernel 中调用外部函数

调用被 `@tla.extern` 装饰的函数会在 TLA IR 中生成 `tla.call_extern`。本样例在 AIV 区域内传入 GM/UB 指针和元素个数，连续完成两路输入搬运：

```python
with tla.vector():
    tla_user_gm_to_ub_f32(gm_a.ptr, ub_ptr_a, TILE_ELE)
    tla_user_gm_to_ub_f32(gm_b.ptr, ub_ptr_b, TILE_ELE)
    tla.set_flag(ub_loaded)
    tla.wait_flag(ub_loaded)

    with tla.vec.func(mode="simd"):
        # 使用 TLA DSL 完成分块 load、add 和 store
        # ...
```

`tla.compile()` 会根据外部函数的实际调用区域选择目标。本样例仅在 AIV 区域调用，因此内联源码会由 Ascend C 编译器编译为 `extern.0.aiv.c310.bc`，再加入 `hivmc-a5 --link-aicore-bitcode` 的链接输入。不同 source 按首次调用顺序编号；外部源码内容和编译器信息也会参与 Kernel 缓存键计算，修改 source 后会生成新的缓存项。

### 在 AIC 和 AIV 中调用同一个符号

`extern_dual_core.py` 的两个区域在 Python/TLA 层复用同一个 extern 声明：

```python
with tla.cube():
    tla_user_store_i32(result.ptr, 0, AIC_VALUE)

with tla.vector():
    index = (1 + tla.arch.sub_block_idx()) * ELEMENTS_PER_CACHE_LINE
    tla_user_store_i32(result.ptr, index, AIV_VALUE)
```

Lower Extern Call 阶段保留原始符号 `tla_user_store_i32`。由于该符号同时从 AIC 和 AIV 区域调用，其声明会被标记为 `AIC_OR_AIV`，拆分后的两个 mixed kernel 入口仍然调用同一个符号。CCEC 分别使用 AIC 和 AIV target 编译同一份 `extern.0.cpp`，生成 `extern.0.aic.c310.bc` 和 `extern.0.aiv.c310.bc`，再由 HIVMC 链接到对应的 mixed kernel 入口。
