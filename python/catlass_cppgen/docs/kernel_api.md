# Kernel 基类 API (Python)

本节记录了 CATLASS 中 kernel 基类及其代码生成接口的 Python API，接口定义见 [`kernel_base.py`](../catlass_cppgen/kernel/kernel_base.py)。

---

## `KernelBase` 类

**该类是所有 kernel 定义以及代码生成的基类。**

### 主要方法

- `get_default_tile_shape(self) -> Tuple[GemmShape, GemmShape]`
  *抽象方法。* 返回 kernel 的默认 L1 和 L0 tile 形状。

- `get_workspace_size(self) -> int`
  返回 kernel 所需的工作区大小（默认 0）。

- `need_workspace(self) -> bool`
  是否需要外部工作区（默认为 `False`）。

- `get_core_num(self) -> int`
  返回 kernel 使用的核心数（默认 0）。

- `get_default_dispatch_policy_list(self) -> List[DispatchPolicy]`
  获取默认的 dispatch_policy 列表。子类可以重写此方法以定义自己的默认 dispatch_policy 列表。如果子类不重写，默认返回空列表。

#### 调优接口

- `set_l1_tile_shape(self, l1_tile_shape: GemmShape)`
  设置 L1 tile 形状。

- `set_l0_tile_shape(self, l0_tile_shape: GemmShape)`
  设置 L0 tile 形状。

- `set_dispatch_policy(self, dispatch_policy: Union[DispatchPolicy, List[DispatchPolicy]])`
  设置 dispatch policy。可以传入单个 policy 或 policy 列表。如果传入单个 policy，会自动转换为包含该 policy 的列表。

- `get_dispatch_policy(self) -> List[DispatchPolicy]`
  获取 dispatch policy 列表。每个算子类（如 `BasicMatmulKernel`）可以定义自己的默认 dispatch_policy 列表，通过重写 `get_default_dispatch_policy_list()` 方法。

  如果未显式设置，会自动使用算子类定义的默认列表。列表的第一个元素 `[0]` 是默认策略。
  目前默认返回只包含一个 policy 的列表，但接口设计支持未来扩展为多个 policy。

- `set_block_scheduler(self, block_scheduler)`
  设置 block scheduler（保留参数）。

- `tune(self, l1_tile_shape: Optional[GemmShape] = None, l0_tile_shape: Optional[GemmShape] = None, dispatch_policy: Optional[Union[DispatchPolicy, List[DispatchPolicy]]] = None, block_scheduler: Optional[BlockScheduler] = None)`
  一次性调优 kernel 的 tile 形状和调度策略。所有参数均为可选，如果为 `None` 则使用当前值或默认值。

#### 特性支持查询

- `is_support(self, feature: str) -> bool`
  查询该 kernel 是否支持某特性。

- `is_support_evg(self) -> bool`
  查询该 kernel 是否支持 `evg` 特性。

- `is_support_hf32(self) -> bool`
  查询该 kernel 是否支持 `hf32` 特性。

#### 代码生成与渲染

- `get_render_params(self) -> Dict[str, Any]`
  *抽象方法。* 返回用于模板渲染的参数字典。

- `gen_includes(self) -> str`
  生成 C++ 的 `#include` 文件头，根据 kernel 需求自动拉齐。

- `gen_kernel_name(self) -> str`
  根据参数生成 kernel 的函数名。

- `gen_params_device(self, def_mode: bool = False) -> str`
  生成 device 端函数参数列表，`def_mode=True` 用于函数定义，否则用于调用。

- `gen_kernel_template(self) -> str`
  使用渲染参数填充 kernel 模板，生成核函数"核心代码块"。

- `gen_layout_template(self) -> str`
  生成 layout 相关信息代码，包括 M, K, N 的定义和 layout tag 的创建。

- `codegen(self) -> str`
  生成完整 C++ kernel，包括头文件、函数签名、参数、代码块与 kernel 启动。

---

## 示例：生成 Kernel 代码

### 简单用法（GEMM kernel 示例）

```python
from catlass_cppgen.kernel.gemm.basic_matmul import BasicMatmulKernel
from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.arch.arch import Arch

# 假设 BasicMatmulKernel 是 GemmKernelBase 的子类，已实现必备方法
gemm_kernel = BasicMatmulKernel(
    element_accumulator=DataType.FLOAT,
    element_A=DataType.FLOAT,
    element_B=DataType.FLOAT,
    element_C=DataType.FLOAT,
    element_Bias=DataType.FLOAT,
    layout_A=RowMajor((128, 256)),
    layout_B=RowMajor((256, 384)),
    layout_Bias=RowMajor((128, 384)),
    arch_tag=Arch.Ascend950
)

# 优化 tile 形状
gemm_kernel.tune(
    l1_tile_shape=GemmShape(128, 256, 64),
    l0_tile_shape=GemmShape(128, 256, 64)
)

# 生成头文件
print("Includes:")
print(gemm_kernel.gen_includes())

# 生成核函数参数（定义模式）
print("Params (def_mode=True):")
print(gemm_kernel.gen_params_device(def_mode=True))

# 生成核函数参数（调用模式）
print("Params (def_mode=False):")
print(gemm_kernel.gen_params_device(def_mode=False))

# 生成核函数模板
print("Kernel Template:")
print(gemm_kernel.gen_kernel_template())

# 生成 layout 模板
print("Layout Template:")
print(gemm_kernel.gen_layout_template())
```

### 所有 Kernel 类型的使用示例

#### BasicMatmulKernel（基础矩阵乘法）

适用于标准的 2D 矩阵乘法运算，支持可选的 Bias 参数。

```python
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm.dispatch_policy import MmadPingpong

def test_basic_matmul_kernel():
    # 使用 OpTensor.from_shape_stride 创建输入（避免实例化实际 tensor 数据）
    a = OpTensor.from_shape_stride(
        shape=(128, 256),
        stride=(256, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    b = OpTensor.from_shape_stride(
        shape=(256, 384),
        stride=(384, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    gemm_plan = Gemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor)
    kernels = gemm_plan.get_kernels(A=a, B=b)
    basic_kernel = kernels[0]  # BasicMatmulKernel

    basic_kernel.tune(
        GemmShape(128, 256, 64),
        GemmShape(128, 256, 64),
        dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950)
    )

    # 生成头文件
    print("Includes:")
    print(basic_kernel.gen_includes())

    # 生成核函数参数（定义模式）
    print("Params (def_mode=True):")
    print(basic_kernel.gen_params_device(def_mode=True))

    # 生成核函数模板
    print("Kernel Template:")
    print(basic_kernel.gen_kernel_template())

    # 生成 layout 模板
    print("Layout Template:")
    print(basic_kernel.gen_layout_template())
```

#### BatchedMatmulKernel（批处理矩阵乘法）

适用于批处理场景，输入张量 A 和 B 为 3 维（batchCount, M, K）和（batchCount, K, N）。

```python
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm.dispatch_policy import MmadPingpong

def test_batched_matmul_kernel():
    # 批处理：batchCount=8, M=128, K=256, N=384
    # 使用 OpTensor.from_shape_stride 创建输入（避免实例化实际 tensor 数据）
    a = OpTensor.from_shape_stride(
        shape=(8, 128, 256),  # (batchCount, M, K)
        stride=(32768, 256, 1),  # batched RowMajor stride: (m*n, n, 1)
        dtype=DataType.FLOAT
    )
    b = OpTensor.from_shape_stride(
        shape=(8, 256, 384),  # (batchCount, K, N)
        stride=(98304, 384, 1),  # batched RowMajor stride: (k*n, n, 1)
        dtype=DataType.FLOAT
    )
    gemm_plan = Gemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor)
    kernels = gemm_plan.get_kernels(A=a, B=b)
    batched_kernel = kernels[0]  # BatchedMatmulKernel

    batched_kernel.tune(
        GemmShape(128, 256, 64),
        GemmShape(128, 256, 64),
        dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950, enable_unit_flag=True)
    )

    # 生成头文件
    print("Includes:")
    print(batched_kernel.gen_includes())

    # 生成核函数参数（定义模式）
    print("Params (def_mode=True):")
    print(batched_kernel.gen_params_device(def_mode=True))

    # 生成核函数模板
    print("Kernel Template:")
    print(batched_kernel.gen_kernel_template())

    # 生成 layout 模板（包含 stride 信息）
    print("Layout Template:")
    print(batched_kernel.gen_layout_template())
```

#### StreamkMatmulKernel（StreamK 矩阵乘法）

适用于大规模矩阵乘法，使用 StreamK 调度策略优化性能。

```python
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm.dispatch_policy import MmadPingpong

def test_streamk_matmul_kernel():
    # 使用 OpTensor.from_shape_stride 创建输入（避免实例化实际 tensor 数据）
    a = OpTensor.from_shape_stride(
        shape=(128, 256),
        stride=(256, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    b = OpTensor.from_shape_stride(
        shape=(256, 384),
        stride=(384, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    gemm_plan = Gemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor)
    kernels = gemm_plan.get_kernels(A=a, B=b)
    streamk_kernel = kernels[2]  # StreamkMatmulKernel（索引可能因实现而异）

    # StreamK kernel 默认使用较大的 tile shape，并指定 dispatch_policy
    streamk_kernel.tune(
        GemmShape(256, 256, 128),
        GemmShape(256, 256, 32),
        dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950, enable_unit_flag=True)
    )

    # 生成头文件
    print("Includes:")
    print(streamk_kernel.gen_includes())

    # 生成核函数参数（定义模式，包含 aicCoreNum 参数）
    print("Params (def_mode=True):")
    print(streamk_kernel.gen_params_device(def_mode=True))

    # 生成核函数模板
    print("Kernel Template:")
    print(streamk_kernel.gen_kernel_template())

    # 生成 layout 模板
    print("Layout Template:")
    print(streamk_kernel.gen_layout_template())
```

#### MultiCoreSplitkMatmulKernel（多核 SplitK 矩阵乘法）

适用于需要多核并行计算的大规模矩阵乘法，通过 SplitK 策略提高并行度。

```python
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm.dispatch_policy import MmadPingpong

def test_multi_core_splitk_matmul_kernel():
    # 使用 OpTensor.from_shape_stride 创建输入（避免实例化实际 tensor 数据）
    a = OpTensor.from_shape_stride(
        shape=(128, 256),
        stride=(256, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    b = OpTensor.from_shape_stride(
        shape=(256, 384),
        stride=(384, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    gemm_plan = Gemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor)
    kernels = gemm_plan.get_kernels(A=a, B=b)
    splitk_kernel = kernels[1]  # MultiCoreSplitkMatmulKernel（索引可能因实现而异）

    # SplitK kernel 默认使用较大的 tile shape，并指定 dispatch_policy
    splitk_kernel.tune(
        GemmShape(256, 256, 128),
        GemmShape(256, 256, 32),
        dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950, enable_unit_flag=True)
    )

    # 生成头文件
    print("Includes:")
    print(splitk_kernel.gen_includes())

    # 生成核函数参数（定义模式，包含 aicCoreNum 参数）
    print("Params (def_mode=True):")
    print(splitk_kernel.gen_params_device(def_mode=True))

    # 生成核函数模板
    print("Kernel Template:")
    print(splitk_kernel.gen_kernel_template())

    # 生成 layout 模板
    print("Layout Template:")
    print(splitk_kernel.gen_layout_template())
```

#### TailMultiCoreSplitkMatmulKernel（尾部多核 SplitK 矩阵乘法）

适用于处理 SplitK 策略中的尾部计算，与 MultiCoreSplitkMatmulKernel 配合使用。

```python
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm.dispatch_policy import MmadPingpong

def test_tail_multi_core_splitk_matmul_kernel():
    # 使用 OpTensor.from_shape_stride 创建输入（避免实例化实际 tensor 数据）
    a = OpTensor.from_shape_stride(
        shape=(128, 256),
        stride=(256, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    b = OpTensor.from_shape_stride(
        shape=(256, 384),
        stride=(384, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    gemm_plan = Gemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor)
    kernels = gemm_plan.get_kernels(A=a, B=b)
    tail_splitk_kernel = kernels[3]  # TailMultiCoreSplitkMatmulKernel（索引可能因实现而异）

    # Tail SplitK kernel 默认使用较大的 tile shape，并指定 dispatch_policy
    tail_splitk_kernel.tune(
        GemmShape(256, 256, 128),
        GemmShape(256, 256, 32),
        dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950, enable_unit_flag=True)
    )

    # 生成头文件
    print("Includes:")
    print(tail_splitk_kernel.gen_includes())

    # 生成核函数参数（定义模式，包含 aicCoreNum 参数）
    print("Params (def_mode=True):")
    print(tail_splitk_kernel.gen_params_device(def_mode=True))

    # 生成核函数模板
    print("Kernel Template:")
    print(tail_splitk_kernel.gen_kernel_template())

    # 生成 layout 模板
    print("Layout Template:")
    print(tail_splitk_kernel.gen_layout_template())
```

#### GroupedMatmulSliceMKernel（分组矩阵乘法 - Slice M）

适用于分组 GEMM 场景，多个不同大小的矩阵乘法问题，通过 Slice M 策略优化。

```python
from catlass_cppgen.op.group_gemm import GroupGemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm.dispatch_policy import MmadPingpong

def test_grouped_matmul_slice_m_kernel():
    # 分组 GEMM：每个问题的 A 和 B 矩阵维度相同
    # 使用 OpTensor.from_shape_stride 创建输入（避免实例化实际 tensor 数据）
    from catlass_cppgen.catlass.layout.layout import VectorLayout

    a = OpTensor.from_shape_stride(
        shape=(128, 256),  # 单个问题的 A 矩阵
        stride=(256, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    b = OpTensor.from_shape_stride(
        shape=(256, 384),  # 单个问题的 B 矩阵
        stride=(384, 1),   # RowMajor stride: (n, 1)
        dtype=DataType.FLOAT
    )
    # 创建 groupList OpTensor（一维，int64_t 类型）
    # groupList 的长度即为 problemCount（这里是 4）
    groupList = OpTensor(
        dtype=DataType.INT64,
        layout=VectorLayout(4),  # 4 个 group
        shape=(4,)
    )

    group_gemm_plan = GroupGemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor)
    kernels = group_gemm_plan.get_kernels(A=a, B=b, groupList=groupList)
    grouped_kernel = kernels[0]  # GroupedMatmulSliceMKernel

    # Grouped kernel 默认使用较大的 tile shape，并指定 dispatch_policy
    grouped_kernel.tune(
        GemmShape(256, 256, 256),
        GemmShape(256, 256, 64),
        dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950, enable_unit_flag=True)
    )

    # 生成头文件
    print("Includes:")
    print(grouped_kernel.gen_includes())

    # 生成核函数参数（定义模式，包含 problemCount 和 deviceGroupList 参数）
    print("Params (def_mode=True):")
    print(grouped_kernel.gen_params_device(def_mode=True))

    # 生成核函数模板
    print("Kernel Template:")
    print(grouped_kernel.gen_kernel_template())

    # 生成 layout 模板
    print("Layout Template:")
    print(grouped_kernel.gen_layout_template())
```

---

## 说明

- 如需拓展 kernel 代码生成，继承 `KernelBase` 并实现 `get_render_params` 和 `get_default_tile_shape`。
- 子类需设置模板元字段（如 `_INCLUDES`, `_PARAMS_DEVICE`, `_KERNEL_TEMPLATE`）以适配专用 kernel。
- `codegen()` 方法整合所有渲染环节，一键输出最终 C++ kernel 源码。

更多细节与二次开发见 [`kernel_base.py`](../catlass_cppgen/kernel/kernel_base.py)。
