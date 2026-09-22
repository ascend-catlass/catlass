# `OpTensor` 基础文档 (Python)

本文档为 `catlass_cppgen` 中 `OpTensor` 的使用说明，介绍其创建方法、基础特性（如数据类型，布局分布）等。

---

## 目录

- [OpTensor 构造方法](#optensor-构造方法)
  - [方式1: 直接构造函数](#方式1-直接构造函数)
  - [方式2: from_shape_stride 类方法](#方式2-from_shape_stride-类方法)
  - [方式3: from_tensor 类方法](#方式3-from_tensor-类方法)
- [Operation 支持的输入类型](#operation-支持的输入类型)
- [数据类型支持](#数据类型支持)
- [布局类型支持](#布局类型支持)
- [使用示例](#使用示例)

---

## OpTensor 构造方法

`OpTensor` 是操作（Operation）的输入输出 tensor 的抽象，提供了统一的 tensor 表示。它可以从多种方式创建，支持避免实例化实际 tensor 数据，从而提高代码生成效率。

### 方式1: 直接构造函数

**方法签名：**

```python
OpTensor(
    dtype: DataType,
    layout: Layout,
    shape: Optional[tuple[int, ...]] = None,
    data_ptr: Optional[ctypes.c_void_p] = None
)
```

**参数说明：**

- `dtype`: tensor 的数据类型（`DataType` 枚举）
- `layout`: tensor 的布局（`Layout` 对象，如 `RowMajor`、`ColumnMajor`）
- `shape`: 可选的完整形状。如果不提供，则使用 `layout.shape`。对于 batched tensor，需要显式指定包含 batch 维度的完整 shape
- `data_ptr`: 可选的数据指针（`ctypes.c_void_p`），用于实际数据访问

**示例：**

```python
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor, ColumnMajor
import ctypes

# 1a. 基本创建（只指定 dtype 和 layout）
a_op = OpTensor(dtype=DataType.FLOAT, layout=RowMajor((128, 256)))
b_op = OpTensor(dtype=DataType.FLOAT, layout=RowMajor((256, 384)))
print(f"A: shape={a_op.shape}, dtype={a_op.dtype}")
# 输出: A: shape=(128, 256), dtype=DataType.FLOAT

# 1b. 指定 shape（用于 batched tensor）
a_op_batched = OpTensor(
    dtype=DataType.FLOAT,
    layout=RowMajor((128, 256)),  # 内层矩阵的 layout
    shape=(8, 128, 256)  # 完整的 shape，包含 batch 维度
)
print(f"A (batched): shape={a_op_batched.shape}")
# 输出: A (batched): shape=(8, 128, 256)

# 1c. 指定 data_ptr（用于实际数据指针）
data_ptr = ctypes.c_void_p(0x12345678)
a_op_with_ptr = OpTensor(
    dtype=DataType.FLOAT,
    layout=RowMajor((128, 256)),
    data_ptr=data_ptr
)
print(f"A (with data_ptr): data_ptr={a_op_with_ptr.data_ptr}")

# 1d. 使用 ColumnMajor layout
a_op_col = OpTensor(dtype=DataType.FLOAT, layout=ColumnMajor((128, 256)))
print(f"A (ColumnMajor): stride={a_op_col.stride}")
# 输出: A (ColumnMajor): stride=(1, 128)

# 1e. 使用不同的数据类型
a_op_fp16 = OpTensor(dtype=DataType.FLOAT16, layout=RowMajor((128, 256)))
print(f"A (FLOAT16): dtype={a_op_fp16.dtype}")
```

---

### 方式2: from_shape_stride 类方法

**方法签名：**

```python
@classmethod
OpTensor.from_shape_stride(
    cls,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: DataType
) -> "OpTensor"
```

**参数说明：**

- `shape`: tensor 的形状（可以是 2D 或 3D，支持 batched）
- `stride`: tensor 的步长
- `dtype`: tensor 的数据类型

**说明：**

- 该方法会自动从 `shape` 和 `stride` 推断 `Layout` 类型
- 对于 batched tensor（3D），只推断内层矩阵的布局
- 适合在已知 shape 和 stride 但不想实例化实际 tensor 的场景使用

**示例：**

```python
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType

# 2a. 2D tensor，RowMajor stride
a_op = OpTensor.from_shape_stride(
    shape=(128, 256),
    stride=(256, 1),  # RowMajor stride: (n, 1)
    dtype=DataType.FLOAT
)
print(f"A: shape={a_op.shape}, stride={a_op.stride}, layout={a_op.layout}")
# 输出: A: shape=(128, 256), stride=(256, 1), layout=RowMajor((128, 256))

# 2b. 2D tensor，ColumnMajor stride
a_op = OpTensor.from_shape_stride(
    shape=(128, 256),
    stride=(1, 128),  # ColumnMajor stride: (1, m)
    dtype=DataType.FLOAT
)
print(f"A: shape={a_op.shape}, stride={a_op.stride}, layout={a_op.layout}")
# 输出: A: shape=(128, 256), stride=(1, 128), layout=ColumnMajor((128, 256))

# 2c. 3D batched tensor
a_op = OpTensor.from_shape_stride(
    shape=(8, 128, 256),  # (batch, m, n)
    stride=(32768, 256, 1),  # batched RowMajor stride
    dtype=DataType.FLOAT
)
print(f"A (batched): shape={a_op.shape}, layout={a_op.layout}")
# 输出: A (batched): shape=(8, 128, 256), layout=RowMajor((128, 256))
```

---

### 方式3: from_tensor 类方法

**方法签名：**

```python
@classmethod
OpTensor.from_tensor(
    cls,
    tensor: SupportedTensor,  # torch.Tensor 或 np.ndarray
    layout: Optional[Layout] = None,
    dtype: Optional[DataType] = None
) -> "OpTensor"
```

**参数说明：**

- `tensor`: `torch.Tensor` 或 `np.ndarray` 对象
- `layout`: 可选的 Layout，如果不提供则从 tensor 的 stride 自动推断
- `dtype`: 可选的 DataType，如果不提供则从 tensor 的 dtype 自动推断

**说明：**

- 该方法会从实际的 tensor 对象中提取 shape、stride、dtype 等信息
- 如果提供了 `layout` 或 `dtype`，会覆盖从 tensor 推断的值
- 适合从已有的 tensor 对象创建 `OpTensor`

**示例：**

```python
import torch
import numpy as np
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor, ColumnMajor

# 3a. 从 torch.Tensor 创建（自动推断所有信息）
torch_a = torch.ones(128, 256, dtype=torch.float32)
a_op = OpTensor.from_tensor(torch_a)
print(f"A: shape={a_op.shape}, dtype={a_op.dtype}, stride={a_op.stride}")
# 输出: A: shape=(128, 256), dtype=DataType.FLOAT, stride=(256, 1)

# 3b. 从 torch.Tensor 创建，指定 layout
torch_a = torch.ones(128, 256, dtype=torch.float32)
a_op = OpTensor.from_tensor(torch_a, layout=RowMajor((128, 256)))
print(f"A: shape={a_op.shape}, layout={a_op.layout}")

# 3c. 从 torch.Tensor 创建，指定 dtype
torch_a = torch.ones(128, 256, dtype=torch.float32)
a_op = OpTensor.from_tensor(torch_a, dtype=DataType.FLOAT16)
print(f"A: shape={a_op.shape}, dtype={a_op.dtype}")

# 3d. 从 torch.Tensor 创建，同时指定 layout 和 dtype
torch_a = torch.ones(128, 256, dtype=torch.float32)
a_op = OpTensor.from_tensor(
    torch_a,
    layout=ColumnMajor((128, 256)),
    dtype=DataType.FLOAT16
)
print(f"A: shape={a_op.shape}, dtype={a_op.dtype}, layout={a_op.layout}")

# 3e. 从 np.ndarray 创建
np_a = np.ones((128, 256), dtype=np.float32)
a_op = OpTensor.from_tensor(np_a)
print(f"A: shape={a_op.shape}, dtype={a_op.dtype}, stride={a_op.stride}")

# 3f. 从 batched torch.Tensor 创建
torch_a_batched = torch.ones(8, 128, 256, dtype=torch.float32)
a_op = OpTensor.from_tensor(torch_a_batched)
print(f"A (batched): shape={a_op.shape}, layout={a_op.layout}")
# 输出: A (batched): shape=(8, 128, 256), layout=RowMajor((128, 256))
```

---

## Operation 支持的输入类型

Operation（如 `Gemm`、`GroupGemm`）的 `get_kernels()` 方法支持以下输入类型：

### 支持的输入类型

1. **OpTensor**（推荐）
   - 使用 `OpTensor` 可以避免实例化实际的 tensor 数据
   - 适合在代码生成阶段使用，只需要 tensor 的元数据（shape、dtype、layout）

2. **torch.Tensor**（向后兼容）
   - 直接传入 `torch.Tensor` 对象
   - 系统会自动提取信息，但需要实际的数据对象

3. **np.ndarray**（向后兼容）
   - 直接传入 `numpy.ndarray` 对象
   - 系统会自动提取信息，但需要实际的数据对象

4. **None**
   - 传入 `None` 表示使用 Operation 初始化时指定的默认值

### 示例

```python
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.arch.arch import Arch
import torch
import numpy as np

gemm_plan = Gemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor)

# 方式1: 使用 OpTensor（推荐）
a_op = OpTensor(dtype=DataType.FLOAT, layout=RowMajor((128, 256)))
b_op = OpTensor(dtype=DataType.FLOAT, layout=RowMajor((256, 384)))
kernels = gemm_plan.get_kernels(A=a_op, B=b_op)
print(f"获取到 {len(kernels)} 个 kernels")

# 方式2: 使用 torch.Tensor（向后兼容）
torch_a = torch.ones(128, 256, dtype=torch.float32)
torch_b = torch.ones(256, 384, dtype=torch.float32)
kernels = gemm_plan.get_kernels(A=torch_a, B=torch_b)
print(f"获取到 {len(kernels)} 个 kernels")

# 方式3: 使用 np.ndarray（向后兼容）
np_a = np.ones((128, 256), dtype=np.float32)
np_b = np.ones((256, 384), dtype=np.float32)
kernels = gemm_plan.get_kernels(A=np_a, B=np_b)
print(f"获取到 {len(kernels)} 个 kernels")

# 方式4: 混合使用（部分使用 OpTensor，部分使用 None）
a_op = OpTensor(dtype=DataType.FLOAT, layout=RowMajor((128, 256)))
kernels = gemm_plan.get_kernels(A=a_op, B=None)  # B 使用默认值
```

---

## 数据类型支持

### 数据类型转换

从 `torch.Tensor` 或 `np.ndarray` 创建 `OpTensor` 时，系统会自动将 tensor 的 dtype 转换为对应的 `DataType`：

```python
import torch
from catlass_cppgen.common.op_tensor import OpTensor

# torch.float32 -> DataType.FLOAT
torch_a = torch.ones(128, 256, dtype=torch.float32)
a_op = OpTensor.from_tensor(torch_a)
print(a_op.dtype)  # DataType.FLOAT

# torch.float16 -> DataType.FLOAT16
torch_b = torch.ones(128, 256, dtype=torch.float16)
b_op = OpTensor.from_tensor(torch_b)
print(b_op.dtype)  # DataType.FLOAT16
```

---

## 布局类型支持

`OpTensor` 支持以下布局类型（通过 `Layout` 类及其子类）：

### 基本布局

- **RowMajor** - 行主序布局（C 风格）
  - 2D: `RowMajor((m, n))`，stride = `(n, 1)`
  - 示例: `RowMajor((128, 256))`

- **ColumnMajor** - 列主序布局（Fortran 风格）
  - 2D: `ColumnMajor((m, n))`，stride = `(1, m)`
  - 示例: `ColumnMajor((128, 256))`

- **VectorLayout** - 向量布局
  - 1D: `VectorLayout(n)`
  - 示例: `VectorLayout(256)`

### 布局推断

当使用 `from_shape_stride` 或 `from_tensor` 时，系统会自动从 stride 推断布局：

```python
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType

# 自动推断为 RowMajor
a_op = OpTensor.from_shape_stride(
    shape=(128, 256),
    stride=(256, 1),  # RowMajor stride
    dtype=DataType.FLOAT
)
print(a_op.layout)  # RowMajor((128, 256))

# 自动推断为 ColumnMajor
b_op = OpTensor.from_shape_stride(
    shape=(128, 256),
    stride=(1, 128),  # ColumnMajor stride
    dtype=DataType.FLOAT
)
print(b_op.layout)  # ColumnMajor((128, 256))
```

---

## 使用示例

### 示例1: 基本 GEMM 操作

```python
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.arch.arch import Arch

# 创建 GEMM 计划
gemm_plan = Gemm(
    atlas_arch=Arch.Ascend950,
    element=DataType.FLOAT,
    layout=RowMajor
)

# 使用 OpTensor 创建输入
a_op = OpTensor(dtype=DataType.FLOAT, layout=RowMajor((128, 256)))
b_op = OpTensor(dtype=DataType.FLOAT, layout=RowMajor((256, 384)))

# 获取 kernels
kernels = gemm_plan.get_kernels(A=a_op, B=b_op)
print(f"获取到 {len(kernels)} 个 kernels")
```

### 示例2: Batched GEMM 操作

```python
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.arch.arch import Arch

gemm_plan = Gemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor)

# 方式1: 直接构造函数，指定 batched shape
batch_count = 8
a_op = OpTensor(
    dtype=DataType.FLOAT,
    layout=RowMajor((128, 256)),  # 内层矩阵的 layout
    shape=(batch_count, 128, 256)  # 完整的 shape，包含 batch 维度
)
b_op = OpTensor(
    dtype=DataType.FLOAT,
    layout=RowMajor((256, 384)),
    shape=(batch_count, 256, 384)
)

kernels = gemm_plan.get_kernels(A=a_op, B=b_op)
print(f"获取到 {len(kernels)} 个 kernels")
if len(kernels) > 0:
    batched_kernel = kernels[0]  # BatchedMatmulKernel
    print(f"Kernel 类型: {type(batched_kernel).__name__}")
    print(f"BatchCount: {batched_kernel.batchCount}")

# 方式2: 使用 from_shape_stride
a_op = OpTensor.from_shape_stride(
    shape=(batch_count, 128, 256),
    stride=(32768, 256, 1),  # batched RowMajor stride
    dtype=DataType.FLOAT
)
b_op = OpTensor.from_shape_stride(
    shape=(batch_count, 256, 384),
    stride=(98304, 384, 1),
    dtype=DataType.FLOAT
)
kernels = gemm_plan.get_kernels(A=a_op, B=b_op)
```

### 示例3: GroupGemm 操作

```python
from catlass_cppgen.op.group_gemm import GroupGemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.arch.arch import Arch

group_gemm_plan = GroupGemm(
    atlas_arch=Arch.Ascend950,
    element=DataType.FLOAT,
    layout=RowMajor
)
problem_count = 4  # 4 个不同的 GEMM 问题

# 使用 OpTensor 创建（2D tensor，每个问题的 A 和 B 矩阵维度相同）
a_op = OpTensor(dtype=DataType.FLOAT, layout=RowMajor((128, 256)))
b_op = OpTensor(dtype=DataType.FLOAT, layout=RowMajor((256, 384)))

kernels = group_gemm_plan.get_kernels(
    A=a_op,
    B=b_op,
    problemCount=problem_count
)
print(f"获取到 {len(kernels)} 个 kernels")
if len(kernels) > 0:
    grouped_kernel = kernels[0]  # GroupedMatmulSliceMKernel
    print(f"Kernel 类型: {type(grouped_kernel).__name__}")
    print(f"ProblemCount: {grouped_kernel.problemCount}")
```

### 示例4: 从实际 tensor 创建

```python
import torch
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.arch.arch import Arch

gemm_plan = Gemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor)

# 从 torch.Tensor 创建 OpTensor
torch_a = torch.ones(128, 256, dtype=torch.float32)
torch_b = torch.ones(256, 384, dtype=torch.float32)

a_op = OpTensor.from_tensor(torch_a)
b_op = OpTensor.from_tensor(torch_b)

kernels = gemm_plan.get_kernels(A=a_op, B=b_op)
print(f"获取到 {len(kernels)} 个 kernels")
```

---

### 数据类型和布局的一致性

确保 Operation 初始化时指定的数据类型和布局与输入 tensor 一致，或者让系统自动推断：

```python
# 方式1: Operation 指定默认值，输入使用 None
gemm_plan = Gemm(
    atlas_arch=Arch.Ascend950,
    element=DataType.FLOAT,
    layout=RowMajor
)
kernels = gemm_plan.get_kernels(A=None, B=None)  # 使用默认值

# 方式2: 输入显式指定，覆盖默认值
a_op = OpTensor(dtype=DataType.FLOAT16, layout=ColumnMajor((128, 256)))
kernels = gemm_plan.get_kernels(A=a_op, B=b_op)  # 使用 a_op 指定的值
```

---

## 总结

本文档介绍了 catlass_cppgen 中所有支持的输入方式：

1. **OpTensor 的三种构造方法**：
   - 直接构造函数：`OpTensor(dtype, layout, shape=None, data_ptr=None)`
   - `from_shape_stride`：从 shape 和 stride 创建，自动推断 layout
   - `from_tensor`：从 `torch.Tensor` 或 `np.ndarray` 创建

2. **Operation 支持的输入类型**：
   - `OpTensor`（推荐）
   - `torch.Tensor`（向后兼容）
   - `np.ndarray`（向后兼容）
   - `None`（使用默认值）

3. **数据类型和布局**：
   - 支持多种数据类型（FLOAT、FLOAT16、INT8 等）
   - 支持多种布局（RowMajor、ColumnMajor、VectorLayout 等）

4. **最佳实践**：
   - 优先使用 `OpTensor` 避免实例化数据
   - 根据场景选择合适的构造方法
   - 正确处理 batched tensor 的维度

通过合理使用这些输入方式，可以高效地进行代码生成和 kernel 调优。
