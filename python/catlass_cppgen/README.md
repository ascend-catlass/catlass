# catlass_cppgen

## 项目介绍

`catlass_cppgen` 是一个基于 Python 的代码生成框架，用于构建和生成 CATLASS 高性能算子。该框架提供了灵活的接口来定义算子参数、选择优化策略，并自动生成对应的 C++ 核函数代码。

### 主要特性

- **算子代码生成**：通过 Python API 定义算子参数，自动生成优化的 C++ 核函数代码
- **灵活的调优接口**：支持自定义TileShape、DispatchPolicy等参数
- **多架构支持**：支持多种硬件架构（包括AtlasA2/A3, Ascend950）
- **类型安全**：提供完整的数据类型和布局抽象

### 工程结构

以下是本项目目录结构说明：

```plain
./catlass_cppgen
├── catlass_cppgen
│   ├── __init__.py
│   ├── catlass                      # CATLASS 相关特性组件
│   │   ├── __init__.py
│   │   ├── arch                     # 代际声明
│   │   ├── evg                      # EVG 特性承载
│   │   ├── evg_extension.py
│   │   ├── gemm
│   │   ├── gemm_coord.py
│   │   ├── layout
│   │   └── library.py
│   ├── common                       # 通用组件
│   │   ├── __init__.py
│   │   ├── data_type.py
│   │   ├── op_tensor.py
│   │   ├── typing.py
│   │   └── utils.py
│   ├── kernel
│   │   ├── __init__.py
│   │   ├── gemm                      # GEMM 类算子特化类
│   │   ├── group_gemm                # Group GEMM 类算子特化类
│   │   ├── kernel_base.py
│   │   └── visitor_kernel_base.py
│   ├── op                            # 算子Kernel基类
│   │   ├── __init__.py
│   │   ├── gemm.py
│   │   ├── group_gemm.py
│   │   └── op.py
│   └── _version.py
├── docs                              # API 文档
│   ├── evg_api.md
│   ├── kernel_api.md
│   └── optensor_api.md
├── tests                             # 单元测试组件
│   ├── catlass                       # 面向 CATLASS 相关特性的测试件
│   ├── common                        # 面向通用组件的测试件(类型、排布)
│   └── op                            # 面向算子kernel生成的测试件
├── pyproject.toml
├── README.md                         # 主README文档
└── uv.lock
```

## 支持的算子

### GEMM 类（矩阵乘法）

| 算子类型 | Kernel 类 | 主要特性 | 切分轴 |
|---------|----------|---------|--------|
| **基础矩阵乘法** | `BasicMatmulKernel` | • 输入张量 A 和 B 为 2 维<br>• `alpha = 1.0` 且 `beta = 0.0`<br>• 支持可选的 Bias 参数 | 无 |
| **批处理矩阵乘法** | `BatchedMatmulKernel` | • 输入张量 A 和 B 为 3 维（batchCount, M, K）和（batchCount, K, N）<br>• 所有批次共享相同的矩阵维度<br>• `alpha = 1.0` 且 `beta = 0.0` | 无 |
| **Strided 批处理矩阵乘法** | `StridedBatchedMatmulKernel` | • layout 直接编码 batch stride<br>• `transA/transB` 支持将物理 batch 轴放到矩阵两维之间<br>• stride 和 leading dimension 由 M/N/K/batchCount 自动推导 | 无 |
| **EVG Visitor 矩阵乘法** | `BasicMatmulTlaVisitorKernel` | • 支持CATLASS模板库后处理框架 EVG（Epilogue Visitor Graph） | 无 |
| **多核 Split-K** | `MultiCoreSplitkMatmulKernel` | • 输入张量 A 和 B 为 2 维<br>•  优化动作：沿 K 方向多核切分<br>• 支持可选的 Bias 参数 | K |
| **尾块多核 Split-K** | `TailMultiCoreSplitkMatmulKernel` | • 输入张量 A 和 B 为 2 维<br>•  多核切K的尾块优化变体<br>• 支持可选的 Bias 参数 | K |
| **Stream-K** | `StreamkMatmulKernel` | • 输入张量 A 和 B 为 2 维<br>• 优化动作：Stream-K 调度策略<br>• 支持可选的 Bias 参数 | K |

### Group GEMM 类（分组矩阵乘法）

| 算子类型 | Kernel 类 | 主要特性 | 切分轴 |
|---------|----------|---------|--------|
| **分组矩阵乘（M 轴切分）** | `GroupedMatmulSliceMKernel` | 多组不同 M 维度的矩阵乘法 | M |

### EVG 后处理 (Epilogue Visitor Graph)

支持通过EVG(Epilogue Visitor Graph)框架实现后处理功能，支持的后处理类别包括：

 - **单一计算环节**：可通过运算符或函数调用表达以下算子：

   | 类别 | 算子 | 写法示例 |
   |------|------|----------|
   | 二元运算 | add | `accum + bias` |
   | 二元运算 | sub | `accum - bias` |
   | 二元运算 | mul | `accum * scale` |
   | 二元运算 | div | `accum / scale` |
   | 激活函数 | relu | `relu(accum)` |
   | 激活函数 | leakyRelu | `leakyRelu(accum, alpha)` |
   | 激活函数 | Prelu | `Prelu(accum, weight)` |
   | 激活函数 | sigmoid | `sigmoid(accum)` |
   | 激活函数 | silu | `silu(accum)` |
   | 比较/选择 | max / min | `max(a, b)` / `min(a, b)` |
   | 类型转换 | cast | `cast(accum, "float16", "float")` |

   | 常量 | constant | `constant(1.0, "float")` |
 - **组合计算**：支持多个计算节点拼接；
 - **广播计算**：支持行广播计算。

## 安装

### 从源码安装

1. **构建分发包**：

   ```bash
   pip install build
   python -m build
   ```

   这会在 `dist/` 目录下生成 `.whl` 和 `.tar.gz` 文件。

2. **安装分发包**：

   ```bash
   pip install dist/catlass_cppgen-*.whl

   ```

   或者：

   ```bash
   pip install dist/catlass_cppgen-*.tar.gz
   ```

3. **直接安装（开发模式）**：

   ```bash
   pip install -e .

   ```

### 从本地目录安装

如果您想直接从项目目录安装：

```bash
pip install .

```

## 使用示例

当前 `catlass_cppgen` 支持 matmul、grouped_matmul 以及 EVG 后处理特性的代码生成。以下是应用`cppgen`的环节示意：

```plain
Gemm / GroupGemm（算子规划）
    ↓ get_kernels()
Kernel 对象（调优与特性查询）
    ↓ tune() / to_evg()
配置完成的 Kernel
```

详细使用示例和 API 文档请参考下述文档：

 - [Kernel API 基础文档](docs/kernel_api.md)
 - [`OpTensor` 基础文档](docs/optensor_api.md)
 - [`EVG` 基础文档](docs/evg_api.md)

### 基础 GEMM

以下是一个基础的创建 matmul 算子cppgen对象的示例：

```python
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm.dispatch_policy import MmadPingpong

# 1. 描述输入张量（无需绑定底层数据）
a = OpTensor.from_shape_stride((128, 256), (256, 1), DataType.FLOAT)
b = OpTensor.from_shape_stride((256, 384), (384, 1), DataType.FLOAT)

# 2. 创建算子并获取 Kernel
gemm = Gemm(atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor, A=a, B=b)
kernels = gemm.get_kernels()

# 3. [可选] 可以显式指定使用的Kernel组件（以`BasicMatmulKernel`为例）
from catlass_cppgen.kernel.gemm import BasicMatmulKernel
kernel = find_kernel_by_type(kernels, BasicMatmulKernel)
# 非定向指定: kernel = kernels[0]

# 4. [可选] 调优 Tile 形状与调度策略
kernel.tune(
    GemmShape(128, 256, 64),
    GemmShape(128, 256, 64),
    dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950),
)

# 5. [可选] matmul hello world: 核函数的代码生成
print(f"[Kernel] \n{kernel.gen_kernel_template()}")
```

得到kernel对象后，可以调用`gen_kernel_template()`，`gen_params_device()`等方法，针对核函数和参数绑定做代码生成。

Strided Batched Matmul 可通过 `Gemm` 规划器选择，stride 由 shape 和转置配置直接生成：

```python
a = OpTensor.from_shape_stride((3, 128, 256), (256, 3 * 256, 1), DataType.FLOAT16)
b = OpTensor.from_shape_stride((3, 256, 384), (256 * 384, 384, 1), DataType.FLOAT16)

kernel = Gemm(
    atlas_arch=Arch.AtlasA2,
    element_C=DataType.FLOAT16,
    A=a,
    B=b,
    strided_batched=True,
    transA=True,
    transB=False,
).get_kernels()[0]
```

### Group GEMM

以下是建立 matmul 算子cppgen对象的示例：

```python
from catlass_cppgen.op.group_gemm import GroupGemm
from catlass_cppgen.catlass.layout.layout import VectorLayout

# ...

# 1. 创建groupList 张量
groupList = OpTensor(dtype=DataType.INT64, layout=VectorLayout(4), shape=(4,))

# 2. 建立 group matmul 对象
group_gemm = GroupGemm(atlas_arch=Arch.Ascend950, A=a, B=b_3d, groupList=groupList)

# 3. [可选] 取得kernel并做 Tiling 调优
kernels = group_gemm.get_kernels()
kernels[0].tune(GemmShape(256, 256, 256), GemmShape(256, 256, 64))
```

### EVG 后处理

```python
# ...

# 1. 进行evg对象声明
# - fn_src: EVG对象函数头
# - example_inputs: 后处理过程中涉及的`name:tensor`键值对
evg_config = {
    "fn_src": "def epilogue(accum, bias):\n    return relu(accum + bias)",
    "example_inputs": {
        "accum": OpTensor.from_shape_stride((128, 256), (256, 1), DataType.FLOAT),
        "bias":  OpTensor.from_shape_stride((1, 256), (256, 1), DataType.FLOAT),
        "result": OpTensor.from_shape_stride((128, 256), (256, 1), DataType.FLOAT),
    },
}

# 2. 创建 matmul 算子对象并获取 Kernel
gemm = Gemm(atlas_arch=Arch.Ascend950, evg_config=evg_config, A=a, B=b)
kernel = gemm.get_kernels()[0]
# `is_support_evg` 特性为 True
assert kernel.is_support_evg

# [可选] 3. 进行 Tile 形状调优
kernel.tune(GemmShape(128, 256, 64), GemmShape(128, 256, 64))
```

支持二元运算（add/sub/mul/div）、激活函数（relu/silu/sigmoid/leakyRelu/prelu）、类型转换（cast）、常量（constant）等，可串联组合使用。
