# python扩展

为方便开发者使用CATLASS算子，代码仓基于pybind11和torch提供了使用python调用CATLASS算子的示例。

## 代码结构

```bash
python_extension
├── CMakeLists.txt                      # CMake配置文件
├── README.md                           # 说明文档
├── pyproject.toml                      # 项目配置文件
├── setup.py                            # 安装脚本
├── src
│   ├── bindings
│   │   ├── pybind_bindings.cpp         # pybind11绑定文件
│   │   └── torch_bindings.cpp          # torch绑定文件
│   ├── include
│   │   └── wrapper
│   │       └── catlass_kernel_wrapper.h    # wrapper头文件
│   └── wrapper
│       └── catlass_kernel_wrapper.cpp      # catlass算子wrapper文件
└── torch_catlass
    └── __init__.py                     # 初始化入口，用于打包
tests
└── test_python_extension.py        # 测试脚本
```

## 编译产物结构

```bash
output/python_extension
├── libcatlass_torch.so                             # torch动态链接库
└── torch_catlass-0.1.0.20250330120000.cp310-cp310-linux-x86_64.whl  # pybind11动态链接库的wheel包
```

## 使用说明

- 假定开发者已在`shared_lib/`中，增加了所需算子的实现和入口。

### pybind接口实现

由于pybind传入参数为`at::Tensor`而非AscendC中的地址指针（`GM_ADDR`），所以需要对python端传来的数据进行处理。
主要步骤为根据输入tensor的信息填充运行信息参数，申请输出内存。
此部分与算子本身参数相关，可参考已有的`BasicMatmul`实现。

### 编译

各部分代码完成后：

- 使用`bash scripts/build.sh python_extension`编译pybind扩展。
- 使用`bash scripts/build.sh torch_library`编译torch扩展。

编译环境与[README](../../README.md)相同，但需要增加如下python依赖：

- 必须：
  - `pybind11`
  - `gcc` _版本为9.0+_
  - `torch`_建议使用2.1+_
  - `TorchNPU`_配套`torch`和`CANN`的最新版本，可在[Ascend/pytorch](https://gitcode.com/ascend/pytorch)查询_
- 可选：
  - `pybind11-stubgen`

### 安装

- 对于`torch`扩展，只需要在使用算子前增加如下代码：

```python
torch.ops.load_library("output/python_extension/libcatlass_torch.so")
```

- 对于pybind扩展，编译产物即为一个wheel包，执行`pip install torch_catlass-xxxxx.whl`即可。

### 运行

```python
import torch_catlass
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

class CatlassTest(TestCase):
    def test_basic_matmul(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        result = torch_catlass.basic_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)
    def test_basic_matmul_torch_lib(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        torch.ops.load_library("../../output/python_extension/libcatlass_torch.so") # 确保加载正确路径
        result = torch.ops.CatlassTorch.basic_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)

if __name__ == "__main__":
    run_tests()
```

## 注意事项

- 本示例仅用于CATLASS算子接入pybind/torch.ops的参考，为保证代码简洁，不进行泛化的支持，如多个算子、多个平台等。
- 我们目前提供了四种典型算子作为示例：
  - `BasicMatmul`：基本矩阵乘法，提供类型模板的实现方法
  - `GroupedMatmul`：分组矩阵乘法，提供分组输入输出示例
  - `OptimizedMatmul`：优化矩阵乘法，提供CV融合的示例
  - `ConvBias`：卷积算子
- 示例仅支持以下产品：
  - Atlas A2系列产品、Atlas A3系列产品（`2201`架构）
