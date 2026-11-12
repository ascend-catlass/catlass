# `EVG` 基础文档 (Python)

本文档为 `catlass_cppgen` 中 [EVG（Epilogue Visitor Graph）模块](../../../docs/zh/2_Design/03_evg/01_evg_design.md)的使用说明。EVG 将 Python 描述的 epilogue 后处理函数解析为 DAG（Directed Acyclic Graph，有向无环图），并生成对应的 C++ Visitor 代码。

核心入口函数定义见 [`evg_extension.py`](../catlass_cppgen/catlass/evg_extension.py)，节点与定义见 [`evg/`](../catlass_cppgen/catlass/evg/)。

---

## `evg()` — 入口函数

解析 epilogue Python 函数，生成 EVG 定义字符串与参数。

```python
def evg(
    fn_src: str,
    example_inputs: Dict[str, OpTensor],
) -> Tuple[str, str, str, EVGArgRenames]:
```

| 参数 | 说明 |
|------|------|
| `fn_src` | epilogue 函数的 Python 源码字符串，函数固定命名为 `epilogue`，最后一个 `return` 语句的返回值作为输出 |
| `example_inputs` | 输入/输出张量的元数据字典，key 为变量名，value 为 `OpTensor` |

**返回值 `(callback_name, evg_args, evg_str, arg_renames)`：**

| 字段 | 类型 | 说明 |
| ------ | ------ | ------ |
| `callback_name` | `str` | 生成的 epilogue callback 名称（固定为 `"EVGResult"`） |
| `evg_args` | `str` | EVG 参数结构体声明 C++ 代码，包含 `Arguments` 和 `computeLength` |
| `evg_str` | `str` | EVG Visitor 类型定义 C++ 代码，包含 `VisitorAccLoad`、`VisitorAuxLoad`、`VisitorCompute`、`TreeVisitor`、`TopologicalVisitor` 等 |
| `arg_renames` | `EVGArgRenames` | 参数重命名信息（当前版本暂返回 `None`） |

### 使用示例

```python
from catlass_cppgen.catlass.evg_extension import evg
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.common.op_tensor import OpTensor

fn_src = """
def epilogue(accum, bias):
    result = accum + bias
    return result
"""

inputs = {
    "accum": OpTensor.from_shape_stride((128, 256), (256, 1), DataType.FLOAT),
    "bias":  OpTensor.from_shape_stride((1, 256), (256, 1), DataType.FLOAT),
    "result": OpTensor.from_shape_stride((128, 256), (256, 1), DataType.FLOAT),
}

callback_name, evg_args, evg_str, _ = evg(fn_src=fn_src, example_inputs=inputs)
print(callback_name)  # "EVGResult"
print(evg_args)       # typename EVGResult::Arguments evg_args{...};
print(evg_str)        # using Result = Catlass::Epilogue::Fusion::VisitorAuxStore<...>; ...
```

### 支持的 Epilogue 算子

在 `fn_src` 的 `epilogue` 函数中，可使用以下算子：

| 类别 | 写法 | 说明 |
| ------ | ------ | ------ |
| add | `accum + bias` | 二元加法 |
| sub | `accum - bias` | 二元减法 |
| mul | `accum * scale` | 二元乘法 |
| div | `accum / scale` | 二元除法 |
| relu | `relu(accum)` | ReLU 激活 |
| leakyRelu | `leakyRelu(accum, alpha)` | LeakyReLU 激活 |
| Prelu | `Prelu(accum, weight)` | PReLU 激活 |
| sigmoid | `sigmoid(accum)` | Sigmoid 激活 |
| silu | `silu(accum)` | SiLU 激活 |
| maximum | `maximum(a, b)` | 逐元素取最大值 |
| minimum | `minimum(a, b)` | 逐元素取最小值 |
| cast | `cast(accum, "float16", "float")` | 类型转换（参数：目标类型, 源类型 [, RoundMode]） |
| constant | `constant(1.0, "float")` | 创建常量值 |

算子可串联组合，例如：`relu(accum) + bias` 会生成包含 `VisitorCompute(Relu)` 和 `VisitorCompute(Add)` 的 TreeVisitor 链。

**广播**：当前支持 `RowBroadcast`（行广播）。输入张量形状为 `(1, N)` 时，自动匹配 `(M, N)` 累加器形状并标记为 `RowBroadcast`。`ColumnBroadcast` 暂不支持。

---

## EVG 使用参考

在`catlass_cppgen`中，要使用 EVG 特性，有下述三种办法：

- 在创建kernel时直接通过 `Gemm(evg_config=...)` 传入，参考示例：

```python
from catlass_cppgen.op.gemm import Gemm

evg_config = {
    "fn_src": "def epilogue(accum, bias):\n    return relu(accum + bias)",
    "example_inputs": {"accum": ..., "bias": ..., "result": ...},
}
gemm = Gemm(..., evg_config=evg_config, A=a, B=b)

```

- 对已有 Kernel 直接调用 `to_evg(evg_config)`，参考示例：

```python
gemm = Gemm(...)

gemm = gemm.to_evg(evg_config=evg_config)
```

- 直接调用`evg(fn_src, example_inputs)` 生成 EVG 定义，如上述[使用示例](#使用示例)所示。
