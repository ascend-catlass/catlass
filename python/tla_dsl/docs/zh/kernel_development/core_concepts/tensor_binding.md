---
nav_order: 40
---

# Tensor 接入

本文说明如何在 Host 侧得到可供 `tla.compile` / 启动使用的 `tla.Tensor`：用 [DLPack](https://github.com/dmlc/dlpack) 的 `from_dlpack`（`torch` / `torch_npu`）绑定真实缓冲，或用 `make_fake_tensor` 造不带设备指针的类型样本。编译、缓存和重复启动见 [编译与启动](compile_and_launch.md)。静态 / 动态 layout 见 [Layout](layout.md)。接口完整说明见 [程序生命周期参考](../../api/runtime/index.md)。

---

## 使用 `from_dlpack` 显式转换

运行时提供将兼容 DLPack 的 tensor 转为 `tla.Tensor` 的接口。下面从 torch 开始，经 `from_dlpack` 绑定，一直到 `compile` / launch：

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

@tla.kernel
def foo(src: tla.Tensor, dst: tla.Tensor) -> None:
    n = src.origin_shape[0]
    # ... 在 vector/cube 区域内完成计算 ...

torch.npu.set_device(0)

x = torch.rand(1024, dtype=torch.float32, device="npu") * 10.0 - 5.0
y = torch.zeros_like(x)

tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
ty = from_dlpack(y.contiguous(), layout_tag=tla.arch.RowMajor)
# 需要跨 shape 复用编译产物时再 mark_*_dynamic
tx = tx.mark_compact_shape_dynamic(0)
ty = ty.mark_compact_shape_dynamic(0)

compiled = tla.compile(foo, tx, ty, options="--npu-arch 3510")
compiled(tx, ty, block_num=1)
torch.npu.synchronize()
```

编译产物如何缓存、首次与重复启动有何区别，以及如何在多个设备上复用同一份编译
产物，见 [编译与启动](compile_and_launch.md)。

示例中的 `x` 是实现 `__dlpack__` 的 NPU 张量。返回的 `tla.Tensor` 与 `x` 零拷贝共享设备内存，并持有对 `x` 的引用。

默认得到**静态 layout**（具体 shape / stride / origin 写进编译类型）。要得到动态或混合静态/动态 layout，须在转换后调用 `mark_layout_dynamic` / `mark_compact_shape_dynamic`（见 [Layout](layout.md)）。

完整签名如下：

```python
def from_dlpack(
    tensor_dlpack: object,
    *,
    layout_tag: Any,
    origin_shape: Any | None = None,
    assumed_align: int | None = None,
    stream: int | None = -1,
    element_type: type | None = None,
) -> Tensor:
```

| 参数 | 说明 |
|------|------|
| `tensor_dlpack` | 实现 `__dlpack__` 的对象。**Ascend/NPU** 缓冲区可用于启动 kernel。CPU 缓冲区（如 PyTorch 张量和 NumPy 数组）仅提供 `tla.compile` 所需的元数据，返回的张量 `data_ptr == 0`。 |
| `layout_tag` | 必填的 `tla.arch` 布局标签。对于四维预打包 NZ 视图，`from_dlpack` 按此标签校验导出的形状和步长。 |
| `origin_shape` | 以 Python 元组表示的逻辑形状，例如 `(M, N)`。线性布局可从输入推导。一维预打包 NZ 缓冲区须显式传入，四维预打包 NZ 视图默认使用填充后的尺寸。 |
| `assumed_align` | **预留参数**。当前传入值不影响实际行为 |
| `stream` | 传给框架的 `__dlpack__(stream=...)`，只影响转换时框架侧是否做流同步；默认 `-1` 表示不做同步 |
| `element_type` | 可选，指定 TLA 元素类型。默认 `None`，使用 DLPack 元素类型。 |

### 布局与物理存储

未传 `origin_shape` 时，二维 `RowMajor` / `ColumnMajor` 须满足下列物理布局约定：

| `layout_tag` | Host 侧准备 |
|--------------|-------------|
| `RowMajor` | `tensor.contiguous()`（物理行主序） |
| `ColumnMajor` | `tensor.permute(1, 0).contiguous()`（逻辑列主，物理仍为行主序紧凑缓冲） |

物理布局与 `layout_tag` 不一致时抛出 `RuntimeTensorError`。

若显式传入 `origin_shape`，则使用该逻辑 origin，且不再按上表校验物理布局。`zN` 等嵌套布局下，逻辑 origin 为二维 `(M, N)`，对应的 shape / stride 为嵌套四元组。

四维预打包 NZ 视图按 `(m0, m1, n0, n1)` 的顺序导出物理维度。形状和有效步长须匹配预打包布局。传入 `origin_shape=(M, N)` 可保留填充前的逻辑尺寸。将预打包 NZ 张量作为 GM 参数传给 kernel 时，编译前须调用 `mark_layout_dynamic()`。

PyTorch 的 `as_strided` 通过 `size` 参数接收视图的形状。

```python
# packed_zn 是逻辑尺寸为 (32, 64) 的 FP32 zN 一维预打包 NPU 缓冲区。
view_4d = packed_zn.as_strided(size=(16, 2, 8, 8), stride=(8, 128, 1, 256))
gm = tla.from_dlpack(view_4d, layout_tag=tla.arch.zN).mark_layout_dynamic()
```

### 代码示例

下面演示如何用 `from_dlpack` 将 PyTorch tensor 转为 `tla.Tensor`，并查看转换结果：

```python
import torch
import torch_npu
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

torch.npu.set_device(0)
x = torch.rand(30, 20, dtype=torch.float32, device="npu") * 10.0 - 5.0
y = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)

print(y.shape)         # 逻辑 shape
print(y.stride)        # stride 树
print(y.origin_shape)  # 逻辑 origin
print(y.layout_tag)    # 如 'RowMajor'
print(y.dtype)         # 如 'f32'
print(y.addrspace)     # 如 'gm'
print(y.data_ptr)      # 设备地址
print(y)               # 编译类型字符串
```

转换后可通过下列属性查看 tensor 信息：

- `tensor.shape`：逻辑 shape
- `tensor.stride`：stride
- `tensor.origin_shape`：逻辑 origin
- `tensor.layout_tag`：布局标签
- `tensor.dtype`：元素类型
- `tensor.addrspace`：地址空间
- `tensor.data_ptr`：设备指针

---

## 使用 `make_fake_tensor` 造不带 ptr 的 fake tensor

仅需给 `tla.compile` 提供类型 / layout 样本、不绑定真实 NPU 缓冲时，用 `make_fake_tensor`。它始终返回未绑定 Host tensor（`data_ptr == 0`）；真实 buffer 请走 `from_dlpack`。

用法：必传 ``(dtype, shape, stride)``；``layout_tag`` 可选，默认 ``RowMajor``。显式传入的 shape/stride 按原样使用。

```python
import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

fake = make_fake_tensor(
    tla.Float32,
    (64, 128),
    (128, 1),
)
assert fake.data_ptr == 0
assert fake.layout_tag == "RowMajor"

# zN 等嵌套布局：自行传入嵌套 shape/stride（逻辑 origin 仍为二维）
zn = make_fake_tensor(
    tla.Float16,
    ((16, 2), (16, 4)),
    ((16, 256), (1, 512)),
    layout_tag=tla.arch.zN,
    origin_shape=(32, 64),
)
```
