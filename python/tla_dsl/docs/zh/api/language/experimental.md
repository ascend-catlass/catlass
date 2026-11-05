---
nav_order: 12
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/experimental.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# 实验性接口

[语言参考](index.md) | [API 速查](../quick_reference.md)

`tla.experimental` 下的接口可能变更或移除，不保证向后兼容。仍使用普通的
`tla.kernel` / `tla.jit` 装饰器；该命名空间不切换另一套编译器。

## `tla.experimental.img2col`

**源码：** [`catlass.tla.experimental.img2col`](../../../../catlass/tla/experimental/__init__.py)

功能说明：

将 L1 特征图 tile 以 img2col 展开加载到 L0A，供卷积使用。

函数原型：

```python
tla.experimental.img2col(dst: Tensor, src: Tensor, params: Img2colParams) -> None
```

参数说明：

- `dst`（`Tensor`）：L0A tile，标签 `zN`，形状 `(M, K)`。形状给出加载范围，
  coord 给出起点，因此同一 L1 tile 上瞄准不同窗口即可对隐式 GEMM 的 M/K 做分块，
  不增加额外 L1 流量。必填。
- `src`（`Tensor`）：L1 特征图 tile，标签 `zNUnAlign`，NC1HWC0 嵌套为
  `((h*w, 1), (C0, c1))`。必填。
- `params`（`Img2colParams`）：卷积几何。必填。
  - `fmap_hw`（`tuple`）：特征图 `(H, W)`。必填。
  - `filter_hw`（`tuple[int, int]`）：卷积核 `(kH, kW)`；须为编译期整数。必填。
  - `stride_hw`（`tuple`）：步长 `(sH, sW)`。默认 `(1, 1)`。
  - `dilation_hw`（`tuple`）：膨胀 `(dH, dW)`。默认 `(1, 1)`。
  - `pad`（`tuple`）：填充 `(left, right, top, bottom)`。默认 `(0, 0, 0, 0)`。
  - `pad_value`（`int`）：填充值。默认 `0`。

约束说明：

- 须在 `@tla.kernel` 装饰的 kernel 函数体内调用。
- 须在 `tla.cube()` 内调用。
- 元素类型须为 `f32` / `f16` / `bf16`（仅 2、4 字节；无 int8/fp8 路径）。
- `src` 与 `dst` 须同元素类型。
- 源通道来自 NC1HWC0 源形状，与目的 K 窗口无关。源 C0/C1 与目的 K 范围及 K
  坐标须为静态；`filter_hw` 须为编译期整数。
- 目的 K 范围与起始坐标须覆盖完整 C1 切片（`kh * kw * C0` 个元素），且落在源
  完整 K 轴内。源通道与目的 K 范围须在 1..65535；目的 K 坐标须在 0..65535。
- `fmap_hw` 各分量须在 1..65535，stride / dilation 在 1..255，pad 各分量在
  0..255。编译期常量超范围会被拒绝；运行期取值满足上述范围为调用方前置条件，
  违反则行为未定义。
- 卷积核尺寸为编译期整数，范围 1..511。

调用示例：

```python
with tla.cube():
    # l1_fmap 为 NC1HWC0；l1_filter 为 (K, N) 权重矩阵。
    tla.experimental.img2col(l0_a, l1_fmap, tla.experimental.Img2colParams(
        fmap_hw=(hi, wi), filter_hw=(3, 3), pad=(1, 1, 1, 1)))
    tla.copy(l0_b, l1_filter)
    tla.mmad(l0_c, l0_a, l0_b, init_c=True)
```
