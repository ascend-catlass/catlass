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

## `tla.experimental.mx_to`

**源码：** [`catlass.tla.experimental.mx_to`](../../../../catlass/tla/experimental/__init__.py)

功能说明：

在 `tla.vector()` 中对张量 tile 执行 MX 转换。普通 `Tensor` 表示
`f16` / `bf16` / `f32` 等非 MX 数据，`(payload, scale)` 元组表示 MX payload 和 `e8m0` scale。
该操作将结果写入目标张量或元组，并原样返回该对象。普通输入转换为 MX 时，
会按分组轴每 32 个元素推导一个 scale。MX 输入可转换为普通浮点张量，或重新量化为另一种 MX 格式。

调用者无需进入 `tla.vec.func()`。bitcode 循环处理完整 tile；一维和二维张量的行或列支持非连续 stride。
通用 bitcode 路径支持 `zN`、`nZ`、`zZMxScale`、`nNMxScale` 等二维分块布局，MX 到 MX 转换可以改变 scale 分组轴。
MX tile 坐标须从 32 元素分组边界开始，连续打包的 FP4 tile 须从字节边界开始。
普通数据与 MX 之间转换时，普通张量可使用不同的 stride 和连续轴；scale 张量也可以使用与 payload 张量不同的 stride。
通用布局路径逐组暂存数据，可能比连续或受支持的非连续快速路径慢。
对于运行时 shape、stride 或坐标，调用者须确保满足这些对齐要求，payload shape 一致，scale 范围足够，
且二维打包 FP4 tile 的行 stride 为偶数。根据运行时范围推断分块布局的分组轴时，该轴须至少包含 32 个元素。
二维布局的 scale 分组轴须能根据编译时 payload 和 scale 范围无歧义地确定。输入和输出 payload 的存储空间不能相互重叠，也不能与 scale 存储空间重叠。
输入和输出 scale 可共享存储空间，但共享的视图须从同一地址开始。目标布局须将不同的逻辑元素或 scale 分组映射到不同的存储位置。
`VLn` 掩码按照分组顺序选择前 n 个元素（MX 输出按目标分组，否则按输入分组），`ALLF` 不选择任何元素；不支持其他掩码。
未选中的 payload 元素会写入零；输出 scale 仍会写入，扩宽转换会复制输入 scale。
尚未支持的布局会被拒绝。
GM 与 UB 之间的数据搬运仍由调用者显式安排。

函数原型：

```python
tla.experimental.mx_to(src, dst, *, round_mode=RoundMode.CAST_ROUND, mask=None) -> Tensor | tuple[Tensor, Tensor]
```

参数说明：

- `src`（`Tensor | tuple[Tensor, Tensor]`）：普通 UB tile 或 MX `(payload, scale)` 元组。必填。
- `dst`（`Tensor | tuple[Tensor, Tensor]`）：目标 UB tile 或 MX `(payload, scale)` 元组；原样返回。必填。
- `round_mode`（`RoundMode`）：payload 舍入模式；默认使用 `RoundMode.CAST_ROUND`。可选。
- `mask`（`pattern | None`）：可选 `ALL`、`ALLF` 或 `VLn` 前缀掩码。

约束说明：

- 须在 `@tla.kernel` 函数体及 `tla.vector()` 中调用，且不能在 `tla.vec.func()` 中调用。
- 至少一个参数必须是 MX 元组。普通输入须转换到 MX 元组；MX 输入可转换到普通浮点 tile 或另一个 MX 元组。
- MXFP4 目标仅支持 `RoundMode.CAST_ROUND`。

调用示例：

```python
with tla.vector():
    tla.experimental.mx_to(dense_ub, mx_fp8_ub)
    tla.experimental.mx_to(mx_fp8_ub, dense_out_ub)
    tla.experimental.mx_to(mx_fp8_ub, mx_fp4_ub)
```
