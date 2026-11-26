# CATLASS Python

CATLASS Python 是 CATLASS 的 Python 前端。它在 AscendNPU-IR 的基础上构建 TLA Dialect，将 Python 中描述的 TLA 操作降级为 MLIR，并由 AscendNPU-IR 工具链编译为可执行的核函数产物。

目前，CATLASS Python 主要对 TLA 框架提供 Python 封装，可以使用 `tla.make_tensor` 等 API 来描述 TLA 操作，完成算子的实现。未来，我们将继续扩展CATLASS Python，在全量支持TLA tensor抽象接口、矩阵向量计算等接口的基础之上，进行类似CATLASS 模板的封装，实现对齐CATLASS C++模板算子的开发体验。

## 文档

- [**快速开始**](docs/zh/quick_start.md)：安装与首个示例
- [**Kernel 开发指南**](docs/zh/kernel_development/index.md)：核心概念（语法约束、控制流、Layout、Tensor 接入、编译启动）与调测优化
- [**框架开发指南**](docs/zh/dsl_development/index.md)：环境搭建、构建测试、特性开发、进阶 CMake 调试
- [**API 文档**](docs/zh/api/index.md)：语言参考 / 程序生命周期参考与文档生成

## 快速开始

安装 CATLASS Python 并运行首个算子示例，请参阅[快速开始](docs/zh/quick_start.md)。

## SIMD 复合计算

在 `@tla.kernel` 的 `tla.vec.func(mode="simd")` 内通过 `tla.arch` 调用复合计算接口。下表中的 `i` 表示启用的输入通道。

| 接口 | 计算公式 |
| --- | --- |
| [`tla.arch.axpy(x, y, alpha)`](docs/zh/api/language/simd/arithmetic.md#tlaarchaxpy) | `out[i] = alpha * x[i] + y[i]` |
| [`tla.arch.sub_abs(lhs, rhs)`](docs/zh/api/language/simd/arithmetic.md#tlaarchsub_abs) | `out[i] = abs(lhs[i] - rhs[i])` |
| [`tla.arch.muls_cast(src, scalar, reg_slot=...)`](docs/zh/api/language/simd/arithmetic.md#tlaarchmuls_cast) | `out[2*i + slot] = FP16(src[i] * scalar)` |

`muls_cast` 以 FP32 计算乘积，再按 `CAST_ROUND` 转为 FP16。`reg_slot` 默认为 `tla.params.RegSlot.ZERO`，对应 `slot = 0`，写入偶数槽位。`tla.params.RegSlot.ONE` 对应 `slot = 1`，写入奇数槽位。只读取启用输入通道对应的输出槽位。

三个接口都接受按输入元素类型构造的 `mask`。省略 `mask` 时启用有效输入范围内的通道。`axpy` 和 `sub_abs` 将未启用的通道置零。

## 兼容性

CATLASS Python 各版本支持的硬件平台及所需的最低 CANN 版本如下表：

| CATLASS Python 版本 | 最低支持 CANN 包版本 | 支持昇腾产品 |
| --- | --- | --- |
| 当前 | `>= 9.2.0` | Ascend 950PR&950DT系列产品 |

软硬件环境要求：

- CPU 架构：`aarch64` / `x86_64`
- 系统：CANN 支持的 Linux
- 软件依赖：Python `>= 3.10, < 3.14`、CMake `>= 3.28, < 4.0`、Ninja `>= 1.12`、Clang / Clang++ `>= 10`（构建 AscendNPU-IR 时；推荐 19）、lld、lit、FileCheck 与 LLVM 配套、AscendNPU-IR `feature/regbase@a07821269…`

完整的环境要求与安装方式见[环境准备](docs/zh/dsl_development/build_guide/environment.md)。

## 目录概览

```text
python/tla_dsl/
├── 3rdparty/AscendNPU-IR/  # AscendNPU-IR 子模块
├── catlass/                # Python 前端与运行时
├── csrc/mlir/              # TLA Dialect 与编译器实现
├── docs/                   # 文档
├── tests/                  # 单元、lit 与端到端测试
├── Dockerfile              # 开发环境镜像定义
├── build_docker_image.sh   # 镜像构建入口
├── build.sh                # 项目构建入口
└── requirements.txt        # 开发 Python 依赖
```

算子样例与教学示例位于仓库根目录的 [examples/python/tla_dsl/](../../examples/python/tla_dsl/README.md)：

```text
examples/python/tla_dsl/
├── tla/kernel/             # TLA 算子样例（matmul / attention / conv / gmm / vec）
├── dsl_tutorials/          # Python 语言特性与向量 / SIMT / 标量教学示例
├── experimental/           # 实验性算子样例，目录结构与 tla/ 一致
└── helpers/                # 示例共用的参数、精度比对与工具
```
