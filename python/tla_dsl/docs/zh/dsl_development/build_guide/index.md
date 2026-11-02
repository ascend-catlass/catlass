---
nav_order: 10
---

# 环境准备与构建

本目录介绍如何准备开发环境、构建 CATLASS Python 并运行测试。

## 环境搭建

| 文档 | 范围 |
|------|------|
| [环境准备](environment.md) | 环境要求总表、依赖软件与环境检查。 |
| [Conda 安装](conda.md) | 使用仓库 `environment.yml` 创建开发环境（不含 CANN 与 AscendNPU-IR）。 |
| [Docker 安装](docker.md) | 使用仓库 `Dockerfile` 构建开发镜像，包含任何开发依赖。 |
| [构建 AscendNPU-IR](ascend_npu_ir.md) | 手动构建 Python 所依赖的 AscendNPU-IR。 |

## 构建与测试

| 文档 | 范围 |
|------|------|
| [构建 CATLASS Python](build.md) | `./build.sh` 构建（Development / Release 模式）、产物检查。 |
| [运行测试用例](testing.md) | pytest、lit test 与端到端验证。 |
