# AGENTS.md — CATLASS

> **CA**NN **T**emplates for **L**inear **A**lgebra **S**ubroutine**s**

昇腾算子模板库，提供高性能矩阵乘类算子基础模板与示例。

## Available Skills

| Skill                           | Description                                               | Location                                        |
| ------------------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| `catlass-ascend950-migration`   | 将 CATLASS 算子从 AtlasA2/A3（2201）迁移到 Ascend950（3510）。    | `.agents/skills/catlass-ascend950-migration/`   |
| `catlass-example-to-pytest`     | 将 numbered CATLASS examples 接入 tests/optest 测试框架。 | `.agents/skills/catlass-example-to-pytest/`     |
| `catlass-example-to-torch-intf` | 将 catlass 示例迁移为 PyTorch extension 接口。            | `.agents/skills/catlass-example-to-torch-intf/` |
| `catlass-tile-level-ut`         | 为 Gemm/Tile 拷贝组件（CopyGmToL1 等及 TLA 变体）编写单元测试。 | `.agents/skills/catlass-tile-level-ut/`         |
| `catlass-docs-translation`      | 用 Git 增量更新中英文 Markdown 文档。                    | `.agents/skills/catlass-docs-translation/`      |

## 文档术语与写作规范

- 产品与项目名保持原样：`CATLASS`、`CANN`、`TLA`、`Ascend`、`Atlas A2/A3`、`Ascend 950`；不使用代号式替换。
- API、类、函数、宏、文件路径、命令、环境变量、URL、锚点和代码标识符保持原样；不要翻译或改写。
- 术语沿用同一篇及相邻文档的既有英文表达；没有可靠先例时使用简洁技术英语，不凭空扩展缩写或命名。
- 英文文档使用清晰的技术说明语气；保留原有 Markdown 层级、链接结构和示例格式。

## 目录快速导航

| Path               | Description                            |
| ------------------ | -------------------------------------- |
| `examples/`        | 算子示例源码（00～120+）               |
| `include/catlass/` | 模板头文件（Kernel / Block / Tile）    |
| `docs/zh/`         | 中文文档：实践指南、设计总结、API 参考 |

详细文档见 [README.md](README.md)。
