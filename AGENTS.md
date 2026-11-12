# AGENTS.md — CATLASS

> **CA**NN **T**emplates for **L**inear **A**lgebra **S**ubroutine**s**

昇腾算子模板库，提供高性能矩阵乘类算子基础模板与示例。

## Available Skills

| Skill                           | Description                                               | Location                                        |
| ------------------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| `catlass-ascend950-migration`   | 将 CATLASS 算子从 Atlas A2系列产品、Atlas A3系列产品（2201）迁移到 Ascend 950（3510）。    | `.agents/skills/catlass-ascend950-migration/`   |
| `catlass-example-to-pytest`     | 将 numbered CATLASS examples 接入 tests/optest 测试框架。 | `.agents/skills/catlass-example-to-pytest/`     |
| `catlass-example-to-torch-intf` | 将 catlass 示例迁移为 PyTorch extension 接口。            | `.agents/skills/catlass-example-to-torch-intf/` |
| `catlass-tile-level-ut`         | 为 Gemm/Tile 拷贝组件（CopyGmToL1 等及 TLA 变体）编写单元测试。 | `.agents/skills/catlass-tile-level-ut/`         |
| `catlass-docs-review`          | 审查文档语言与用词问题（术语、错别字、模糊量化、规格来源、数值单位），改文档后触发。 | `.agents/skills/catlass-docs-review/`           |
| `catlass-docs-translation`      | 改动文档后先判定待同步的英文文档与行号，再按 Git 增量更新。 | `.agents/skills/catlass-docs-translation/`      |
| `catlass-docs-consistency`      | 检测文档与代码一致性：静态检查构建目标、运行参数、路径、include 与 API 符号，也可按代际上板执行文档命令。 | `.agents/skills/catlass-docs-consistency/`      |

## 文档改动后的语言审查

新增、修改或删除任意 `*.md` 后，提交前调用 `catlass-docs-review` skill 逐项检查并修复，再进入英文同步。

- 审查范围限定为改动 hunk 及受其直接影响的同文档内容，不做无关润色。
- 该 skill 只处理固定规则无法覆盖的语言与用词问题；行尾空白、文件末尾换行等格式问题交给仓库 `pre-commit` 静态检查，相对链接与锚点交给 `scripts/check_md_links.py`。
- 新增易错项同步补入该 skill 的审查清单。

## 文档改动后的英文同步

修改 `docs/zh/**` 下的 Markdown 后，提交前调用 `catlass-docs-translation` skill，先输出待同步清单（每条为「英文文档 + 行号区间 + 改动摘要」），再按 skill 的块粒度增量翻译。

- 清单为空即无需同步；清单非空而未同步，不得提交。
- 配对关系由 skill 按仓库约定判定（`docs/zh/**` ↔ `docs/en/**`，非标准布局查同目录配对文件），不另建映射表。
- 行号取改动 hunk 展开后的完整 Markdown 块（标题节、段落、连续列表、表格、围栏代码块）在英文文档中的对应区间。

## 文档术语与写作规范

产品与项目名保持原样，API、路径、命令、标识符不翻译改写，英文沿用同一篇及相邻文档的既有表达并保留原有结构；完整规则与易错项清单见 [`catlass-docs-review`](.agents/skills/catlass-docs-review/SKILL.md) 的审查清单。改动文档时按该清单逐项检查。

## 贡献须知

为CATLASS进行贡献时，必须阅读[贡献指南](CONTRIBUTING.md)中的指导说明，如Issue规范及欢迎贡献的代码范围。禁止进行违反贡献指南的操作，如不按照模板提出Issue，或对当前不接受贡献的代码段进行修改。

提交 Issue 与 PR 前，须先读取 `.gitcode/ISSUE_TEMPLATE/` 下对应模板与 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`，按模板逐字段填写并核对；自查标准是“是否符合模板”，而不是“内容是否与草稿一致”。通过 API 直接提交的同样适用。

## 目录快速导航

| Path               | Description                            |
| ------------------ | -------------------------------------- |
| `examples/`        | 算子示例源码（00～120+）               |
| `include/catlass/` | 模板头文件（Kernel / Block / Tile）    |
| `docs/zh/`         | 中文文档：实践指南、设计总结、API 参考 |

详细文档见 [README.md](README.md)。
