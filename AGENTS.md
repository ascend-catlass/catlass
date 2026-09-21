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

### 常见术语与易错项检查清单

编写文档或提交前，请逐项确认以下常见术语与格式的正确性；如发现新的易错项，请及时补充到下方对应清单中。

#### 产品与术语写法

| 正确写法 | 常见错误写法 | 说明 |
| -------- | ------------ | ---- |
| `Atlas A2` / `Atlas A3` | `AtlasA2` / `AtlasA3` | 产品名，中间必须带空格 |
| `Ascend 950` / `Ascend 950PR` / `Ascend 950DT` | `Ascend950` / `Ascend950PR` | 产品名，中间必须带空格 |
| `AI Core` | `AICore` | 计算核心，中间必须带空格 |
| `Ascend C` | `AscendC` | 正文写法必须带空格；仅 C++ 命名空间 `AscendC::` 保持无空格原样 |
| `Flash Attention` | `FlashAttention` | 正文写法带空格；仅代码标识符/样例名（如 `FlashAttentionInfer`）保持原样 |
| `L2 Cache` | `L2Cache` | 缓存名称带空格；仅 profiling 产出文件名 `L2Cache.csv` 保持原样 |
| `Cube Core` | `Cube-core` | 不带连字符 |
| `PIPE_FIX` | `PIPE_FIXED` | Ascend C 流水类型枚举值；勿臆造 `PIPE_FIXED` |

#### 中文错别字与用词

| 正确写法 | 常见错误写法 | 说明 |
| -------- | ------------ | ---- |
| 显式补零 | 显示补零 | 「显式」表示明确指定，勿写成同音的「显示」 |
| 归约（ReduceAdd） | 规约 | 并行累加统一用「归约」 |
| 分形 | 分型 | L0/L1 上的 16×16、16×32 数据块统一称「分形」 |
| 起始地址 | 其实地址 | 「起始」勿写成同音的「其实」 |
| 受到影响 | 收到影响 | 「受到」勿写成同音的「收到」 |
| 微缩放（MX Scale） | 微缩缩放 | MX Scale 统一称「微缩放」 |
| 不再 | 不在 | 「不再（no longer）」勿写成同音的「不在」 |
| 该值 | 改值 | 「该值（this value）」勿写成同音的「改值」 |
| 在 … 上 | 再 … 上 | 表示位置或载体时用「在」，勿写成同音的「再」 |
| 不同的是 | 不同地 | 「与 … 不同的是」为固定搭配 |
| 简洁地描述 | 简洁的描述 | 修饰动词用副词「地」 |
| 使用 | 完成使用 | 去除冗余动词，如「在 workspace 上使用…」 |
| 详细使用示例…请参考 | 详细参考使用示例…请参考 | 去除重复的「参考」 |

#### 英文拼写易错项

| 正确写法 | 常见错误写法 | 说明 |
| -------- | ------------ | ---- |
| `utilities` / `utility` | `utilties` / `utilty` | 标题与正文中的单词拼写 |
| `TreeVisitor` | `TreeVistor` | EVG 访问者节点名 |
| `ColumnMajor` | `ColumMajor` | 矩阵布局名 |
| `DataCopy` | `Datacopy` | Ascend C 数据搬运接口名 |
| `GetTile` | `Getile` | TLA 取 Tile 接口名 |
| `loadDataParams` | `loadDataPrams` | 结构体变量名 |
| `block` | `bloc` | 正文单词拼写 |
| `width` | `wight` | 参数含义注释 |
| `true` | `ture` | 布尔字面量 |

其他易错项：

- **代码形式优先**：术语若位于代码块或行内代码（反引号）中，一律保持代码原样，不补空格、不拆分（如 `Arch::AtlasA2`、`AscendC::GlobalTensor`、`MmadAtlasA2Pingpong`、`FlashAttentionInfer`）；仅代码块之外的正文、标题、表格文字按上表更正。
- **代码块语言标记**：`json` 代码块内不得出现 `//` 注释（会导致解析失败）；语言标记须与内容一致（C++ 代码请用 `c++`）。
- **命令行参数**：必须使用 ASCII 连字符 `--`（如 `--details`），不得混入 en-dash `–` 等全角/破折号字符。
- **示例代码一致性**：文档中的示例代码须与可编译形式一致，模板参数、逗号与标识符拼写不得有误（如模板参数列表末尾逗号、`loadDataParams` 等）。
- **Markdown 结构**：标题层级逐级递增，不得从 H3 直接跳到 H5；有序列表编号连续，不得重复或跳号。

## 目录快速导航

| Path               | Description                            |
| ------------------ | -------------------------------------- |
| `examples/`        | 算子示例源码（00～120+）               |
| `include/catlass/` | 模板头文件（Kernel / Block / Tile）    |
| `docs/zh/`         | 中文文档：实践指南、设计总结、API 参考 |

详细文档见 [README.md](README.md)。
