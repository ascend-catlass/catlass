---
name: catlass-docs-consistency
description: 检测并验证 CATLASS 文档与代码是否一致：静态检查构建目标、运行参数、仓库路径、include 与 API 符号，也可按代际 profile 真实编译并执行文档中的命令。修改文档或代码后、评审 PR、或需要按文档实操验证时触发。
---

# CATLASS 文档-代码一致性检测

本技能负责“文档写的内容在当前代码里是否仍然成立”，结果精确到 `文件:行` 与所在小节。仓库语言与写作规范见根 [`AGENTS.md`](../../../AGENTS.md)。

提供两种模式：

- **静态检测**：只读扫描，不需要工具链与设备，可全量或按改动增量执行。
- **上板检测**：按文档中的完整命令真实编译、运行，需要 CANN 与对应代际的设备；`--build-only` 可只编译。

## 触发条件

满足任一条件即执行本技能：

- 新增、修改、删除 `docs/**`、`examples/**`、`python/**` 下的 Markdown，或根目录 `README*.md`。
- 修改 `examples/**/CMakeLists.txt`、`scripts/build.sh`、`include/**` 等被文档引用的代码。
- 评审 PR 时需要确认“文档是否跟随代码更新”。
- 用户要求检查文档中的命令、路径、头文件或 API 是否可用。
- 用户需要按文档在真实设备上走一遍编译与运行流程。

不触发：仅修改代码且不涉及文档引用；纯外链可达性（交给 `scripts/check_md_links.py`）；文档语言与用词（交给 `catlass-docs-review`）。

## 模式与推荐场景

| 场景 | 模式 | 典型命令 |
| --- | --- | --- |
| 改完文档立即自检，快速定位失效引用 | 静态，增量 | `check_docs.py --changed` |
| 提交或评审前核对全部文档 | 静态，全量 | `check_docs.py` |
| 改了示例源码、构建脚本，确认文档命令仍能编译 | 上板，只编译 | `check_docs.py --mode onboard --profile <代际> --build-only` |
| 按文档全流程实操验证（需对应代际设备） | 上板，全量 | `check_docs.py --mode onboard --profile <代际>` |
| 只抽查少数示例 | 上板，筛选 | `check_docs.py --mode onboard --profile <代际> --examples 00_basic_matmul,16_*` |
| 先看会执行哪些命令，不真正运行 | 上板，dry-run | `check_docs.py --mode onboard --profile <代际> --dry-run` |

`check_docs.py` 位于本技能 `scripts/` 下；在仓库内也可用等价入口 `python3 scripts/check_docs_consistency.py`。技能独立安装时，用 `--repo-root` 指向 CATLASS 检出目录：

```bash
python3 <skill>/scripts/check_docs.py --repo-root /path/to/catlass --changed
```

未显式指定时，仓库根按 `CATLASS_REPO_ROOT` 环境变量、再由当前目录的 `git rev-parse --show-toplevel` 推导。

## 快速使用

```bash
# 全量静态检测（默认模式）
python3 scripts/check_docs_consistency.py

# 只检测工作区改动影响的文档（含代码改动反向定位到文档）
python3 scripts/check_docs_consistency.py --changed

# 只检测相对某个基线改动影响的文档（用于 PR）
python3 scripts/check_docs_consistency.py --base upstream/master

# 指定文件；Markdown 直接检查，代码文件用于反查受影响的文档
python3 scripts/check_docs_consistency.py docs/zh/1_Practice/01_quick_start.md
python3 scripts/check_docs_consistency.py include/catlass/gemm/kernel/basic_matmul.hpp

# 只列出会检查哪些文档，不执行检测
python3 scripts/check_docs_consistency.py --base upstream/master --list-docs

# 上板检测：默认代际 a2a3，另支持 ascend950
python3 scripts/check_docs_consistency.py --mode onboard --profile a2a3
python3 scripts/check_docs_consistency.py --mode onboard --profile ascend950 --build-only
python3 scripts/check_docs_consistency.py --mode onboard --profile a2a3 --dry-run
```

退出码：`0` 表示未发现问题，`1` 表示发现问题，`2` 表示参数或运行环境错误（如未知代际、`ASCEND_HOME_PATH` 未设置、预检失败）。

输出格式：

```text
docs/zh/1_Practice/02_host_example_assembly.md:392: [build-target] build target 'basic_matmul' is not defined by any CMakeLists.txt [section: 编译]
docs/zh/1_Practice/01_quick_start.md:131: [onboard-run] documented run command failed (exit 1): ... [section: 执行示例]
```

## 静态检测项

| 检查项 | 判定方式 | 典型失败 |
| ------ | -------- | -------- |
| `build-target` | 文档中 `bash scripts/build.sh <target>` 的 target 必须出现在某个 `CMakeLists.txt` 的目标定义中，或属于 `build.sh` 内置目标 | 示例改名后文档仍写旧 target |
| `run-binary` | `./<binary>` / `output/bin/<binary>` 必须是 CMake 目标；示例 README 只能引用本示例目录下的可执行文件 | README 复制自其他示例，二进制名未改 |
| `run-arity` | 运行命令的位置参数个数必须落在示例 `Parse`/`ParseArguments`/`main` 的 `argc` 约束内 | 参数解析新增必选参数后，README 命令未同步 |
| `repo-path` | 文档中的 `docs/`、`examples/`、`include/`、`scripts/`、`tests/`、`python/` 等路径必须存在；支持示例目录内相对路径 | 文件重命名或目录调整后路径失效 |
| `include-path` | C++ 代码块中 `#include "catlass/..."`、`"tla/..."` 必须能解析到真实头文件 | 头文件改名后指南代码块未更新 |
| `api-symbol` | `docs/zh/3_API/**`、`docs/en/3_API/**` 代码块中 `class`/`struct` 声明的符号必须存在于 `include/` | 公共模板删除或改名后 API 参考未更新 |
| `qualified-symbol` | 其他文档代码块中的 `Catlass::...::Type` 限定名必须能在 `include/` 中找到同名标识符 | 接口改名后开发指南中的限定名失效 |

命令参数个数从示例源码静态推导：优先 `ParseArguments`，其次 `Parse`，再 `main`，最后公共 `examples/common/options.hpp` 中的 Options 结构；无法可靠推导时跳过该命令的参数个数检查，不产生误报。

## 代际 profile

代际参数集中在技能根目录的 [`profiles.json`](profiles.json)，包含 `arch`、产品别名、`build_flags`、设备预检命令、超时与额外环境变量。

| profile | arch | 适用产品 | 状态 |
| --- | --- | --- | --- |
| `a2a3` | `2201` | Atlas A2系列产品、Atlas A3系列产品 | 可用 |
| `ascend950` | `3510` | Ascend 950PR&950DT系列产品 | 可用 |
| `ascend960` | 待定 | Ascend 960 | 预留，`enabled: false` |

新增代际只需补全 `profiles.json` 的 `arch`、别名与 `build_flags` 并将 `enabled` 置为 `true`，无需改代码。启用前使用该 profile 会明确提示“预留未启用”。

## 上板执行规则

- **执行对象**：同一小节中成组的 `bash scripts/build.sh <target>` 与 `cd output/bin && ./<binary> <args>`。命令按文档原样执行，用于验证文档本身是否可照做。
- **文档优先级**：`docs/**`、根目录文档以及示例下的非 README 指南视为 guide，`examples/**/README*.md` 视为 readme。任务按 guide 优先执行与汇总，README 同样覆盖；同一命令在两类文档重复时以 guide 文档为主引用。
- **代际过滤**：文档“支持产品型号 / Supported products”按 profile 别名匹配；文档显式写了 `-DCATLASS_ARCH` 且与 profile 不一致时跳过；未声明产品型号的示例默认跳过，可用 `--include-undeclared` 纳入。文档未写 arch 时按 profile 自动补齐编译参数。
- **不执行的命令**：含 `<...>`、`$VAR` 等占位符的命令，以及重定向命令。片段式 C++ 代码块不做执行，仍由静态模式的 include 与符号检查覆盖。
- **输出校验**：`--verify-output auto`（默认）只对 guide 文档校验“执行结果”文本块，且仅校验不含数字、表格等随版本变化内容的稳定输出；`all` 校验全部，`none` 关闭。
- **构建目录**：复用仓库的 `build/` 与 `output/`；切换 profile 时自动追加 `--clean`，也可用 `--clean` / `--no-clean` 控制。任务日志写入 `build/docs-onboard/<profile>/<target>.log`。
- **失败分类**：构建失败、运行失败、超时、输出缺失分别记为 `onboard-build`、`onboard-run`、`onboard-timeout`、`onboard-output`；环境问题以退出码 `2` 单独返回，不计入文档问题。
- `--fail-on guide` 可让只有 guide 文档失败时才返回 `1`，README 失败仍会打印并计入汇总；默认 `all`。
- `--json` 输出机器可读结果，每条含 `doc`、`line`、`kind`、`status`、`phase`、`command`、`exit_code`。

## 结果解读与修复

- 以代码为准修改文档，不为了让检查通过而放宽代码或改示例命名。
- 同一错误在中文、英文文档中同时存在时，一并修复。
- 文档确实在描述“新建示例/新建文件”时，检查器会识别“创建/新建/新增/create”所在行的路径并跳过；该措辞仅用于真实的创建型文档，不用于绕过检查。
- 构建产物（路径中包含 `build/`）与 `output/bin` 属于用户生成内容，不参与路径存在性检查。
- 上板失败先看 `build/docs-onboard/<profile>/<target>.log`，再确认是否为设备占用、环境变量或代际不匹配等环境因素。
- 修复后重新执行同一命令，确认问题数归零。

## 与其他检查的边界

| 检查项 | 负责方 |
| ------ | ------ |
| 相对链接、锚点、空白、换行、YAML/JSON 语法 | `pre-commit`，其中链接与锚点由 `scripts/check_md_links.py` 完成 |
| 文档语言、术语、错别字、量化与来源 | `catlass-docs-review` |
| `docs/zh/**` 改动后的英文同步 | `catlass-docs-translation` |
| 文档命令、路径、头文件、API 符号与代码的一致性 | 本技能（静态） |
| 文档命令在真实代际设备上的可执行性 | 本技能（上板） |

## 维护

脚本改动后至少执行：`python3 -m pytest tests/python/test_docs_consistency.py`、一次全量静态检测、以及 `--mode onboard --dry-run` 检查任务计划。仓库 `pre-commit` 不覆盖 `.agents/` 目录，脚本的 `ruff check` / `ruff format` 需手动运行。
