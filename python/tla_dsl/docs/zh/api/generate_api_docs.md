---
nav_order: 20
---

# 生成 API 文档

CATLASS Python API 文档由脚本生成 Markdown，再由仓库根目录的 MkDocs 配置构建为
静态站点。Host 文档仅使用 AST 静态解析；Kernel 文档结合 AST 静态解析与运行时
docstring hook，后者补充 `copy` 搬运通路和矩阵计算数据类型。

- 语言参考：`core_api.py`、`tla/tensor.py` → `docs/en/api/language/`
- 程序生命周期参考：Host 源文件 → `docs/en/api/runtime/`

每份参考由一个入口页和若干主题页组成；SIMD 各子节拆为 `language/simd/` 下的独立页面。
Host 与 Kernel 是条目内的调用上下文标签，不再作为一级文档章节。

公共解析 / Markdown 渲染在 `tools/common.py`。

## 前置条件

Host 文档生成只需解析 Python 源码。Kernel 文档生成会导入 `catlass.core_api` 执行
docstring hook，因此还需要 MLIR Python bindings。生成或检查 Kernel 参考前，先完成
[Development 模式构建](../dsl_development/build_guide/build.md)，或安装包含这些 bindings
的软件包；`--check` 也需要相同环境。

构建 MkDocs 站点前安装文档依赖：

```bash
# /path/to/catlass 需替换为你 clone 的 CATLASS 仓库实际路径
cd /path/to/catlass/python/tla_dsl
python -m pip install -r requirements-docs.txt
```

## 生成 API Reference Markdown

源码 docstring 使用英文段落标题，并带目录标签：

- `Directory:`：该 API 在参考文档 TOC 中的路径（可用 `/` 表示嵌套）
- `Description:` / `Parameters:` / `Constraints:` / `Example:`

生成器根据 `Directory:` 建目录树；章节顺序与章节简介写在脚本的
`DIRECTORY_SECTIONS`（Kernel）与 `HOST_DIRECTORY_SECTIONS`（Host）。
同节内 API 顺序跟随源码定义顺序。

生成器产出英文参考页面（自动生成，勿手改）：

- `docs/en/api/language/index.md` 及 `docs/en/api/language/` 下各主题页（`tools/generate_kernel_api_reference.py`）
- `docs/en/api/runtime/index.md` 及 `docs/en/api/runtime/` 下各主题页（`tools/generate_host_api_reference.py`）

中文页面 `docs/zh/api/language/` 与 `docs/zh/api/runtime/` 为**手工维护**，采用相同页面结构；
英文页面变更后请同步翻译更新对应中文页面。

环境变量见 [环境变量](../kernel_development/core_concepts/env_vars.md)，**不由** 程序生命周期生成器扫描。

```bash
cd /path/to/catlass/python/tla_dsl
python tools/generate_kernel_api_reference.py
python tools/generate_host_api_reference.py
```

检查生成文件是否与当前代码一致，但不改写文件：

```bash
python tools/generate_kernel_api_reference.py --check
python tools/generate_host_api_reference.py --check
```

`--check` 在文件过期时输出 diff 并返回非零状态，适合用于提交前检查。
