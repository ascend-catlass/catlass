---
nav_order: 50
---

# 环境变量

本文说明运行与编译 CATLASS Python 时可设置的环境变量。除特别注明外，变量名均以 `CATLASS_DSL_` 为前缀。

## 布尔开关的取值约定

部分变量为开关。下列取值（不区分大小写）均视为开启：

`1`、`true`、`yes`、`on`

下列取值均视为关闭：

`0`、`false`、`no`、`off`

下文表格中的“默认值”指未设置该变量时采用的取值。

---

## 编译与运行

### `CATLASS_DSL_CACHE`

控制是否启用编译缓存。开启后，在相同配置下可复用既有编译结果。

| 取值 | 含义 |
| --- | --- |
| 默认值：`1` | 启用缓存 |
| `0` | 禁用缓存，每次重新编译 |

---

### `CATLASS_DSL_CACHE_DIR`

指定编译缓存目录。

| 取值 | 含义 |
| --- | --- |
| 默认值：`$HOME/.cache/catlass` | 若已设置 `XDG_CACHE_HOME`，则为 `$XDG_CACHE_HOME/catlass` |
| 目录路径 | 将缓存写入该目录 |

禁用缓存时，该目录不会被长期使用。

---

### `CATLASS_DSL_FORCE_RECOMPILE`

控制是否强制重新编译（忽略既有缓存）。

| 取值 | 含义 |
| --- | --- |
| 默认值：`0` | 不强制；可命中缓存 |
| `1` | 强制重新编译 |

---

### `CATLASS_DSL_PRINT_IR`

控制是否在编译过程中打印中间表示（IR），用于诊断编译问题。

| 取值 | 含义 |
| --- | --- |
| 默认值：`0` | 不打印 |
| `1` | 打印编译过程中的中间 IR |

---

### `CATLASS_DSL_KEEP`

编译成功后，将指定产物**额外复制**到导出目录（见 `CATLASS_DSL_DUMP_DIR`）。
可指定多项，以英文逗号分隔，不区分大小写；无法识别的名称将被忽略。

每个取值对应一个产物，以产出该产物的流水线阶段命名。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 不额外复制；产物仍保留在缓存目录中 |
| `tla-ir` | 前端 TLA IR（`<kernel>.tlair.mlir`） |
| `tla-passes` | TLA pass 流水线逐 pass 的 IR dump（`<kernel>.tla-passes.mlir`）；该取值自动开启 `CATLASS_DSL_PRINT_IR` |
| `hivmc-input` | 送入 hivmc-a5 的 MLIR（`<kernel>.mlir`） |
| `kernel` | 设备侧编译产物（`.o`） |
| `all` | 导出上述全部产物 |

示例：`tla-ir,kernel` 或 `all`。

每个取值只启用其自身产物所需的工作，因此请求某个产物不会顺带产出其他产物。
旧取值 `ir` 与 `ir-debug` 已移除：`ir-debug` 同时代表两个产物，若保留为别名，会让未请求过程 dump 的调用方也被动开启 IR 打印。

---

### `CATLASS_DSL_DUMP_DIR`

`KEEP` 导出文件的目标目录。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 启用 `KEEP` 时导出到当前工作目录 |
| 目录路径 | 导出到该目录；若不存在则自动创建 |

说明：仅设置本变量而未启用 `KEEP` 时，不会导出任何产物；产物导出完全由 `KEEP` 的取值决定。

---

### `CATLASS_DSL_NPU_DEVICE`

在未使用 `torch_npu` 当前设备时，指定执行所用的 NPU 设备号。

| 取值 | 含义 |
| --- | --- |
| 默认值：`0` | 使用设备号 `0` |
| 非负整数（如 `0`、`1`） | 使用指定设备号 |

若环境中已加载可用的 `torch` / `torch_npu`，则优先采用 torch 当前设备，本变量不生效。

---

## 构建环境

### `CATLASS_DSL_PREBUILT_ASCENDNPU_IR`

本机已构建的 AscendNPU-IR 目录，供构建 Python 与运行测试使用。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 构建使用仓库内 `python/tla_dsl/3rdparty/AscendNPU-IR` |
| 目录路径 | 指向已构建的 AscendNPU-IR 目录，例如 `/opt/AscendNPU-IR` |

---

### `MLIR_TBLGEN_INCLUDE_DIR`

供 TableGen 生成与编译工具链使用的 MLIR 头文件目录。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 由构建脚本根据已解析的 AscendNPU-IR 安装目录自动设置 |
| 目录路径 | 手动指定该目录 |

一般无需修改。

---

### `CATLASS_DSL_CATLASS_INCLUDE_DIR`

查找 Catlass 头文件时使用的目录（构建相关）。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 由 CMake 依据 `CATLASS_INCLUDE_DIR` 或仓库布局自动推导 |
| 目录路径 | 手动指定头文件所在目录 |

一般无需修改。

---

## 其它常用环境变量

### `ASCEND_HOME_PATH`

昇腾 CANN 工具包安装路径（通常由 `set_env.sh` 设置）。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 依赖系统 `PATH` 查找编译器等工具 |
| 目录路径 | 在该安装树下查找编译工具 |

---

### `XDG_CACHE_HOME`

在未指定 `CATLASS_DSL_CACHE_DIR` 时，影响默认缓存目录。

| 取值 | 含义 |
| --- | --- |
| 默认值：空 | 缓存目录为 `$HOME/.cache/catlass` |
| 目录路径 | 默认缓存为该路径下的 `catlass` 子目录 |

---

### `PYTHONPATH` / `LD_LIBRARY_PATH`

分别为 Python 模块搜索路径与动态库搜索路径。按系统惯例以 `:` 分隔多个目录。应包含的具体路径见环境准备文档。
