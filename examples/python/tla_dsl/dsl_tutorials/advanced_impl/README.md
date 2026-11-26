# 进阶算法算子实现（advanced_impl）

本目录收录进阶算法算子和向量运算组合示例。

| 目录 | 概述 |
| --- | --- |
| [`sort_topk/`](sort_topk/README.md) | 基数排序与 TopK |
| [`mask_merging/`](mask_merging/README.md) | 使用 `tla.where` 合并运算结果与目的向量原值。 |

## 代码组织

```plain
./advanced_impl
├── mask_merging
│   ├── mask_merging.py
│   └── README.md
└── sort_topk
    ├── sort.py
    ├── topk.py
    └── README.md
```

## 运行方式

在 CATLASS 仓库根目录执行，参数详见各子目录的 README：

```bash
python examples/python/tla_dsl/dsl_tutorials/advanced_impl/sort_topk/sort.py --device 0
python examples/python/tla_dsl/dsl_tutorials/advanced_impl/sort_topk/topk.py --device 0 --k 8
python examples/python/tla_dsl/dsl_tutorials/advanced_impl/mask_merging/mask_merging.py
```
