# 进阶算法算子实现（advanced_impl）

本目录存放一些**进阶算法算子**的实现：完整的算子，而非单个指令或语言特性的演示。

| 目录 | 算法 |
| --- | --- |
| [`sort_topk/`](sort_topk/README.md) | 基数排序与 TopK |

## 代码组织

```plain
./advanced_impl
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
```
