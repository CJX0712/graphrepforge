"""GraphForge 数据层：合成图生成 + 文件载入 + 数据集注册表。

对外只需记住三个入口：
    * ``generate_sbm`` / ``generate_karate`` / ``generate_grid`` / ``generate_lfr``
    * ``load_edgelist`` / ``load_edge_csv`` / ``load_graph``
    * ``get_dataset(name, random_state=...)``

作者：晨星
"""

from __future__ import annotations

from graphforge.data import loaders as loaders
from graphforge.data import registry as registry
from graphforge.data import synthetic as synthetic

from graphforge.data.loaders import (
    attach_features,
    attach_labels,
    load_edge_csv,
    load_edgelist,
    load_feature_csv,
    load_graph,
    load_label_csv,
)
from graphforge.data.registry import (
    DATASET_REGISTRY,
    get_dataset,
    has_dataset,
    list_datasets,
    register_dataset,
)
from graphforge.data.synthetic import (
    SYNTHETIC_GENERATORS,
    generate,
    generate_grid,
    generate_karate,
    generate_lfr,
    generate_sbm,
    list_synthetic,
)

__all__ = [
    # 子模块
    "synthetic",
    "loaders",
    "registry",
    # 合成图
    "generate_sbm",
    "generate_karate",
    "generate_grid",
    "generate_lfr",
    "generate",
    "list_synthetic",
    "SYNTHETIC_GENERATORS",
    # 载入器
    "load_edgelist",
    "load_edge_csv",
    "load_label_csv",
    "load_feature_csv",
    "load_graph",
    "attach_labels",
    "attach_features",
    # 注册表
    "DATASET_REGISTRY",
    "register_dataset",
    "list_datasets",
    "get_dataset",
    "has_dataset",
]
