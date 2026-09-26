"""GraphForge 预处理层：邻接规范化、数据集划分、节点特征工程。

典型用法::

    graph = build_adjacency(get_dataset("sbm"), normalize="none")
    split = get_splitter("stratified", random_state=42).split(graph, spec)
    features = build_features(graph, method="degree")

作者：晨星
"""

from __future__ import annotations

from graphforge.preprocess import build as build
from graphforge.preprocess import features as features
from graphforge.preprocess import split as split

from graphforge.preprocess.build import (
    NORMALIZATIONS,
    adjacency_from_edges,
    build_adjacency,
    degree_vector,
    edge_list,
    is_symmetric,
    largest_connected_component,
    normalize_adjacency,
)
from graphforge.preprocess.features import (
    FEATURE_METHODS,
    NodeFeatureBuilder,
    build_features,
    concat_features,
    list_feature_methods,
)
from graphforge.preprocess.split import (
    SPLITTER_REGISTRY,
    EdgeSplitter,
    RandomNodeSplitter,
    StratifiedNodeSplitter,
    allocate_counts,
    get_splitter,
    list_splitters,
    sample_negative_edges,
    validate_ratios,
)

__all__ = [
    # 子模块
    "build",
    "split",
    "features",
    # 邻接构建
    "build_adjacency",
    "normalize_adjacency",
    "adjacency_from_edges",
    "edge_list",
    "degree_vector",
    "largest_connected_component",
    "is_symmetric",
    "NORMALIZATIONS",
    # 划分
    "StratifiedNodeSplitter",
    "RandomNodeSplitter",
    "EdgeSplitter",
    "SPLITTER_REGISTRY",
    "get_splitter",
    "list_splitters",
    "validate_ratios",
    "allocate_counts",
    "sample_negative_edges",
    # 特征
    "NodeFeatureBuilder",
    "build_features",
    "concat_features",
    "list_feature_methods",
    "FEATURE_METHODS",
]
