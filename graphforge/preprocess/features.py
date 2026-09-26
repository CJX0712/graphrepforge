"""节点特征工程：为"无嵌入基线"与特征拼接提供结构化特征。

可用方法（``NodeFeatureBuilder.METHODS``）：
    * ``none`` —— 空特征 ``(n, 0)``（仅作占位，不能直接喂给估计器）；
    * ``degree`` —— 度（``(n, 1)``，按最大度归一化到 ``[0, 1]``）；
    * ``onehot_degree`` —— 度的 one-hot（``(n, max_degree + 1)``，超出部分并入最后一维）；
    * ``clustering`` —— 局部聚类系数（``(n, 1)``，需 networkx，延迟导入）。

作者：晨星
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from graphforge.core.errors import UnknownMethodError
from graphforge.core.logging import get_logger
from graphforge.core.types import GraphData
from graphforge.core.utils import degree_vector

__all__ = [
    "FEATURE_METHODS",
    "NodeFeatureBuilder",
    "build_features",
    "concat_features",
    "list_feature_methods",
]

_LOGGER = get_logger(__name__)

#: 全部特征方法名。
FEATURE_METHODS: Tuple[str, ...] = ("none", "degree", "onehot_degree", "clustering")


class NodeFeatureBuilder:
    """节点特征构造器。

    Attributes:
        method: 特征方法名。
        max_degree: onehot 的度上限（超过的并入最后一维）。
    """

    METHODS = FEATURE_METHODS

    def __init__(self, method: str = "degree", max_degree: int = 32) -> None:
        """构造器。

        Args:
            method: 特征方法名，见 :data:`FEATURE_METHODS`。
            max_degree: onehot 编码的度上限。

        Raises:
            UnknownMethodError: 方法名未注册。
        """
        if method not in FEATURE_METHODS:
            raise UnknownMethodError(
                f"未知特征方法 {method!r}", hint=f"可选值：{list(FEATURE_METHODS)}"
            )
        self.method = str(method)
        self.max_degree = max(1, int(max_degree))

    def build(self, graph: GraphData) -> np.ndarray:
        """构造特征矩阵。

        Args:
            graph: 输入图。

        Returns:
            ``(n, d)`` 的 float64 特征矩阵。
        """
        num_nodes = int(graph.num_nodes)
        if self.method == "none":
            return np.zeros((num_nodes, 0), dtype=np.float64)
        degrees = degree_vector(graph.adjacency)
        if self.method == "degree":
            scale = float(degrees.max()) if degrees.size and degrees.max() > 0 else 1.0
            return (degrees / scale).reshape(num_nodes, 1).astype(np.float64)
        if self.method == "onehot_degree":
            clipped = np.minimum(degrees.astype(np.int64), self.max_degree)
            matrix = np.zeros((num_nodes, self.max_degree + 1), dtype=np.float64)
            matrix[np.arange(num_nodes), clipped] = 1.0
            return matrix
        # clustering：networkx 延迟导入
        import networkx as nx  # noqa: WPS433 - 仅本方法需要

        nx_graph = graph.to_networkx()
        coefficients = nx.clustering(nx_graph)
        values = np.asarray(
            [float(coefficients.get(node_id, 0.0)) for node_id in graph.node_ids],
            dtype=np.float64,
        )
        return values.reshape(num_nodes, 1)

    def dim(self, graph: GraphData) -> int:
        """返回该方法的特征维度（不构造完整矩阵时可用）。"""
        if self.method == "none":
            return 0
        if self.method == "degree" or self.method == "clustering":
            return 1
        return self.max_degree + 1


def build_features(
    graph: GraphData, method: str = "degree", max_degree: int = 32
) -> np.ndarray:
    """便捷函数：一步构造节点特征。

    Args:
        graph: 输入图。
        method: 特征方法名。
        max_degree: onehot 编码的度上限。

    Returns:
        ``(n, d)`` 特征矩阵。
    """
    return NodeFeatureBuilder(method=method, max_degree=max_degree).build(graph)


def concat_features(*blocks: Optional[np.ndarray]) -> np.ndarray:
    """横向拼接多个特征块（自动跳过 ``None`` 与 0 维块）。

    Args:
        *blocks: 特征矩阵（``(n, d_i)``），允许 ``None``。

    Returns:
        拼接后的 ``(n, sum(d_i))`` 矩阵；全部为空时返回 ``(n, 0)``。

    Raises:
        ValueError: 各块行数不一致。
    """
    usable: List[np.ndarray] = []
    for block in blocks:
        if block is None:
            continue
        array = np.asarray(block, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.shape[1] == 0:
            continue
        usable.append(array)
    if not usable:
        return np.zeros((0, 0), dtype=np.float64)
    rows = {int(array.shape[0]) for array in usable}
    if len(rows) != 1:
        raise ValueError(f"特征块行数不一致：{sorted(rows)}")
    return np.hstack(usable)


def list_feature_methods() -> List[str]:
    """返回全部特征方法名。"""
    return list(FEATURE_METHODS)
