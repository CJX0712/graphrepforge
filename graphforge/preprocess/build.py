"""邻接矩阵构建与规范化（对称化 / 去自环 / CSR 化 / 度归一化）。

下游（嵌入层、任务层）**只消费本模块产出的规范图**：
``(n, n)`` 的 float64 CSR，无向图对称、无自环。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp

from graphforge.core.errors import EmptyGraphError, InvalidAdjacencyError
from graphforge.core.logging import get_logger
from graphforge.core.types import GraphData
from graphforge.core.utils import (
    degree_vector,
    remove_self_loops,
    symmetrize_adjacency,
    to_csr,
)

__all__ = [
    "NORMALIZATIONS",
    "to_csr",
    "build_adjacency",
    "normalize_adjacency",
    "adjacency_from_edges",
    "edge_list",
    "degree_vector",
    "largest_connected_component",
    "is_symmetric",
]

_LOGGER = get_logger(__name__)

#: 支持的归一化方式。
NORMALIZATIONS: Tuple[str, ...] = ("none", "symmetric", "random_walk")


def is_symmetric(adjacency: Any, tol: float = 1e-8) -> bool:
    """判断稀疏矩阵是否对称。

    Args:
        adjacency: 输入矩阵。
        tol: 容差。

    Returns:
        对称返回 True。
    """
    matrix = to_csr(adjacency)
    diff = matrix - matrix.T
    if diff.nnz == 0:
        return True
    return bool(np.abs(diff.data).max() <= tol)


def normalize_adjacency(adjacency: Any, method: str = "none") -> sp.csr_matrix:
    """度归一化邻接矩阵。

    Args:
        adjacency: 输入邻接矩阵（CSR）。
        method: ``"none"`` 原样返回；``"symmetric"`` 返回 ``D^-1/2 A D^-1/2``；
            ``"random_walk"`` 返回 ``D^-1 A``。

    Returns:
        归一化后的 ``csr_matrix``。

    Raises:
        InvalidAdjacencyError: ``method`` 非法。
    """
    if method not in NORMALIZATIONS:
        raise InvalidAdjacencyError(
            f"归一化方式 {method!r} 非法", hint=f"可选值：{list(NORMALIZATIONS)}"
        )
    matrix = to_csr(adjacency)
    if method == "none":
        return matrix
    degrees = degree_vector(matrix)
    safe = np.where(degrees > 0, degrees, 1.0)
    if method == "random_walk":
        scale = 1.0 / safe
        return (sp.diags(scale) @ matrix).tocsr()
    inv_sqrt = 1.0 / np.sqrt(safe)
    diagonal = sp.diags(inv_sqrt)
    return (diagonal @ matrix @ diagonal).tocsr()


def build_adjacency(
    graph: GraphData,
    symmetric: bool = True,
    remove_loops: bool = True,
    normalize: str = "none",
) -> GraphData:
    """把任意 :class:`GraphData` 规范成"干净"的图。

    Args:
        graph: 输入图。
        symmetric: 是否对称化（无向图应设为 True）。
        remove_loops: 是否去除自环。
        normalize: 归一化方式，见 :data:`NORMALIZATIONS`。

    Returns:
        新的 :class:`GraphData`（结构不变，仅替换邻接矩阵）。

    Raises:
        EmptyGraphError: 规范化后无边。
    """
    adjacency = to_csr(graph.adjacency)
    if not sp.issparse(adjacency) or adjacency.ndim != 2:
        raise InvalidAdjacencyError("邻接矩阵必须是二维稀疏矩阵", hint="检查构图逻辑")
    if symmetric and not graph.is_directed:
        adjacency = symmetrize_adjacency(adjacency, method="max")
    if remove_loops:
        adjacency = remove_self_loops(adjacency)
    adjacency = normalize_adjacency(adjacency, normalize)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    if adjacency.nnz == 0:
        raise EmptyGraphError(
            f"图 {graph.name} 规范化后无边", hint="检查原始图或归一化参数"
        )
    _LOGGER.debug(
        "build_adjacency：%s → %d 节点 / %d 有向弧（normalize=%s）",
        graph.name,
        adjacency.shape[0],
        adjacency.nnz,
        normalize,
    )
    return GraphData(
        adjacency=adjacency,
        node_ids=list(graph.node_ids),
        node_labels=graph.node_labels,
        node_features=graph.node_features,
        is_directed=graph.is_directed,
        name=graph.name,
        metadata={
            **dict(graph.metadata),
            "normalized": normalize,
            "symmetrized": bool(symmetric and not graph.is_directed),
        },
    )


def adjacency_from_edges(
    num_nodes: int,
    edges: Sequence[Sequence[int]],
    weights: Optional[Sequence[float]] = None,
    directed: bool = False,
) -> sp.csr_matrix:
    """由边数组构造 CSR 邻接矩阵。

    Args:
        num_nodes: 节点总数。
        edges: ``(m, 2)`` 的边数组。
        weights: 边权；None 时全为 1.0。
        directed: 有向图时为 True（无向图会自动补反向边）。

    Returns:
        ``(n, n)`` 的 ``csr_matrix``。

    Raises:
        InvalidAdjacencyError: 边数组形状错误或下标越界。
    """
    array = np.asarray(edges, dtype=np.int64)
    if array.size == 0:
        return sp.csr_matrix((int(num_nodes), int(num_nodes)), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise InvalidAdjacencyError(
            f"边数组形状应为 (m, 2)，实际 {array.shape}", hint="检查边数组"
        )
    if array.min() < 0 or array.max() >= int(num_nodes):
        raise InvalidAdjacencyError(
            "边下标越界", hint=f"节点下标必须在 [0, {int(num_nodes) - 1}]"
        )
    values = np.ones(array.shape[0], dtype=np.float64) if weights is None else np.asarray(
        weights, dtype=np.float64
    )
    rows = array[:, 0]
    cols = array[:, 1]
    if not directed:
        rows = np.concatenate([rows, array[:, 1]])
        cols = np.concatenate([cols, array[:, 0]])
        values = np.concatenate([values, values])
    matrix = sp.coo_matrix(
        (values, (rows, cols)), shape=(int(num_nodes), int(num_nodes))
    ).tocsr()
    return symmetrize_adjacency(matrix, method="max") if not directed else matrix


def edge_list(graph: GraphData) -> np.ndarray:
    """返回无向图的边数组（只取上三角，``i < j``）。

    Args:
        graph: 输入图。

    Returns:
        ``(m, 2)`` 的 int 数组。
    """
    matrix = to_csr(graph.adjacency)
    coo = sp.triu(matrix, k=1).tocoo()
    return np.stack([coo.row, coo.col], axis=1).astype(np.int64)


def largest_connected_component(graph: GraphData) -> GraphData:
    """取最大连通分量（保持节点顺序与标签/特征同步裁剪）。

    Args:
        graph: 输入图。

    Returns:
        最大连通分量构成的 :class:`GraphData`。

    Raises:
        EmptyGraphError: 图为空。
    """
    from scipy.sparse.csgraph import connected_components  # 局部导入，避免顶层膨胀

    if graph.num_nodes == 0:
        raise EmptyGraphError("图为空，无法取连通分量", hint="检查数据集")
    matrix = to_csr(graph.adjacency)
    count, labels = connected_components(matrix, directed=graph.is_directed)
    if count <= 1:
        return graph
    sizes = np.bincount(labels, minlength=count)
    best = int(np.argmax(sizes))
    indices = np.flatnonzero(labels == best).astype(np.int64)
    _LOGGER.info(
        "取最大连通分量：%d/%d 个分量，保留 %d/%d 节点",
        count,
        count,
        indices.size,
        graph.num_nodes,
    )
    return graph.subgraph(indices)
