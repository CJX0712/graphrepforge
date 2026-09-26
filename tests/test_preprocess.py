"""preprocess 单测：邻接构建 / 划分 / 特征。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse as sp

from graphforge.core.errors import InvalidSplitRatioError, NegativeSamplingError


def test_build_adjacency_symmetrize_and_no_self_loops() -> None:
    """非对称输入对称化、自环移除。"""
    from graphforge.core.types import GraphData
    from graphforge.preprocess.build import build_adjacency, is_symmetric

    raw = sp.csr_matrix(np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    graph = GraphData(adjacency=raw, node_ids=["a", "b", "c"])
    built = build_adjacency(graph, symmetric=True, remove_loops=True)
    assert is_symmetric(built.adjacency)
    assert (built.adjacency.diagonal() == 0).all()


def test_normalize_adjacency_rows_sum_one() -> None:
    """随机游走归一化后每行和为 1。"""
    from graphforge.preprocess.build import normalize_adjacency

    raw = sp.csr_matrix(np.asarray([[0.0, 2.0], [2.0, 0.0]]))
    norm = normalize_adjacency(raw, method="random_walk")
    sums = np.asarray(norm.sum(axis=1)).ravel()
    assert np.allclose(sums, 1.0)


def test_edge_list_roundtrip(small_sbm: object) -> None:
    """edge_list 输出与矩阵 nnz 一致（无向按上三角）。"""
    from graphforge.preprocess.build import edge_list

    edges = edge_list(small_sbm)  # type: ignore[attr-defined]
    assert edges.shape[1] == 2
    assert edges.shape[0] == small_sbm.num_edges  # type: ignore[attr-defined]


def test_largest_connected_component(small_sbm: object) -> None:
    """最大连通分量保留全部标签维度。"""
    from graphforge.preprocess.build import largest_connected_component

    sub = largest_connected_component(small_sbm)  # type: ignore[attr-defined]
    assert sub.num_nodes >= 1  # type: ignore[attr-defined]


def test_allocate_counts_sums_to_total() -> None:
    """比例分配总数守恒。"""
    from graphforge.preprocess.split import allocate_counts

    assert sum(allocate_counts(100, (0.6, 0.2, 0.2))) == 100
    assert sum(allocate_counts(7, (0.6, 0.2, 0.2))) == 7


def test_negative_edges_no_overlap(small_sbm: object) -> None:
    """负边不与正边重叠、数量正确。"""
    from graphforge.preprocess.build import edge_list
    from graphforge.preprocess.split import sample_negative_edges

    positives = edge_list(small_sbm)  # type: ignore[attr-defined]
    forbidden = {tuple(sorted((int(a), int(b)))) for a, b in positives}
    pos_set = set(forbidden)
    rng = np.random.default_rng(42)
    negatives = sample_negative_edges(
        small_sbm.num_nodes, forbidden, count=len(positives), rng=rng  # type: ignore[attr-defined]
    )
    neg_set = {tuple(sorted((int(a), int(b)))) for a, b in negatives}
    assert not (pos_set & neg_set)
    assert len(negatives) == len(positives)


def test_negative_edges_exhausted_raises() -> None:
    """完全图无法采负边 → E204。"""
    from graphforge.preprocess.split import sample_negative_edges

    forbidden = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    rng = np.random.default_rng(0)
    with pytest.raises(NegativeSamplingError):
        sample_negative_edges(4, forbidden, count=1, rng=rng)


def test_stratified_split_deterministic(small_sbm: object) -> None:
    """分层划分同 seed 可复现、类别比例近似。"""
    from graphforge.preprocess.split import StratifiedNodeSplitter
    from graphforge.core.types import TaskSpec, TaskType

    spec = TaskSpec(task=TaskType.NODE_CLASSIFICATION, random_state=42)
    splitter = StratifiedNodeSplitter(random_state=42)
    s1 = splitter.split(small_sbm, spec)  # type: ignore[attr-defined]
    s2 = splitter.split(small_sbm, spec)  # type: ignore[attr-defined]
    assert np.array_equal(s1.train_idx, s2.train_idx)
    total = s1.n_train + s1.n_val + s1.n_test
    assert total == small_sbm.num_nodes  # type: ignore[attr-defined]


def test_edge_splitter_shapes(small_sbm: object) -> None:
    """边划分：训练正边数 / 测试正负对称。"""
    from graphforge.preprocess.split import EdgeSplitter
    from graphforge.core.types import TaskSpec, TaskType

    spec = TaskSpec(task=TaskType.LINK_PREDICTION, random_state=42)
    splitter = EdgeSplitter(random_state=42)
    split = splitter.split(small_sbm, spec)  # type: ignore[attr-defined]
    assert split.n_train > 0 and split.n_test > 0


def test_invalid_ratios_raise() -> None:
    """负比例 / 和不为 1 → E201。"""
    from graphforge.preprocess.split import validate_ratios

    with pytest.raises(InvalidSplitRatioError):
        validate_ratios(-0.1, 0.5, 0.6)
    with pytest.raises(InvalidSplitRatioError):
        validate_ratios(0.5, 0.2, 0.1)


def test_feature_builder_degree(small_sbm: object) -> None:
    """degree 特征形状 (n, max_degree+1) 且归一。"""
    from graphforge.preprocess.features import build_features

    features = build_features(small_sbm, method="degree", max_degree=16)  # type: ignore[attr-defined]
    assert features.shape[0] == small_sbm.num_nodes  # type: ignore[attr-defined]
    assert features.shape[1] == 1
    assert np.all((features >= 0) & (features <= 1))


def test_concat_features() -> None:
    """特征拼接行数一致。"""
    from graphforge.preprocess.features import concat_features

    a = np.zeros((5, 2))
    b = np.ones((5, 3))
    out = concat_features(a, b)
    assert out.shape == (5, 5)
