"""data.synthetic 单测：确定性、结构与标签。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import EmptyGraphError


def test_sbm_deterministic() -> None:
    """同 seed 两次生成的邻接矩阵逐元素一致。"""
    from graphforge.data.synthetic import generate_sbm

    g1 = generate_sbm(n_blocks=2, block_size=10, random_state=42)
    g2 = generate_sbm(n_blocks=2, block_size=10, random_state=42)
    assert (g1.adjacency != g2.adjacency).nnz == 0
    assert g1.node_ids == g2.node_ids


def test_sbm_structure_and_labels() -> None:
    """SBM 对称、无自环、标签在社区数范围内。"""
    from graphforge.data.synthetic import generate_sbm

    graph = generate_sbm(n_blocks=3, block_size=10, random_state=7)
    assert graph.num_nodes == 30
    assert graph.has_labels
    labels = np.asarray(graph.node_labels)
    assert labels.min() >= 0 and labels.max() <= 2
    assert (graph.adjacency.diagonal() == 0).all()
    assert (graph.adjacency != graph.adjacency.T).nnz == 0


def test_karate_is_zachary() -> None:
    """空手道俱乐部：34 节点 78 边、两类标签。"""
    from graphforge.data.synthetic import generate_karate

    graph = generate_karate()
    assert graph.num_nodes == 34
    assert graph.num_edges == 78
    assert set(np.unique(np.asarray(graph.node_labels)).tolist()) <= {0, 1}


def test_grid_structure() -> None:
    """10×10 网格：100 节点、180 条边、无标签。"""
    from graphforge.data.synthetic import generate_grid

    graph = generate_grid(m=10, n=10, random_state=42)
    assert graph.num_nodes == 100
    assert graph.num_edges == 180


def test_lfr_falls_back_on_failure() -> None:
    """LFR 参数苛刻（超时/迭代超限）时回退 SBM，且不抛异常。"""
    from graphforge.data.synthetic import generate_lfr

    graph = generate_lfr(n=60, tau1=3.0, tau2=2.0, mu=0.9, min_community=5, random_state=42)
    assert graph.num_nodes == 60
    assert graph.has_labels


def test_unknown_generator_raises() -> None:
    """未知生成器 → E305。"""
    from graphforge.core.errors import UnknownMethodError
    from graphforge.data.synthetic import generate

    with pytest.raises(UnknownMethodError):
        generate("no-such-graph")
