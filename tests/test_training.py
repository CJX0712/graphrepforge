"""training 单测：估计器构造、防泄漏、fit/predict。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import EstimatorBuildError


def test_make_estimator_logistic() -> None:
    """logistic_regression 可构造、random_state 注入。"""
    from graphforge.training.estimators import make_estimator

    model = make_estimator("logistic_regression", {"C": 1.0}, random_state=42)
    assert model is not None


def test_make_estimator_rejects_unknown() -> None:
    """未知估计器 → E401。"""
    from graphforge.training.estimators import make_estimator

    with pytest.raises(EstimatorBuildError):
        make_estimator("no-such-model")


def test_make_estimator_rejects_leaked_params() -> None:
    """泄漏内部参数（n_classes）→ E401（三参分离契约）。"""
    from graphforge.training.estimators import make_estimator

    with pytest.raises(EstimatorBuildError):
        make_estimator("logistic_regression", {"n_classes": 3})


def test_train_node_classifier_deterministic(tiny_sbm: object) -> None:
    """节点分类训练 + 同 seed 复现。"""
    from graphforge.core.types import Split, TaskSpec, TaskType
    from graphforge.graph.embeddings.registry import get_embedder
    from graphforge.training.trainer import train_node_classifier

    graph = tiny_sbm
    embedding = get_embedder("spectral", dim=8, random_state=42).fit_transform(graph)  # type: ignore[attr-defined]
    labels = np.asarray(graph.node_labels)  # type: ignore[attr-defined]
    idx = np.arange(graph.num_nodes)  # type: ignore[attr-defined]
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split_at = int(0.7 * len(idx))
    split = Split(
        train_idx=idx[:split_at],
        val_idx=idx[:0],
        test_idx=idx[split_at:],
        y=labels,
    )
    spec = TaskSpec(task=TaskType.NODE_CLASSIFICATION, random_state=42)
    out1 = train_node_classifier(embedding.matrix, labels, split, spec)
    out2 = train_node_classifier(embedding.matrix, labels, split, spec)
    assert out1["accuracy"] == out2["accuracy"]
    assert 0.0 <= out1["accuracy"] <= 1.0


def test_edge_features_ops() -> None:
    """四种边特征算子形状与语义。"""
    from graphforge.training.trainer import edge_features

    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edges = np.asarray([[0, 1], [1, 2]])
    hadamard = edge_features(matrix, edges, op="hadamard")
    assert hadamard.shape == (2, 2)
    assert hadamard[0, 0] == 3.0
    assert edge_features(matrix, edges, op="average").shape == (2, 2)
    assert edge_features(matrix, edges, op="l2").shape == (2, 2)
    assert edge_features(matrix, edges, op="concat").shape == (2, 4)


def test_cross_val_scores(tiny_sbm: object) -> None:
    """CV 分数 keys = scoring 且有限。"""
    from graphforge.graph.embeddings.registry import get_embedder
    from graphforge.training.cv import cross_val_scores

    graph = tiny_sbm
    embedding = get_embedder("spectral", dim=8, random_state=42).fit_transform(graph)  # type: ignore[attr-defined]
    labels = np.asarray(graph.node_labels)  # type: ignore[attr-defined]
    scores = cross_val_scores(
        embedding.matrix,
        labels,
        scoring=("accuracy", "macro_f1"),
        n_splits=3,
        random_state=42,
    )
    assert set(scores) == {"accuracy", "macro_f1"}
    assert all(np.isfinite(list(scores.values())))
