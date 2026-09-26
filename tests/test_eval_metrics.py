"""eval 单测：指标正确性（与 sklearn 直算一致）、方向、表格渲染。"""

from __future__ import annotations

import numpy as np


def test_metrics_match_sklearn() -> None:
    """宏 F1 / ROC-AUC 与 sklearn 直算一致（递归防护生效证明）。"""
    from sklearn.metrics import f1_score as _sk_f1_score
    from sklearn.metrics import roc_auc_score as _sk_roc_auc_score

    from graphforge.eval.metrics import macro_f1_score, roc_auc_score

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=60)
    y_pred = rng.integers(0, 3, size=60)
    assert abs(macro_f1_score(y_true, y_pred) - float(_sk_f1_score(y_true, y_pred, average="macro"))) < 1e-12
    y_bin = rng.integers(0, 2, size=60)
    scores = rng.uniform(size=60)
    assert abs(roc_auc_score(y_bin, scores) - float(_sk_roc_auc_score(y_bin, scores))) < 1e-12


def test_all_metrics_direction_plus_one() -> None:
    """全部注册指标方向 +1。"""
    from graphforge.eval.metrics import check_directions

    assert check_directions() is True


def test_compute_metrics_dispatch() -> None:
    """labels 型与 proba 型指标按 kind 分派输入。"""
    from graphforge.eval.metrics import compute_metrics

    y_true = np.asarray([0, 1, 1, 0])
    y_pred = np.asarray([0, 1, 0, 0])
    y_score = np.asarray([0.1, 0.9, 0.4, 0.2])
    metrics = compute_metrics(
        y_true, y_pred=y_pred, y_score=y_score, scoring=("accuracy", "roc_auc")
    )
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_rank_results_descending() -> None:
    """rank_results 按 primary 指标降序（越大越好）。"""
    from graphforge.core.types import EvalResult, TaskType
    from graphforge.eval.ranking import rank_results

    def _row(method: str, value: float) -> EvalResult:
        return EvalResult(
            task=TaskType.NODE_CLASSIFICATION,
            dataset="sbm",
            method=method,
            backend="internal",
            metrics={"macro_f1": value},
            primary_metric="macro_f1",
            primary_value=value,
            n_samples=10,
            elapsed_sec=0.0,
        )

    ranked = rank_results([_row("a", 0.5), _row("b", 0.9), _row("c", 0.7)])
    assert ranked[0]["rank"] == 1
    assert ranked[0]["row"].method == "b"
    values = [row["row"].primary_value for row in ranked]
    assert values == sorted(values, reverse=True)


def test_render_table_fixed_width() -> None:
    """render_table 各行等宽。"""
    from graphforge.core.types import EvalResult, TaskType
    from graphforge.eval.report import render_table

    row = EvalResult(
        task=TaskType.NODE_CLASSIFICATION,
        dataset="sbm",
        method="node2vec",
        backend="internal",
        metrics={"macro_f1": 0.9},
        primary_metric="macro_f1",
        primary_value=0.9,
        n_samples=100,
        elapsed_sec=0.1,
    )
    table = render_table([row])
    lines = table.splitlines()
    # 表头 + 分隔线 + 1 数据行；分隔线长度 = 列宽 × 列数
    assert len(lines) == 3
    assert len({len(line) for line in lines}) == 1
