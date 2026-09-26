"""训练器：节点分类 / 链路预测的特征构造、拟合与评估。

* 节点分类：特征 = 嵌入矩阵（或结构特征）→ 估计器 → ``accuracy`` / ``macro_f1``
* 链路预测：边特征（hadamard / average / l2 / concat）→ 估计器 → ``roc_auc`` / ``average_precision``

作者：晨星
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from graphforge.core.errors import EstimatorFitError, MetricComputationError
from graphforge.core.logging import get_logger
from graphforge.core.types import EDGE_OPS, Split, TaskSpec

from graphforge.training.cv import cross_val_scores, resolve_metric_fn
from graphforge.training.estimators import make_estimator

__all__ = [
    "edge_features",
    "fit_predict",
    "train_node_classifier",
    "train_link_predictor",
]

_LOGGER = get_logger(__name__)


def edge_features(matrix: np.ndarray, edges: np.ndarray, op: str = "hadamard") -> np.ndarray:
    """由节点嵌入构造边特征。

    Args:
        matrix: ``(n, d)`` 节点嵌入。
        edges: ``(m, 2)`` 边数组。
        op: 算子名，见 :data:`~graphforge.core.types.EDGE_OPS`。

    Returns:
        ``(m, d)`` 或 ``(m, 2d)``（concat）的边特征矩阵。

    Raises:
        MetricComputationError: 算子名不支持（复用 E405，属取值类错误）。
    """
    if op not in EDGE_OPS:
        raise MetricComputationError(
            f"边特征算子 {op!r} 不支持", hint=f"可选值：{list(EDGE_OPS)}"
        )
    array = np.asarray(matrix, dtype=np.float64)
    pairs = np.asarray(edges, dtype=np.int64)
    if pairs.size == 0:
        width = array.shape[1] * 2 if op == "concat" else array.shape[1]
        return np.zeros((0, width), dtype=np.float64)
    left = array[pairs[:, 0]]
    right = array[pairs[:, 1]]
    if op == "hadamard":
        return left * right
    if op == "average":
        return (left + right) / 2.0
    if op == "l2":
        return np.abs(left - right)
    return np.hstack([left, right])


def _positive_score(model: Any, features: np.ndarray) -> Optional[np.ndarray]:
    """取正类概率；``predict_proba`` 不可用时返回 None。"""
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return probabilities.ravel()
    return probabilities[:, 1]


def fit_predict(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec: TaskSpec,
    estimator_factory: Optional[Callable[[str, Dict[str, Any], int], Any]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """拟合估计器并在测试下标上预测。

    Args:
        features: ``(n, d)`` 特征。
        y: ``(n,)`` 标签。
        train_idx: 训练样本下标。
        test_idx: 测试样本下标。
        spec: 任务规格（提供估计器名与超参、随机种子）。
        estimator_factory: 自定义工厂；None 时用 :func:`make_estimator`。

    Returns:
        ``(y_pred, y_score)``，``y_score`` 可能为 None。

    Raises:
        EstimatorFitError: 拟合或预测失败（E402）。
    """
    factory = estimator_factory or make_estimator
    try:
        model = factory(spec.estimator, dict(spec.estimator_params), int(spec.random_state))
        model.fit(np.asarray(features)[train_idx], np.asarray(y)[train_idx])
        y_pred = np.asarray(model.predict(np.asarray(features)[test_idx]))
        y_score = _positive_score(model, np.asarray(features)[test_idx])
    except Exception as exc:  # noqa: BLE001 - sklearn 异常统一包成 E402
        raise EstimatorFitError(
            f"估计器 {spec.estimator!r} 拟合/预测失败：{exc}",
            hint="检查特征维度与标签取值",
        ) from exc
    return y_pred, y_score


def train_node_classifier(
    features: np.ndarray,
    y: np.ndarray,
    split: Split,
    spec: TaskSpec,
    estimator_factory: Optional[Callable[[str, Dict[str, Any], int], Any]] = None,
    metric_fn: Optional[Any] = None,
) -> Dict[str, float]:
    """训练并评估节点分类器。

    ``spec.use_cv`` 为 True 时在 train+val 上做 K 折交叉验证（返回各折均值）；
    否则在 train 上拟合、在 test 上评估。

    Args:
        features: ``(n, d)`` 节点特征（通常为嵌入矩阵）。
        y: ``(n,)`` 节点标签。
        split: 划分结果（下标指向节点）。
        spec: 任务规格。
        estimator_factory: 自定义估计器工厂（可选）。
        metric_fn: 指标函数；None 时延迟解析默认实现（见
            :func:`graphforge.training.cv.resolve_metric_fn`）。

    Returns:
        ``{指标名: 数值}``。

    Raises:
        MetricComputationError: 标签缺失或计算失败。
    """
    evaluate = resolve_metric_fn(metric_fn)
    scoring = spec.resolve_scoring()
    labels = np.asarray(y)
    if labels is None or labels.size == 0:
        raise MetricComputationError(
            "节点分类需要标签", hint="使用带标签的数据集（如 sbm / karate）"
        )
    array = np.asarray(features, dtype=np.float64)
    if array.shape[0] != labels.shape[0]:
        raise MetricComputationError(
            f"特征行数 {array.shape[0]} 与标签数 {labels.shape[0]} 不一致",
            hint="检查嵌入与图是否对齐",
        )
    if spec.use_cv:
        combined = np.concatenate([split.train_idx, split.val_idx])
        return cross_val_scores(
            array[combined],
            labels[combined],
            scoring=scoring,
            n_splits=spec.n_splits,
            random_state=spec.random_state,
            estimator=spec.estimator,
            estimator_params=dict(spec.estimator_params),
            metric_fn=evaluate,
        )
    y_pred, y_score = fit_predict(array, labels, split.train_idx, split.test_idx, spec, estimator_factory)
    return evaluate(labels[split.test_idx], y_pred, y_score, scoring)


def train_link_predictor(
    matrix: np.ndarray,
    edges: np.ndarray,
    y: np.ndarray,
    split: Split,
    spec: TaskSpec,
    estimator_factory: Optional[Callable[[str, Dict[str, Any], int], Any]] = None,
    metric_fn: Optional[Any] = None,
) -> Dict[str, float]:
    """训练并评估链路预测器。

    Args:
        matrix: ``(n, d)`` 节点嵌入。
        edges: ``(m, 2)`` 边数组（正边 + 负边）。
        y: ``(m,)`` 的 0/1 标签。
        split: 划分结果（下标指向 ``edges`` 的行）。
        spec: 任务规格（``edge_op`` 决定边特征算子）。
        estimator_factory: 自定义估计器工厂（可选）。
        metric_fn: 指标函数；None 时延迟解析默认实现。

    Returns:
        ``{指标名: 数值}``。

    Raises:
        MetricComputationError: 边数组为空或计算失败。
    """
    evaluate = resolve_metric_fn(metric_fn)
    scoring = spec.resolve_scoring()
    pairs = np.asarray(edges, dtype=np.int64)
    labels = np.asarray(y)
    if pairs.size == 0 or labels.size == 0:
        raise MetricComputationError(
            "链路预测需要边样本", hint="检查 EdgeSplitter 输出"
        )
    features = edge_features(matrix, pairs, spec.edge_op)
    if spec.use_cv:
        combined = np.concatenate([split.train_idx, split.val_idx])
        return cross_val_scores(
            features[combined],
            labels[combined],
            scoring=scoring,
            n_splits=spec.n_splits,
            random_state=spec.random_state,
            estimator=spec.estimator,
            estimator_params=dict(spec.estimator_params),
            metric_fn=evaluate,
        )
    y_pred, y_score = fit_predict(
        features, labels, split.train_idx, split.test_idx, spec, estimator_factory
    )
    return evaluate(labels[split.test_idx], y_pred, y_score, scoring)
