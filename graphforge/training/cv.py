"""交叉验证（``StratifiedKFold`` / ``KFold``，显式 ``random_state``）。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# --- sklearn 别名导入 ---
from sklearn.model_selection import KFold as _sk_KFold
from sklearn.model_selection import StratifiedKFold as _sk_StratifiedKFold

from graphforge.core.errors import InsufficientSamplesError
from graphforge.core.logging import get_logger
from graphforge.core.utils import derive_seed

from graphforge.training.estimators import make_estimator

__all__ = ["make_folds", "cross_val_scores", "resolve_metric_fn"]


def resolve_metric_fn(metric_fn: Optional[Any] = None) -> Any:
    """解析指标函数（**延迟**绑定 eval 层，保持 training 层只静态依赖 core）。

    调用方（graph 层）可显式注入 ``metric_fn``；未注入时在函数内部延迟导入
    :func:`graphforge.eval.metrics.compute_metrics`，因此 training 层**模块级**
    不会 import eval，层间依赖方向保持单向。

    Args:
        metric_fn: ``(y_true, y_pred, y_score, scoring) -> Dict[str, float]``；
            None 时使用默认实现。

    Returns:
        指标函数。
    """
    if metric_fn is not None:
        return metric_fn

    from graphforge.eval.metrics import compute_metrics  # 延迟导入

    def _wrapped(
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray],
        y_score: Optional[np.ndarray],
        scoring: Sequence[str],
    ) -> Dict[str, float]:
        return compute_metrics(y_true, y_pred=y_pred, y_score=y_score, scoring=scoring)

    return _wrapped

_LOGGER = get_logger(__name__)


def make_folds(
    y: Optional[np.ndarray],
    n_samples: int,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Any]:
    """生成 CV 折（有标签且每类样本够多时用分层折）。

    Args:
        y: 标签；None 时用普通 ``KFold``。
        n_samples: 样本数。
        n_splits: 折数。
        random_state: 随机种子。

    Returns:
        ``[(train_idx, test_idx), ...]``。

    Raises:
        InsufficientSamplesError: 样本数少于折数。
    """
    splits = max(2, int(n_splits))
    if n_samples < splits:
        raise InsufficientSamplesError(
            f"样本数 {n_samples} < 折数 {splits}", hint="降低 n_splits 或增大图规模"
        )
    seed = derive_seed(random_state, "split", "cv")
    if y is None:
        return list(
            _sk_KFold(n_splits=splits, shuffle=True, random_state=seed).split(
                np.zeros(n_samples)
            )
        )
    labels = np.asarray(y)
    counts = np.bincount(labels[labels >= 0]) if labels.size else np.zeros(0)
    if counts.size and counts.min() >= splits:
        return list(
            _sk_StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed).split(
                np.zeros(n_samples), labels
            )
        )
    _LOGGER.warning("存在样本数 < 折数的类别，CV 退化为 KFold")
    return list(
        _sk_KFold(n_splits=splits, shuffle=True, random_state=seed).split(np.zeros(n_samples))
    )


def cross_val_scores(
    features: np.ndarray,
    y: np.ndarray,
    scoring: Sequence[str] = ("accuracy", "macro_f1"),
    n_splits: int = 5,
    random_state: int = 42,
    estimator: str = "logistic_regression",
    estimator_params: Optional[Dict[str, Any]] = None,
    metric_fn: Optional[Any] = None,
) -> Dict[str, float]:
    """在给定特征/标签上做 K 折交叉验证，返回各折均值。

    Args:
        features: ``(n, d)`` 特征矩阵。
        y: ``(n,)`` 标签。
        scoring: 指标名序列。
        n_splits: 折数。
        random_state: 随机种子。
        estimator: 估计器名。
        estimator_params: 估计器超参。
        metric_fn: 指标函数；None 时延迟解析默认实现（见 :func:`resolve_metric_fn`）。

    Returns:
        ``{指标名: 各折均值}``。

    Raises:
        InsufficientSamplesError: 样本量不足。
    """
    evaluate = resolve_metric_fn(metric_fn)
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(y)
    if features.shape[0] != labels.shape[0]:
        raise InsufficientSamplesError(
            f"特征行数 {features.shape[0]} 与标签数 {labels.shape[0]} 不一致",
            hint="检查特征构造",
        )
    folds = make_folds(labels, int(features.shape[0]), n_splits=n_splits, random_state=random_state)
    accumulators: Dict[str, List[float]] = {name: [] for name in scoring}
    for train_index, test_index in folds:
        model = make_estimator(estimator, estimator_params or {}, random_state)
        model.fit(features[train_index], labels[train_index])
        y_pred = model.predict(features[test_index])
        y_score: Optional[np.ndarray] = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features[test_index])
            y_score = (
                probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities.max(axis=1)
            )
        metrics = evaluate(labels[test_index], y_pred, y_score, scoring)
        for name, value in metrics.items():
            accumulators.setdefault(name, []).append(float(value))
    return {name: float(np.mean(values)) for name, values in accumulators.items() if values}
