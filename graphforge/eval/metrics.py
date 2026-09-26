"""指标注册表（**禁止自研指标**，一律复用 scikit-learn）。

两条铁律（架构 §10 坑 2）：
    1. sklearn 的导入**一律使用 ``_sk_`` 前缀别名**，例如
       ``from sklearn.metrics import f1_score as _sk_f1_score``，
       这样本模块即使定义同名函数也不会递归覆盖导入名；
    2. 本项目公开函数另起名字（``macro_f1_score`` 等），与 sklearn 原名区分。

方向约定：**全部指标 direction = +1（越大越好）**，便于跨任务排序比较。

作者：晨星
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- sklearn 别名导入（禁止与下方公开函数重名）---
from sklearn.metrics import accuracy_score as _sk_accuracy_score
from sklearn.metrics import average_precision_score as _sk_average_precision_score
from sklearn.metrics import f1_score as _sk_f1_score
from sklearn.metrics import roc_auc_score as _sk_roc_auc_score

from graphforge.core.errors import MetricComputationError, MissingMetricError
from graphforge.core.logging import get_logger
from graphforge.core.types import TaskType

__all__ = [
    "Scorer",
    "METRIC_REGISTRY",
    "accuracy_score",
    "macro_f1_score",
    "micro_f1_score",
    "weighted_f1_score",
    "roc_auc_score",
    "average_precision_score",
    "list_metrics",
    "get_metric",
    "compute_metrics",
    "default_scoring",
    "check_directions",
]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class Scorer:
    """指标描述。

    Attributes:
        name: 指标名。
        func: 计算函数 ``(y_true, y_out) -> float``。
        direction: 方向，**恒为 +1**（越大越好）。
        kind: ``"labels"`` 需要预测标签，``"proba"`` 需要预测概率/得分。
    """

    name: str
    func: Callable[[np.ndarray, np.ndarray], float]
    direction: int = 1
    kind: str = "labels"


# ---------------------------------------------------------------- 公开指标
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """分类准确率（越大越好）。

    Args:
        y_true: 真实标签。
        y_pred: 预测标签。

    Returns:
        准确率。
    """
    return float(_sk_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """宏平均 F1（越大越好）。"""
    return float(_sk_f1_score(np.asarray(y_true), np.asarray(y_pred), average="macro"))


def micro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """微平均 F1（越大越好）。"""
    return float(_sk_f1_score(np.asarray(y_true), np.asarray(y_pred), average="micro"))


def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """加权平均 F1（越大越好）。"""
    return float(_sk_f1_score(np.asarray(y_true), np.asarray(y_pred), average="weighted"))


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC（越大越好）。

    Args:
        y_true: 二分类真实标签。
        y_score: 正类概率或决策得分。

    Returns:
        ROC-AUC。

    Raises:
        MetricComputationError: 只存在单一类别时无法计算。
    """
    labels = np.asarray(y_true)
    if np.unique(labels).size < 2:
        raise MetricComputationError(
            "ROC-AUC 需要正负两类样本", hint="检查划分结果或负采样比例"
        )
    return float(_sk_roc_auc_score(labels, np.asarray(y_score)))


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision（PR 曲线下面积，越大越好）。"""
    labels = np.asarray(y_true)
    if np.unique(labels).size < 2:
        raise MetricComputationError(
            "Average Precision 需要正负两类样本", hint="检查划分结果或负采样比例"
        )
    return float(_sk_average_precision_score(labels, np.asarray(y_score)))


#: 指标注册表：``name -> Scorer``（方向恒 +1）。
METRIC_REGISTRY: Dict[str, Scorer] = {
    "accuracy": Scorer("accuracy", accuracy_score, 1, "labels"),
    "macro_f1": Scorer("macro_f1", macro_f1_score, 1, "labels"),
    "micro_f1": Scorer("micro_f1", micro_f1_score, 1, "labels"),
    "weighted_f1": Scorer("weighted_f1", weighted_f1_score, 1, "labels"),
    "roc_auc": Scorer("roc_auc", roc_auc_score, 1, "proba"),
    "average_precision": Scorer("average_precision", average_precision_score, 1, "proba"),
}


def list_metrics() -> List[str]:
    """返回全部指标名（升序）。"""
    return sorted(METRIC_REGISTRY)


def get_metric(name: str) -> Scorer:
    """按名字取指标。

    Args:
        name: 指标名。

    Returns:
        :class:`Scorer`。

    Raises:
        MissingMetricError: 指标名未注册（E406）。
    """
    if name not in METRIC_REGISTRY:
        raise MissingMetricError(
            f"未知指标 {name!r}", hint=f"可选值：{list_metrics()}"
        )
    return METRIC_REGISTRY[name]


def default_scoring(task: TaskType) -> Tuple[str, ...]:
    """返回任务的默认指标组合。

    Args:
        task: 任务类型。

    Returns:
        指标名元组。
    """
    if task is TaskType.LINK_PREDICTION:
        return ("roc_auc", "average_precision")
    return ("accuracy", "macro_f1")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
    scoring: Sequence[str] = ("accuracy",),
) -> Dict[str, float]:
    """按 ``scoring`` 计算全部指标。

    Args:
        y_true: 真实标签。
        y_pred: 预测标签（``kind="labels"`` 的指标需要）。
        y_score: 预测概率/得分（``kind="proba"`` 的指标需要）。
        scoring: 指标名序列。

    Returns:
        ``{指标名: 数值}``。

    Raises:
        MissingMetricError: 指标名未注册。
        MetricComputationError: 指标所需输入缺失或计算失败。
    """
    labels = np.asarray(y_true)
    results: Dict[str, float] = {}
    for name in scoring:
        scorer = get_metric(name)
        output = y_pred if scorer.kind == "labels" else y_score
        if output is None:
            raise MetricComputationError(
                f"指标 {name!r} 需要 {'y_pred' if scorer.kind == 'labels' else 'y_score'}",
                hint="检查估计器是否支持 predict_proba",
            )
        try:
            value = scorer.func(labels, np.asarray(output))
        except MetricComputationError:
            raise
        except Exception as exc:  # noqa: BLE001 - sklearn 异常统一包成 E405
            raise MetricComputationError(
                f"指标 {name!r} 计算失败：{exc}", hint="检查真实标签与预测值形状"
            ) from exc
        results[name] = float(value)
    return results


def check_directions() -> bool:
    """断言全部指标方向均为 +1（供单测调用）。"""
    return all(scorer.direction == 1 for scorer in METRIC_REGISTRY.values())
