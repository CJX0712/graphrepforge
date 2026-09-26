"""Protocol 接口清单（L0）。

这里只定义"形状"，不含实现；实现方通过结构化子类型（structural subtyping）满足协议，
无需显式继承。所有 Protocol 均 ``runtime_checkable``，便于在集成测试里做
``isinstance`` 断言。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from graphforge.core.types import (
    BenchmarkResult,
    Embedding,
    EvalResult,
    GraphData,
    HPOSpec,
    SearchSpace,
    Split,
    TaskSpec,
    TaskType,
    TrialResult,
)

__all__ = [
    "GraphSource",
    "Embedder",
    "Splitter",
    "EstimatorFactory",
    "Scorer",
    "TaskRunner",
    "Optimizer",
    "Reporter",
]


@runtime_checkable
class GraphSource(Protocol):
    """图数据源协议。"""

    def load(self, **kwargs: Any) -> GraphData:
        """载入图数据。

        Args:
            **kwargs: 数据源相关参数（如 ``random_state`` / ``path``）。

        Returns:
            :class:`GraphData`。
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """图嵌入器协议（Template Method 的具体形状）。

    约定：``dim`` / ``random_state`` 等**内部参数**在构造器里单独传入，
    :meth:`set_params` 命中保留参数时必须抛 ``E303``。
    """

    #: 方法名（如 ``"node2vec"``）。
    name: str
    #: 后端名（``"internal"`` / ``"karateclub"``）。
    backend: str

    def fit(self, graph: GraphData) -> "Embedder":
        """在给定图上训练嵌入。"""
        ...

    def transform(self) -> Embedding:
        """返回 :meth:`fit` 产出的嵌入。"""
        ...

    def fit_transform(self, graph: GraphData) -> Embedding:
        """训练并直接返回嵌入。"""
        ...

    def get_params(self) -> Dict[str, Any]:
        """返回超参字典（**不含** ``dim`` / ``random_state``）。"""
        ...

    def set_params(self, **params: Any) -> "Embedder":
        """就地修改超参；命中保留参数抛 ``E303``。"""
        ...

    def default_space(self) -> SearchSpace:
        """返回默认搜索空间，供 HPO 使用。"""
        ...


@runtime_checkable
class Splitter(Protocol):
    """划分器协议。"""

    def split(self, graph: GraphData, spec: TaskSpec) -> Split:
        """按任务规格划分样本。

        Args:
            graph: 输入图。
            spec: 任务规格。

        Returns:
            :class:`Split`。
        """
        ...


@runtime_checkable
class EstimatorFactory(Protocol):
    """下游估计器工厂协议（**三参分离**，杜绝参数泄漏）。"""

    def __call__(self, name: str, params: Dict[str, Any], random_state: int) -> Any:
        """构造 sklearn 兼容估计器。

        Args:
            name: 估计器名（``logistic_regression`` / ``random_forest`` ...）。
            params: 估计器超参（**不含** ``random_state`` 等内部参数）。
            random_state: 随机种子，由工厂负责注入。

        Returns:
            具备 ``fit`` / ``predict`` / ``predict_proba`` 的估计器。
        """
        ...


@runtime_checkable
class Scorer(Protocol):
    """指标协议。

    约定：``direction`` **恒为 +1**（所有指标一律"越大越好"），
    本项目不做"越小越好"的指标，避免跨任务比较时口径混乱。
    """

    #: 指标名。
    name: str
    #: 方向，恒为 ``+1``。
    direction: int

    def __call__(self, y_true: np.ndarray, y_out: np.ndarray) -> float:
        """计算指标值。

        Args:
            y_true: 真实标签 / 真实得分。
            y_out: 预测标签 / 预测得分。

        Returns:
            指标数值。
        """
        ...


@runtime_checkable
class TaskRunner(Protocol):
    """任务运行器协议。"""

    #: 任务类型。
    task: TaskType

    def run(
        self,
        graph: GraphData,
        embedding: Optional[Embedding],
        spec: TaskSpec,
    ) -> EvalResult:
        """执行任务并返回评估结果。

        Args:
            graph: 输入图。
            embedding: 嵌入；``None`` 时退回结构特征基线。
            spec: 任务规格。

        Returns:
            :class:`EvalResult`。
        """
        ...


@runtime_checkable
class Optimizer(Protocol):
    """超参优化器协议。"""

    #: 优化器名（``optuna`` / ``grid`` / ``random``）。
    name: str

    def search(
        self,
        objective: Callable[[Dict[str, Any]], float],
        space: SearchSpace,
        spec: HPOSpec,
    ) -> TrialResult:
        """在给定空间上搜索最优参数。

        Args:
            objective: 目标函数，入参为参数字典，返回**越大越好**的分数。
            space: 搜索空间。
            spec: HPO 配置。

        Returns:
            :class:`TrialResult`。
        """
        ...


@runtime_checkable
class Reporter(Protocol):
    """报告渲染协议。"""

    def render(self, result: BenchmarkResult) -> str:
        """把基准结果渲染为文本。

        Args:
            result: 基准结果。

        Returns:
            渲染文本（表格 / Markdown）。
        """
        ...
