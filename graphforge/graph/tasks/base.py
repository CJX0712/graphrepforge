"""任务运行器基类：Template Method 固化 ``run() -> EvalResult`` 骨架。

固化流程：``校验 spec`` → ``划分`` → ``构造特征`` → ``_evaluate（子类）`` → ``计时`` → ``封装 EvalResult``。

作者：晨星
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np

from graphforge.core.errors import MissingMetricError
from graphforge.core.logging import get_logger
from graphforge.core.types import (
    DEFAULT_PRIMARY_METRIC,
    Embedding,
    EvalResult,
    GraphData,
    Split,
    TaskSpec,
    TaskType,
)

__all__ = ["BaseTaskRunner"]

_LOGGER = get_logger(__name__)


class BaseTaskRunner(ABC):
    """任务运行器抽象基类。

    Attributes:
        task: 任务类型（子类必须覆盖）。
    """

    #: 任务类型；子类覆盖。
    task: TaskType = TaskType.NODE_CLASSIFICATION

    def __init__(
        self,
        estimator_factory: Optional[Callable[[str, Dict[str, Any], int], Any]] = None,
        splitter: Optional[Any] = None,
        feature_method: str = "degree",
    ) -> None:
        """构造运行器。

        Args:
            estimator_factory: 下游估计器工厂；None 时使用
                :func:`graphforge.training.estimators.make_estimator`。
            splitter: 自定义划分器；None 时按任务自动选择。
            feature_method: ``embedding`` 为 None 时使用的结构特征方法。
        """
        self._estimator_factory = estimator_factory
        self._splitter = splitter
        self.feature_method = str(feature_method)

    # ---------------------------------------------------------------- 模板
    def run(
        self,
        graph: GraphData,
        embedding: Optional[Embedding],
        spec: TaskSpec,
    ) -> EvalResult:
        """执行任务（模板方法，子类不要覆盖）。

        Args:
            graph: 输入图。
            embedding: 嵌入；None 时退化为结构特征基线。
            spec: 任务规格。

        Returns:
            :class:`EvalResult`。

        Raises:
            MissingMetricError: 结果中缺少主指标（E406）。
        """
        spec.validate()
        graph.validate()
        started = time.perf_counter()
        split = self._make_split(graph, spec)
        split.validate()
        metrics = self._evaluate(graph, embedding, spec, split)
        primary = self._resolve_primary(spec, metrics)
        if primary not in metrics:
            raise MissingMetricError(
                f"结果缺少主指标 {primary!r}", hint=f"可用指标：{sorted(metrics)}"
            )
        elapsed = time.perf_counter() - started
        result = EvalResult(
            task=self.task,
            dataset=graph.name,
            method=embedding.method if embedding is not None else "none",
            backend=embedding.backend if embedding is not None else "internal",
            metrics={key: float(value) for key, value in metrics.items()},
            primary_metric=primary,
            primary_value=float(metrics[primary]),
            n_samples=int(split.test_idx.size),
            elapsed_sec=float(elapsed),
            params=self._snapshot_params(spec, embedding),
        )
        _LOGGER.info(
            "任务 %s 完成：%s/%s primary=%s=%.4f（%.3fs）",
            self.task.value,
            result.dataset,
            result.method,
            result.primary_metric,
            result.primary_value,
            result.elapsed_sec,
        )
        return result

    # ---------------------------------------------------------------- 钩子
    def _make_split(self, graph: GraphData, spec: TaskSpec) -> Split:
        """按任务选择划分器并划分。

        Args:
            graph: 输入图。
            spec: 任务规格。

        Returns:
            :class:`Split`。
        """
        if self._splitter is not None:
            return self._splitter.split(graph, spec)
        from graphforge.preprocess.split import EdgeSplitter, StratifiedNodeSplitter

        if self.task is TaskType.LINK_PREDICTION:
            splitter: Any = EdgeSplitter(
                negative_ratio=spec.negative_ratio, random_state=spec.random_state
            )
        else:
            splitter = StratifiedNodeSplitter(random_state=spec.random_state)
        return splitter.split(graph, spec)

    def _features(
        self, graph: GraphData, embedding: Optional[Embedding], spec: TaskSpec
    ) -> np.ndarray:
        """取节点表示：优先用嵌入，缺失时退化为结构特征。

        Args:
            graph: 输入图。
            embedding: 嵌入；None 时构造结构特征。
            spec: 任务规格（未使用，保留扩展位）。

        Returns:
            ``(n, d)`` 特征矩阵。
        """
        del spec  # 预留扩展位：未来可在此拼接节点属性
        if embedding is not None:
            return embedding.align(graph.node_ids).matrix
        from graphforge.preprocess.features import build_features

        features = build_features(graph, method=self.feature_method)
        if features.shape[1] == 0:
            features = build_features(graph, method="degree")
        return features

    def _resolve_primary(self, spec: TaskSpec, metrics: Dict[str, float]) -> str:
        """解析主指标。

        Args:
            spec: 任务规格。
            metrics: 已算出的指标字典。

        Returns:
            主指标名。
        """
        if spec.primary_metric:
            return spec.primary_metric
        return DEFAULT_PRIMARY_METRIC[self.task]

    def _snapshot_params(
        self, spec: TaskSpec, embedding: Optional[Embedding]
    ) -> Dict[str, Any]:
        """构造结果中的参数快照。"""
        return {
            "estimator": spec.estimator,
            "estimator_params": dict(spec.estimator_params),
            "edge_op": spec.edge_op,
            "random_state": int(spec.random_state),
            "use_cv": bool(spec.use_cv),
            "embedding_params": dict(embedding.params) if embedding is not None else {},
        }

    # ---------------------------------------------------------------- 抽象
    @abstractmethod
    def _evaluate(
        self,
        graph: GraphData,
        embedding: Optional[Embedding],
        spec: TaskSpec,
        split: Split,
    ) -> Dict[str, float]:
        """子类实现：返回指标字典（全部"越大越好"）。

        Args:
            graph: 输入图。
            embedding: 嵌入（可能为 None）。
            spec: 任务规格。
            split: 划分结果。

        Returns:
            ``{指标名: 数值}``。
        """
        raise NotImplementedError
