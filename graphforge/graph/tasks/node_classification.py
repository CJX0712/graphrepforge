"""节点分类任务：嵌入/结构特征 → 下游分类器 → accuracy / macro-F1。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from graphforge.core.errors import MetricComputationError
from graphforge.core.logging import get_logger
from graphforge.core.types import Embedding, GraphData, Split, TaskSpec, TaskType

from graphforge.graph.tasks.base import BaseTaskRunner

__all__ = ["NodeClassificationRunner"]

_LOGGER = get_logger(__name__)


class NodeClassificationRunner(BaseTaskRunner):
    """节点分类运行器（primary = ``macro_f1``）。"""

    task: TaskType = TaskType.NODE_CLASSIFICATION

    def _evaluate(
        self,
        graph: GraphData,
        embedding: Optional[Embedding],
        spec: TaskSpec,
        split: Split,
    ) -> Dict[str, float]:
        """训练并评估节点分类器。

        Args:
            graph: 输入图。
            embedding: 嵌入（None 时用结构特征）。
            spec: 任务规格。
            split: 节点划分。

        Returns:
            ``{指标名: 数值}``。

        Raises:
            MetricComputationError: 图无标签。
        """
        labels = split.y if split.y is not None else graph.node_labels
        if labels is None:
            raise MetricComputationError(
                f"数据集 {graph.name} 无节点标签，无法做节点分类",
                hint="改用 sbm / karate / grid，或先 attach_labels",
            )
        features = self._features(graph, embedding, spec)
        from graphforge.training.trainer import train_node_classifier

        return train_node_classifier(
            features,
            np.asarray(labels),
            split,
            spec,
            estimator_factory=self._estimator_factory,
        )
