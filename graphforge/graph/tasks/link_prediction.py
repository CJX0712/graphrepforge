"""链路预测任务：边特征（hadamard / average / l2 / concat）→ 分类器 → ROC-AUC / AP。

防泄漏约定（架构 §11 A8）：
    * 训练图只含**训练集正边**（``Split.extra["train_adjacency"]``）；
    * 验证/测试的正边与负边均不出现在训练图中。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from graphforge.core.errors import MetricComputationError
from graphforge.core.logging import get_logger
from graphforge.core.types import Embedding, GraphData, Split, TaskSpec, TaskType

from graphforge.graph.tasks.base import BaseTaskRunner

__all__ = ["LinkPredictionRunner"]

_LOGGER = get_logger(__name__)


class LinkPredictionRunner(BaseTaskRunner):
    """链路预测运行器（primary = ``roc_auc``）。"""

    task: TaskType = TaskType.LINK_PREDICTION

    def _evaluate(
        self,
        graph: GraphData,
        embedding: Optional[Embedding],
        spec: TaskSpec,
        split: Split,
    ) -> Dict[str, float]:
        """训练并评估链路预测器。

        Args:
            graph: 输入图。
            embedding: 嵌入（None 时用结构特征）。
            spec: 任务规格（``edge_op`` 决定边特征算子）。
            split: 边划分（``extra["edges"]`` 为边数组）。

        Returns:
            ``{指标名: 数值}``。

        Raises:
            MetricComputationError: 划分结果中缺少边数据。
        """
        edges = split.extra.get("edges")
        labels = split.y
        if edges is None or labels is None:
            raise MetricComputationError(
                "边划分结果缺少 edges / y", hint="请确认使用 EdgeSplitter"
            )
        matrix = self._features(graph, embedding, spec)
        from graphforge.training.trainer import train_link_predictor

        return train_link_predictor(
            matrix,
            np.asarray(edges, dtype=np.int64),
            np.asarray(labels),
            split,
            spec,
            estimator_factory=self._estimator_factory,
        )
