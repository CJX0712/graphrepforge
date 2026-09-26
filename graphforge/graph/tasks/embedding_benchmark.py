"""嵌入基准任务：用**下游任务指标**衡量 embedding 质量（统一口径）。

做法：对同一份嵌入同时跑
    * 节点分类（图有标签时）——指标加 ``nc_`` 前缀；
    * 链路预测——指标加 ``lp_`` 前缀；
主指标默认取 ``nc_macro_f1``（无标签时退化为 ``lp_roc_auc``）。

这样"嵌入基准"与另两个任务共享**同一套指标实现与方向**，可直接横向比较。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from graphforge.core.logging import get_logger
from graphforge.core.types import Embedding, GraphData, Split, TaskSpec, TaskType
from graphforge.eval.metrics import default_scoring

from graphforge.graph.tasks.base import BaseTaskRunner
from graphforge.graph.tasks.link_prediction import LinkPredictionRunner
from graphforge.graph.tasks.node_classification import NodeClassificationRunner

__all__ = ["EmbeddingBenchmarkRunner"]

_LOGGER = get_logger(__name__)

#: 默认子任务（顺序决定主指标的候选顺序）。
DEFAULT_SUBTASKS: Tuple[TaskType, ...] = (
    TaskType.NODE_CLASSIFICATION,
    TaskType.LINK_PREDICTION,
)

#: 子任务 → 指标前缀。
TASK_PREFIX: Dict[TaskType, str] = {
    TaskType.NODE_CLASSIFICATION: "nc",
    TaskType.LINK_PREDICTION: "lp",
    TaskType.EMBEDDING_BENCHMARK: "eb",
}


class EmbeddingBenchmarkRunner(BaseTaskRunner):
    """嵌入基准运行器（聚合多个下游任务的指标）。

    Attributes:
        subtasks: 参与评测的子任务元组。
    """

    task: TaskType = TaskType.EMBEDDING_BENCHMARK

    def __init__(
        self,
        estimator_factory: Any = None,
        splitter: Any = None,
        feature_method: str = "degree",
        subtasks: Sequence[TaskType] = DEFAULT_SUBTASKS,
    ) -> None:
        """构造嵌入基准运行器。

        Args:
            estimator_factory: 估计器工厂。
            splitter: 划分器（通常留空，由各子任务自行选择）。
            feature_method: 无嵌入时的结构特征方法。
            subtasks: 参与评测的子任务。
        """
        super().__init__(
            estimator_factory=estimator_factory,
            splitter=splitter,
            feature_method=feature_method,
        )
        self.subtasks: Tuple[TaskType, ...] = tuple(subtasks)

    def _evaluate(
        self,
        graph: GraphData,
        embedding: Optional[Embedding],
        spec: TaskSpec,
        split: Split,
    ) -> Dict[str, float]:
        """跑全部子任务并合并（带前缀的）指标。

        Args:
            graph: 输入图。
            embedding: 嵌入。
            spec: 任务规格。
            split: 划分结果（本任务不使用，由子任务各自划分）。

        Returns:
            合并后的指标字典。
        """
        del split  # 子任务各自划分，外层划分不使用
        metrics: Dict[str, float] = {}
        runners: Dict[TaskType, BaseTaskRunner] = {
            TaskType.NODE_CLASSIFICATION: NodeClassificationRunner(
                estimator_factory=self._estimator_factory, feature_method=self.feature_method
            ),
            TaskType.LINK_PREDICTION: LinkPredictionRunner(
                estimator_factory=self._estimator_factory, feature_method=self.feature_method
            ),
        }
        for subtask in self.subtasks:
            if subtask is TaskType.NODE_CLASSIFICATION and not graph.has_labels:
                _LOGGER.warning("图 %s 无标签，嵌入基准跳过节点分类子任务", graph.name)
                continue
            runner = runners.get(subtask)
            if runner is None:
                continue
            sub_spec = spec.copy_with(
                task=subtask,
                scoring=default_scoring(subtask),
                primary_metric=None,
            )
            sub_result = runner.run(graph, embedding, sub_spec)
            prefix = TASK_PREFIX.get(subtask, subtask.value)
            for name, value in sub_result.metrics.items():
                metrics[f"{prefix}_{name}"] = float(value)
        if not metrics:
            from graphforge.core.errors import MissingMetricError

            raise MissingMetricError(
                "嵌入基准未产出任何指标", hint="检查 subtasks 配置与图标签"
            )
        return metrics

    def _resolve_primary(self, spec: TaskSpec, metrics: Dict[str, float]) -> str:
        """主指标：显式指定优先，否则取 ``nc_macro_f1`` / ``lp_roc_auc``。"""
        if spec.primary_metric:
            return spec.primary_metric
        for candidate in ("nc_macro_f1", "nc_accuracy", "lp_roc_auc"):
            if candidate in metrics:
                return candidate
        return sorted(metrics)[0]
