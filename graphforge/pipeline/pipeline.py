"""GraphPipeline：数据集 → 预处理 → (HPO) → 嵌入 → 任务 → 评测。

调用方向单向无环：``pipeline → {data, preprocess, graph, training, hpo, eval} → core``。

作者：晨星
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from graphforge.core.errors import UnknownMethodError
from graphforge.core.types import EvalResult, GraphData, HPOSpec, TaskSpec, TaskType
from graphforge.core.utils import Timer

__all__ = ["GraphPipeline", "run_pipeline"]

_LOGGER = logging.getLogger(__name__)


def _coerce_task(task: Union[str, TaskType]) -> TaskType:
    """把任务名规整为 :class:`TaskType`（非法名抛 E305）。"""
    if isinstance(task, TaskType):
        return task
    try:
        return TaskType(str(task))
    except ValueError as exc:
        raise UnknownMethodError(
            f"未知任务 {task!r}",
            hint=f"可选值：{[item.value for item in TaskType]}",
        ) from exc


class GraphPipeline:
    """单次「数据集 × 嵌入方法 × 任务」端到端执行器。

    Attributes:
        dataset: 数据集名（``sbm`` / ``karate`` / ``grid`` / ``lfr``）或 :class:`GraphData`。
        task: 任务名或 :class:`TaskType`。
        method: 嵌入方法名。
        spec: 任务规格；None 时按任务自动构造默认规格。
        hpo: HPO 配置；None 或 ``backend="none"`` 时不搜索。
        dim: 嵌入维度；None 时取全局配置。
    """

    def __init__(
        self,
        dataset: Union[str, GraphData] = "sbm",
        task: Union[str, TaskType] = TaskType.NODE_CLASSIFICATION,
        method: str = "node2vec",
        spec: Optional[TaskSpec] = None,
        hpo: Optional[HPOSpec] = None,
        config: Optional[Any] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        feature_method: str = "degree",
        dim: Optional[int] = None,
    ) -> None:
        """构造管线（不执行任何计算）。"""
        self.dataset = dataset
        self.task = _coerce_task(task)
        self.method = str(method)
        self.spec = spec
        self.hpo = hpo
        self.config = config
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.feature_method = str(feature_method)
        self.dim = dim

    # ---------------------------------------------------------------- 内部
    def _resolve_spec(self) -> TaskSpec:
        """解析任务规格：显式传入优先，否则按任务生成默认。"""
        if self.spec is not None:
            self.spec.validate()
            return self.spec
        from graphforge.eval.metrics import default_scoring

        spec = TaskSpec(task=self.task, scoring=default_scoring(self.task))
        spec.validate()
        return spec

    def _resolve_dim(self) -> int:
        """解析嵌入维度。"""
        if self.dim is not None:
            return int(self.dim)
        from graphforge.core.config import get_config

        return int(get_config().dim)

    # ---------------------------------------------------------------- 执行
    def run(self) -> EvalResult:
        """执行完整链路。

        Returns:
            :class:`EvalResult`（primary 为该任务主指标，越大越好）。
        """
        from graphforge.data.registry import get_dataset
        from graphforge.graph.embeddings.registry import get_embedder
        from graphforge.graph.tasks.registry import get_task
        from graphforge.hpo.runner import run_hpo

        spec = self._resolve_spec()
        task_name = self.task.value

        graph = (
            self.dataset
            if isinstance(self.dataset, GraphData)
            else get_dataset(self.dataset, random_state=spec.random_state, **self.dataset_kwargs)
        )
        dataset_name = graph.name if isinstance(self.dataset, GraphData) else str(self.dataset)

        best_params: Dict[str, Any] = {}
        if self.hpo is not None and str(self.hpo.backend) != "none":
            trial = run_hpo(
                dataset=graph,
                method=self.method,
                task=task_name,
                spec=spec,
                hpo=self.hpo,
                dim=self._resolve_dim(),
                feature_method=self.feature_method,
            )
            best_params = dict(trial.best_params)

        embedder = get_embedder(
            self.method,
            dim=self._resolve_dim(),
            random_state=spec.random_state,
            **best_params,
        )
        with Timer() as timed:
            embedding = embedder.fit_transform(graph)
            runner = get_task(task_name, feature_method=self.feature_method)
            result = runner.run(graph, embedding, spec)
        result.params = {**result.params, "embedder_params": dict(embedder.get_params()), "hpo": best_params}
        result.elapsed_sec = result.elapsed_sec + float(timed.elapsed)
        result.dataset = dataset_name
        _LOGGER.info(
            "Pipeline 完成：%s × %s × %s → %s=%.4f",
            dataset_name,
            self.method,
            task_name,
            result.primary_metric,
            result.primary_value,
        )
        return result

    def dry_run(self) -> Dict[str, Any]:
        """返回执行计划描述（不做任何计算）。"""
        from graphforge.data.registry import has_dataset
        from graphforge.graph.embeddings.registry import has_embedder

        spec = self.spec
        return {
            "dataset": self.dataset if isinstance(self.dataset, str) else self.dataset.name,
            "dataset_known": True if isinstance(self.dataset, GraphData) else has_dataset(str(self.dataset)),
            "task": self.task.value,
            "method": self.method,
            "method_known": has_embedder(self.method),
            "dim": self.dim if self.dim is not None else None,
            "hpo": None if self.hpo is None else self.hpo.to_dict()
            if hasattr(self.hpo, "to_dict")
            else {"backend": self.hpo.backend, "n_trials": self.hpo.n_trials},
            "spec": {
                "scoring": list(spec.resolve_scoring()) if spec else "default",
                "primary": spec.resolve_primary() if spec else "default",
            },
        }


def run_pipeline(**kwargs: Any) -> EvalResult:
    """便捷函数：等价于 ``GraphPipeline(**kwargs).run()``。"""
    return GraphPipeline(**kwargs).run()
