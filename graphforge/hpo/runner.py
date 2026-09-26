"""HPO 执行入口：为「数据集 × 嵌入器 × 任务」组合搜索最优超参。

目标值统一取下游任务的 primary 指标（**越大越好**），保证 HPO 与
评测口径完全一致。

作者：晨星
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from graphforge.core.types import GraphData, HPOSpec, TaskSpec, TrialResult
from graphforge.core.utils import Timer

__all__ = ["run_hpo"]

_LOGGER = logging.getLogger(__name__)


def run_hpo(
    dataset: Union[str, GraphData],
    method: str,
    task: Union[str, Any],
    spec: TaskSpec,
    hpo: Optional[HPOSpec] = None,
    dim: Optional[int] = None,
    feature_method: str = "degree",
    dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> TrialResult:
    """为给定组合执行超参搜索。

    Args:
        dataset: 数据集名或 :class:`GraphData` 实例。
        method: 嵌入方法名。
        task: 任务名或 :class:`TaskType`。
        spec: 任务规格（primary 指标即优化目标）。
        hpo: HPO 配置；None 时用默认（backend=auto, n_trials=20）。
        dim: 嵌入维度；None 时取全局配置。
        feature_method: 无嵌入时的结构特征方法。
        dataset_kwargs: 数据集生成的额外参数（如 SBM 规模）。

    Returns:
        :class:`TrialResult`。

    Raises:
        E1xx/E3xx/E4xx: 数据、嵌入或任务执行失败（单 trial 失败不中断）。
    """
    from graphforge.data.registry import get_dataset
    from graphforge.graph.embeddings.registry import get_embedder
    from graphforge.graph.tasks.registry import get_task
    from graphforge.hpo.backends import resolve_optimizer
    from graphforge.hpo.optimizer import make_optimizer
    from graphforge.hpo.space import embedder_space

    config = hpo or HPOSpec()
    config.validate()
    spec.validate()

    if isinstance(dataset, str):
        graph = get_dataset(dataset, random_state=spec.random_state, **(dataset_kwargs or {}))
    else:
        graph = dataset

    def _objective(params: Dict[str, Any]) -> float:
        embedder = get_embedder(
            method,
            dim=dim,
            random_state=spec.random_state,
            **params,
        )
        embedding = embedder.fit_transform(graph)
        runner = get_task(task, feature_method=feature_method)
        result = runner.run(graph, embedding, spec)
        return float(result.primary_value)

    space = embedder_space(method)
    backend = resolve_optimizer(config.backend)
    optimizer = make_optimizer(backend)
    _LOGGER.info(
        "HPO 开始：%s × %s × %s（backend=%s, n_trials=%d）",
        dataset if isinstance(dataset, str) else dataset.name,
        method,
        task if isinstance(task, str) else task.value,
        backend,
        config.n_trials,
    )
    with Timer() as _:
        result = optimizer.search(_objective, space, config)
    _LOGGER.info(
        "HPO 完成：best_value=%.4f best_params=%s", result.best_value, result.best_params
    )
    return result
