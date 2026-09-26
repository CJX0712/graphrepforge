"""Benchmark：数据集 × 嵌入方法 × 任务 网格评测。

约定：
    * 单个组合失败只记 ``skipped``，不中断整个基准；
    * 全部组合失败（rows 为空且 skipped 非空）才抛 E504；
    * 同一 ``random_state`` 下结果完全可复现。

作者：晨星
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from graphforge.core.errors import BenchmarkAbortedError
from graphforge.core.types import (
    BenchmarkResult,
    EvalResult,
    GraphData,
    HPOSpec,
    TaskSpec,
    TaskType,
)

__all__ = ["Benchmark"]

_LOGGER = logging.getLogger(__name__)


def _coerce_task(value: Union[str, TaskType]) -> TaskType:
    """任务名 → TaskType。"""
    if isinstance(value, TaskType):
        return value
    try:
        return TaskType(str(value))
    except ValueError as exc:
        raise BenchmarkAbortedError(
            f"未知任务 {value!r}",
            hint=f"可选值：{[item.value for item in TaskType]}",
        ) from exc


class Benchmark:
    """网格基准评测器。

    Attributes:
        datasets: 数据集名元组。
        methods: 嵌入方法名元组；None 时取全部已注册方法。
        tasks: 任务名元组。
        spec: 任务规格；None 时按任务自动构造。
        hpo: HPO 配置；None 时不搜索。
    """

    def __init__(
        self,
        datasets: Sequence[Union[str, GraphData]] = ("sbm",),
        methods: Optional[Sequence[str]] = None,
        tasks: Sequence[Union[str, TaskType]] = (TaskType.NODE_CLASSIFICATION,),
        spec: Optional[TaskSpec] = None,
        hpo: Optional[HPOSpec] = None,
        config: Optional[Any] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        feature_method: str = "degree",
        dim: Optional[int] = None,
        name: str = "benchmark",
    ) -> None:
        """构造基准（不执行计算）。"""
        self.datasets: Tuple[Union[str, GraphData], ...] = tuple(datasets)
        self.methods: Optional[Tuple[str, ...]] = tuple(methods) if methods else None
        self.tasks: Tuple[TaskType, ...] = tuple(_coerce_task(item) for item in tasks)
        self.spec = spec
        self.hpo = hpo
        self.config = config
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.feature_method = str(feature_method)
        self.dim = dim
        self.name = str(name)

    # ---------------------------------------------------------------- 执行
    def run(self) -> BenchmarkResult:
        """执行网格评测。

        Returns:
            :class:`BenchmarkResult`。

        Raises:
            BenchmarkAbortedError: 全部组合失败（E504）。
        """
        from graphforge.graph.embeddings.registry import list_embedders
        from graphforge.pipeline.pipeline import GraphPipeline

        methods = tuple(self.methods) if self.methods else tuple(list_embedders())
        rows: List[EvalResult] = []
        skipped: List[Dict[str, str]] = []
        total = len(self.datasets) * len(methods) * len(self.tasks)
        done = 0
        for dataset in self.datasets:
            dataset_name = dataset if isinstance(dataset, str) else dataset.name
            for task in self.tasks:
                for method in methods:
                    done += 1
                    label = f"[{done}/{total}] {dataset_name} × {method} × {task.value}"
                    try:
                        pipeline = GraphPipeline(
                            dataset=dataset,
                            task=task,
                            method=method,
                            spec=self.spec,
                            hpo=self.hpo,
                            config=self.config,
                            dataset_kwargs=self.dataset_kwargs,
                            feature_method=self.feature_method,
                            dim=self.dim,
                        )
                        rows.append(pipeline.run())
                    except Exception as exc:  # noqa: BLE001 - 单组合失败不中断
                        _LOGGER.warning("跳过 %s：%s", label, exc)
                        skipped.append(
                            {
                                "dataset": dataset_name,
                                "method": method,
                                "task": task.value,
                                "reason": f"{type(exc).__name__}: {exc}"[:120],
                            }
                        )
        if not rows and skipped:
            raise BenchmarkAbortedError(
                f"全部 {len(skipped)} 个组合失败",
                hint="检查数据集与方法名；运行 `python -m graphforge.cli doctor` 看后端状态",
            )
        result = BenchmarkResult(
            name=self.name,
            config={
                "datasets": [d if isinstance(d, str) else d.name for d in self.datasets],
                "methods": list(methods),
                "tasks": [t.value for t in self.tasks],
                "dim": self.dim,
                "feature_method": self.feature_method,
                "hpo": None if self.hpo is None else {"backend": self.hpo.backend, "n_trials": self.hpo.n_trials},
            },
            rows=rows,
            skipped=skipped,
        )
        _LOGGER.info(
            "Benchmark 完成：%d 成功 / %d 跳过", len(rows), len(skipped)
        )
        return result
