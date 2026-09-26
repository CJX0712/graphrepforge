"""任务注册表：``TASK_REGISTRY`` + ``list_tasks()`` / ``get_task()``。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from graphforge.core.errors import UnknownMethodError
from graphforge.core.logging import get_logger
from graphforge.core.types import TaskType

from graphforge.graph.tasks.base import BaseTaskRunner
from graphforge.graph.tasks.embedding_benchmark import EmbeddingBenchmarkRunner
from graphforge.graph.tasks.link_prediction import LinkPredictionRunner
from graphforge.graph.tasks.node_classification import NodeClassificationRunner

__all__ = [
    "TASK_REGISTRY",
    "register_task",
    "list_tasks",
    "get_task",
    "has_task",
]

_LOGGER = get_logger(__name__)

#: 任务名 → 运行器类。
TASK_REGISTRY: Dict[str, Type[BaseTaskRunner]] = {
    TaskType.NODE_CLASSIFICATION.value: NodeClassificationRunner,
    TaskType.LINK_PREDICTION.value: LinkPredictionRunner,
    TaskType.EMBEDDING_BENCHMARK.value: EmbeddingBenchmarkRunner,
}


def register_task(
    name: str, cls: Type[BaseTaskRunner], force: bool = False
) -> None:
    """注册（或覆盖）一个任务运行器。

    Args:
        name: 任务名。
        cls: 运行器类。
        force: 允许覆盖。

    Raises:
        UnknownMethodError: 同名已存在且 ``force=False``。
    """
    if name in TASK_REGISTRY and not force:
        raise UnknownMethodError(f"任务 {name!r} 已注册", hint="如需覆盖，请传 force=True")
    TASK_REGISTRY[name] = cls
    _LOGGER.debug("注册任务：%s -> %s", name, cls.__name__)


def list_tasks() -> List[str]:
    """返回已注册任务名（升序）。"""
    return sorted(TASK_REGISTRY)


def has_task(name: str) -> bool:
    """判断任务是否已注册。"""
    return name in TASK_REGISTRY


def get_task(name: Any, **kwargs: Any) -> BaseTaskRunner:
    """构造任务运行器实例。

    Args:
        name: 任务名或 :class:`TaskType`。
        **kwargs: 透传给运行器构造器。

    Returns:
        运行器实例。

    Raises:
        UnknownMethodError: 任务名未注册（E305）。
    """
    key = name.value if isinstance(name, TaskType) else str(name)
    if key not in TASK_REGISTRY:
        raise UnknownMethodError(
            f"未知任务 {name!r}", hint=f"可选值：{list_tasks()}"
        )
    return TASK_REGISTRY[key](**kwargs)
