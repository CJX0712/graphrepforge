"""GraphForge 任务层：节点分类 / 链路预测 / 嵌入基准（统一 ``TaskRunner`` 接口）。

作者：晨星
"""

from __future__ import annotations

from graphforge.graph.tasks import base as base
from graphforge.graph.tasks import embedding_benchmark as embedding_benchmark
from graphforge.graph.tasks import link_prediction as link_prediction
from graphforge.graph.tasks import node_classification as node_classification
from graphforge.graph.tasks import registry as registry

from graphforge.graph.tasks.base import BaseTaskRunner
from graphforge.graph.tasks.embedding_benchmark import EmbeddingBenchmarkRunner
from graphforge.graph.tasks.link_prediction import LinkPredictionRunner
from graphforge.graph.tasks.node_classification import NodeClassificationRunner
from graphforge.graph.tasks.registry import (
    TASK_REGISTRY,
    get_task,
    has_task,
    list_tasks,
    register_task,
)

__all__ = [
    "base",
    "node_classification",
    "link_prediction",
    "embedding_benchmark",
    "registry",
    "BaseTaskRunner",
    "NodeClassificationRunner",
    "LinkPredictionRunner",
    "EmbeddingBenchmarkRunner",
    "TASK_REGISTRY",
    "register_task",
    "list_tasks",
    "get_task",
    "has_task",
]
