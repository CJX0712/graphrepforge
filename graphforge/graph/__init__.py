"""GraphForge 领域层：``embeddings``（图嵌入）+ ``tasks``（下游任务）。

作者：晨星
"""

from __future__ import annotations

from graphforge.graph import embeddings as embeddings
from graphforge.graph import tasks as tasks

from graphforge.graph.embeddings import (
    BaseEmbedder,
    DeepWalkEmbedder,
    GraRepEmbedder,
    Node2VecEmbedder,
    SkipGramTrainer,
    SpectralEmbedder,
    available_backends,
    backend_report,
    get_embedder,
    list_embedders,
)
from graphforge.graph.tasks import (
    BaseTaskRunner,
    EmbeddingBenchmarkRunner,
    LinkPredictionRunner,
    NodeClassificationRunner,
    get_task,
    list_tasks,
)

__all__ = [
    "embeddings",
    "tasks",
    "BaseEmbedder",
    "SpectralEmbedder",
    "DeepWalkEmbedder",
    "Node2VecEmbedder",
    "GraRepEmbedder",
    "SkipGramTrainer",
    "available_backends",
    "backend_report",
    "list_embedders",
    "get_embedder",
    "BaseTaskRunner",
    "NodeClassificationRunner",
    "LinkPredictionRunner",
    "EmbeddingBenchmarkRunner",
    "list_tasks",
    "get_task",
]
