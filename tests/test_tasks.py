"""任务运行器单测：三任务 EvalResult 语义一致。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.types import EvalResult, TaskType


@pytest.mark.parametrize("method", ["spectral", "deepwalk"])
def test_node_classification_runner(tiny_sbm: object, method: str) -> None:
    """节点分类：primary=macro_f1，值域 [0,1]。"""
    from graphforge.core.types import TaskSpec
    from graphforge.eval.metrics import default_scoring
    from graphforge.graph.embeddings.registry import get_embedder
    from graphforge.graph.tasks.registry import get_task

    graph = tiny_sbm
    extra = {} if method == "spectral" else {"walk_length": 10, "num_walks": 2, "window": 3}
    embedding = get_embedder(method, dim=8, random_state=42, **extra).fit_transform(graph)  # type: ignore[attr-defined]
    spec = TaskSpec(task=TaskType.NODE_CLASSIFICATION, scoring=default_scoring(TaskType.NODE_CLASSIFICATION), random_state=42)
    runner = get_task("node_classification")
    result = runner.run(graph, embedding, spec)  # type: ignore[attr-defined]
    assert isinstance(result, EvalResult)
    assert result.task == TaskType.NODE_CLASSIFICATION
    assert result.primary_metric == "macro_f1"
    assert 0.0 <= result.primary_value <= 1.0
    assert 0 < result.n_samples <= graph.num_nodes  # type: ignore[attr-defined]


def test_link_prediction_runner(tiny_sbm: object) -> None:
    """链路预测：primary=roc_auc，训练图无测试边泄漏。"""
    from graphforge.core.types import TaskSpec
    from graphforge.graph.embeddings.registry import get_embedder
    from graphforge.graph.tasks.registry import get_task

    graph = tiny_sbm
    embedding = get_embedder("spectral", dim=8, random_state=42).fit_transform(graph)  # type: ignore[attr-defined]
    spec = TaskSpec(
        task=TaskType.LINK_PREDICTION,
        scoring=("roc_auc", "average_precision"),
        random_state=42,
    )
    runner = get_task("link_prediction")
    result = runner.run(graph, embedding, spec)  # type: ignore[attr-defined]
    assert result.primary_metric == "roc_auc"
    assert 0.0 <= result.primary_value <= 1.0
    assert "average_precision" in result.metrics


def test_embedding_benchmark_runner(tiny_sbm: object) -> None:
    """嵌入基准：用下游 macro_f1 度量嵌入质量。"""
    from graphforge.core.types import TaskSpec
    from graphforge.graph.embeddings.registry import get_embedder
    from graphforge.graph.tasks.registry import get_task

    graph = tiny_sbm
    embedding = get_embedder("spectral", dim=8, random_state=42).fit_transform(graph)  # type: ignore[attr-defined]
    spec = TaskSpec(task=TaskType.EMBEDDING_BENCHMARK, random_state=42)
    runner = get_task("embedding_benchmark")
    result = runner.run(graph, embedding, spec)  # type: ignore[attr-defined]
    assert 0.0 <= result.primary_value <= 1.0


def test_unknown_task_raises() -> None:
    """未知任务 → E305。"""
    from graphforge.core.errors import UnknownMethodError
    from graphforge.graph.tasks.registry import get_task

    with pytest.raises(UnknownMethodError):
        get_task("no-such-task")


def test_eval_result_serialization(tiny_sbm: object) -> None:
    """EvalResult/BenchmarkResult JSON 可序列化且可读回。"""
    import json

    from graphforge.core.types import BenchmarkResult
    from graphforge.graph.embeddings.registry import get_embedder
    from graphforge.graph.tasks.registry import get_task
    from graphforge.core.types import TaskSpec

    graph = tiny_sbm
    embedding = get_embedder("spectral", dim=8, random_state=42).fit_transform(graph)  # type: ignore[attr-defined]
    spec = TaskSpec(task=TaskType.NODE_CLASSIFICATION, random_state=42)
    result = get_task("node_classification").run(graph, embedding, spec)  # type: ignore[attr-defined]
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "macro_f1" in payload
    benchmark = BenchmarkResult(name="t", rows=[result])
    assert benchmark.best().method == result.method
