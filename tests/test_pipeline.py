"""pipeline 单测：GraphPipeline / Benchmark 行为。"""

from __future__ import annotations

import pytest

from graphforge.core.types import TaskSpec, TaskType


def test_pipeline_run(tiny_sbm: object) -> None:
    """GraphPipeline.run：GraphData 输入端到端。"""
    from graphforge.pipeline import GraphPipeline

    pipeline = GraphPipeline(
        dataset=tiny_sbm,  # type: ignore[arg-type]
        task=TaskType.NODE_CLASSIFICATION,
        method="spectral",
        dim=8,
    )
    result = pipeline.run()
    assert result.primary_metric == "macro_f1"
    assert 0.0 <= result.primary_value <= 1.0
    assert result.dataset == tiny_sbm.name  # type: ignore[attr-defined]


def test_pipeline_by_name() -> None:
    """按名字取数据集（karate，34 节点，秒级）。"""
    from graphforge.pipeline import GraphPipeline

    pipeline = GraphPipeline(dataset="karate", task="node_classification", method="spectral", dim=8)
    result = pipeline.run()
    assert result.dataset == "karate"
    assert 0 < result.n_samples <= 34


def test_pipeline_dry_run() -> None:
    """dry_run 不执行计算、描述完整。"""
    from graphforge.pipeline import GraphPipeline

    plan = GraphPipeline(dataset="karate", task="node_classification", method="spectral").dry_run()
    assert plan["dataset_known"] is True
    assert plan["method_known"] is True
    assert plan["task"] == "node_classification"


def test_pipeline_unknown_task_raises() -> None:
    """未知任务名 → E305。"""
    from graphforge.core.errors import UnknownMethodError
    from graphforge.pipeline import GraphPipeline

    with pytest.raises(UnknownMethodError):
        GraphPipeline(task="no-such-task")


def test_pipeline_with_hpo(tiny_sbm: object) -> None:
    """HPO 打开（grid 2 trials）管线照常出结果。"""
    from graphforge.core.types import HPOSpec
    from graphforge.pipeline import GraphPipeline

    pipeline = GraphPipeline(
        dataset=tiny_sbm,  # type: ignore[arg-type]
        task=TaskType.NODE_CLASSIFICATION,
        method="node2vec",
        dim=8,
        hpo=HPOSpec(backend="grid", n_trials=2),
    )
    result = pipeline.run()
    assert 0.0 <= result.primary_value <= 1.0


def test_benchmark_runs_and_skips(tiny_sbm: object) -> None:
    """Benchmark：成功行入 rows，失败组合进 skipped。"""
    from graphforge.pipeline import Benchmark

    bench = Benchmark(
        datasets=("karate",),
        methods=("spectral", "no-such-method"),
        tasks=("node_classification",),
        dim=8,
        name="unit",
    )
    result = bench.run()
    assert len(result.rows) >= 1
    assert len(result.skipped) >= 1
    assert result.rows[0].primary_metric == "macro_f1"


def test_benchmark_all_fail_raises() -> None:
    """全部组合失败 → E504。"""
    from graphforge.core.errors import BenchmarkAbortedError
    from graphforge.pipeline import Benchmark

    bench = Benchmark(
        datasets=("no-such-dataset",),
        methods=("spectral",),
        tasks=("node_classification",),
    )
    with pytest.raises(BenchmarkAbortedError):
        bench.run()


def test_benchmark_deterministic() -> None:
    """同 seed 两次 benchmark 的 primary 序列一致。"""
    from graphforge.pipeline import Benchmark

    def _run() -> list:
        result = Benchmark(
            datasets=("karate",),
            methods=("spectral",),
            tasks=("node_classification",),
            dim=8,
            name="det",
        ).run()
        return [row.primary_value for row in result.rows]

    assert _run() == _run()
