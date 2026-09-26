"""命令行入口：``python -m graphforge.cli <command>``。

六个子命令：
    doctor          后端与优化器健康检查
    list-datasets   列出数据集
    list-methods    列出嵌入方法 / 任务 / 指标 / 估计器
    embed           生成嵌入并输出摘要
    run             单次「数据集 × 方法 × 任务」评测
    benchmark       网格基准评测（落盘 JSON + Markdown）

表头统一固定宽度 ``{:<12}``，避免中英文错位。

作者：晨星
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["main", "build_parser"]

_FORMAT = "{:<12}"


def _setup_logging(level: str = "INFO") -> None:
    """CLI 专用日志初始化（库内不装 handler）。"""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _row(cells: Sequence[Any]) -> str:
    """按 12 字符列宽渲染一行（超长截断）。"""
    return "".join(_FORMAT.format(str(cell)[:12]) for cell in cells)


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器（独立函数，便于单测）。"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="INFO", help="日志级别（DEBUG/INFO/WARNING）")
    parser = argparse.ArgumentParser(
        prog="graphforge",
        description="GraphForge —— 图嵌入与图机器学习评测基准（作者：晨星）",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", parents=[common], help="后端与优化器健康检查")
    sub.add_parser("list-datasets", parents=[common], help="列出全部数据集")

    lm = sub.add_parser("list-methods", parents=[common], help="列出方法 / 任务 / 指标")
    lm.add_argument("--kind", default="embedders", choices=["embedders", "tasks", "metrics", "estimators"])

    em = sub.add_parser("embed", parents=[common], help="生成嵌入并输出摘要")
    em.add_argument("--dataset", default="karate")
    em.add_argument("--method", default="node2vec")
    em.add_argument("--dim", type=int, default=None)
    em.add_argument("--seed", type=int, default=42)
    em.add_argument("--output", default=None, help="摘要 JSON 落盘路径（可选）")

    run = sub.add_parser("run", parents=[common], help="单次评测")
    run.add_argument("--dataset", default="karate")
    run.add_argument("--task", default="node_classification")
    run.add_argument("--method", default="node2vec")
    run.add_argument("--dim", type=int, default=None)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", default=None, help="结果 JSON 落盘路径（可选）")

    bench = sub.add_parser("benchmark", parents=[common], help="网格基准评测")
    bench.add_argument("--datasets", default="sbm,karate", help="逗号分隔")
    bench.add_argument("--methods", default=None, help="逗号分隔；默认全部")
    bench.add_argument("--tasks", default="node_classification", help="逗号分隔")
    bench.add_argument("--dim", type=int, default=32)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--name", default="benchmark")
    bench.add_argument("--output", default="artifacts/benchmark.json")
    bench.add_argument("--report", default=None, help="Markdown 报告路径（可选）")
    return parser


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """doctor：后端 / 优化器 / 版本检查。"""
    import graphforge
    from graphforge.core.config import get_config
    from graphforge.graph.embeddings.backends import backend_report
    from graphforge.hpo.backends import backend_report as hpo_report

    config = get_config()
    print(f"GraphForge v{graphforge.__version__}（作者：{graphforge.__author__}）")
    print(_row(("", "backend", "status", "detail")))
    print(_row(("", "-" * 11, "-" * 11, "")))
    for item in backend_report() + hpo_report():
        print(_row(("", item.get("backend", "?"), item.get("status", "?"), item.get("detail", ""))))
    print(
        _row(("", "config", "value", ""))
    )
    print(_row(("", "random_state", config.random_state, "")))
    print(_row(("", "dim", config.dim, "")))
    print(_row(("", "hpo_backend", config.hpo_backend, "")))
    print(_row(("", "output_dir", config.output_dir, "")))
    return 0


def _cmd_list_datasets(_args: argparse.Namespace) -> int:
    """list-datasets：数据集清单。"""
    from graphforge.data.registry import get_dataset, list_datasets

    names = list_datasets()
    print(_row(("dataset", "nodes", "edges", "labeled")))
    print(_row(("-" * 11, "-" * 11, "-" * 11, "-" * 11)))
    for name in names:
        try:
            graph = get_dataset(name, random_state=42)
            print(_row((name, graph.num_nodes, graph.num_edges, graph.has_labels)))
        except Exception as exc:  # noqa: BLE001 - 生成失败也展示
            print(_row((name, "-", "-", f"error: {exc}")))
    return 0


def _cmd_list_methods(args: argparse.Namespace) -> int:
    """list-methods：按 kind 列出注册项。"""
    kind = args.kind
    if kind == "embedders":
        from graphforge.graph.embeddings.registry import list_embedders

        items: List[str] = list_embedders()
        header = "embedder"
    elif kind == "tasks":
        from graphforge.graph.tasks.registry import list_tasks

        items = list_tasks()
        header = "task"
    elif kind == "metrics":
        from graphforge.eval.metrics import list_metrics

        items = list_metrics()
        header = "metric"
    else:
        from graphforge.training.estimators import list_estimators

        items = list_estimators()
        header = "estimator"
    print(_row((header,)))
    print(_row(("-" * 11,)))
    for item in items:
        print(_row((item,)))
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    """embed：生成嵌入，输出 shape / 向量范数摘要。"""
    from graphforge.data.registry import get_dataset
    from graphforge.graph.embeddings.registry import get_embedder

    graph = get_dataset(args.dataset, random_state=args.seed)
    embedder = get_embedder(args.method, dim=args.dim, random_state=args.seed)
    embedding = embedder.fit_transform(graph)
    norms = embedding.matrix.sum(axis=1) * 0 + (embedding.matrix**2).sum(axis=1) ** 0.5
    summary: Dict[str, Any] = {
        "dataset": graph.name,
        "method": embedding.method,
        "backend": embedding.backend,
        "num_nodes": embedding.num_nodes,
        "dim": embedding.dim,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "params": embedding.params,
    }
    print(_row(("dataset", "method", "backend", "nodes", "dim", "norm_mean")))
    print(_row(("-" * 11, "-" * 11, "-" * 11, "-" * 11, "-" * 11, "-" * 11)))
    print(
        _row(
            (
                summary["dataset"],
                summary["method"],
                summary["backend"],
                summary["num_nodes"],
                summary["dim"],
                f"{summary['norm_mean']:.4f}",
            )
        )
    )
    if args.output:
        from graphforge.core.utils import save_json

        save_json(summary, args.output)
        print(f"摘要已落盘：{args.output}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """run：单次评测并打印结果表。"""
    from graphforge.core.types import TaskSpec, TaskType
    from graphforge.eval.metrics import default_scoring
    from graphforge.pipeline.pipeline import GraphPipeline

    task = TaskType(args.task)
    spec = TaskSpec(task=task, scoring=default_scoring(task), random_state=args.seed)
    pipeline = GraphPipeline(
        dataset=args.dataset,
        task=task,
        method=args.method,
        spec=spec,
        dim=args.dim,
    )
    result = pipeline.run()
    print(_row(("dataset", "method", "backend", "task", "metric", "value")))
    print(_row(("-" * 11,) * 6))
    print(
        _row(
            (
                result.dataset,
                result.method,
                result.backend,
                result.task.value,
                result.primary_metric,
                f"{result.primary_value:.4f}",
            )
        )
    )
    if args.output:
        from graphforge.core.utils import save_json

        save_json(result.to_dict(), args.output)
        print(f"结果已落盘：{args.output}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """benchmark：网格评测，落盘 JSON + 表格 + 可选 Markdown。"""
    from graphforge.core.types import TaskSpec, TaskType
    from graphforge.eval.metrics import default_scoring
    from graphforge.eval.report import BenchmarkReport
    from graphforge.pipeline.benchmark import Benchmark

    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    methods = (
        [item.strip() for item in str(args.methods).split(",") if item.strip()]
        if args.methods
        else None
    )
    tasks = [TaskType(item.strip()) for item in str(args.tasks).split(",") if item.strip()]
    spec = TaskSpec(
        task=tasks[0],
        scoring=default_scoring(tasks[0]),
        random_state=args.seed,
    )
    bench = Benchmark(
        datasets=datasets,
        methods=methods,
        tasks=tasks,
        spec=spec,
        dim=args.dim,
        name=args.name,
    )
    result = bench.run()
    print(result.to_table())
    if args.output:
        result.to_json(args.output)
        print(f"基准结果已落盘：{args.output}")
    if args.report:
        report = BenchmarkReport(result)
        report.save_markdown(args.report)
        print(f"Markdown 报告已落盘：{args.report}")
    return 0


_COMMANDS = {
    "doctor": _cmd_doctor,
    "list-datasets": _cmd_list_datasets,
    "list-methods": _cmd_list_methods,
    "embed": _cmd_embed,
    "run": _cmd_run,
    "benchmark": _cmd_benchmark,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 主入口。

    Args:
        argv: 参数列表；None 时取 ``sys.argv[1:]``。

    Returns:
        退出码（0 成功）。
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _setup_logging(args.log_level)
    handler = _COMMANDS[args.command]
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
