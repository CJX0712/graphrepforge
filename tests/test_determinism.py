"""确定性单测：同 seed 全链路逐字节可复现。"""

from __future__ import annotations

import json

import numpy as np

from graphforge.cli import main


def _run_benchmark(tmp_path, seed: int) -> dict:
    target = tmp_path / f"bench_{seed}.json"
    code = main(
        [
            "--log-level", "ERROR", "benchmark",
            "--datasets", "karate", "--methods", "spectral,deepwalk",
            "--tasks", "node_classification", "--dim", "8", "--seed", str(seed),
            "--output", str(target),
        ]
    )
    assert code == 0
    return json.loads(target.read_text(encoding="utf-8"))


def test_benchmark_json_reproducible(tmp_path) -> None:
    """同 seed 两轮 benchmark JSON 的 rows 完全一致（elapsed 除外）。"""

    def _strip(payload: dict) -> list:
        return [
            {key: value for key, value in row.items() if key != "elapsed_sec"}
            for row in payload["rows"]
        ]

    first = _strip(_run_benchmark(tmp_path, 42))
    second = _strip(_run_benchmark(tmp_path, 42))
    assert first == second


def test_different_seed_differs(tmp_path) -> None:
    """不同 seed 的 deepwalk 嵌入矩阵不同（随机性真实生效）。"""
    from graphforge.core.config import get_config
    from graphforge.data.registry import get_dataset
    from graphforge.graph.embeddings.registry import get_embedder

    graph = get_dataset("karate", random_state=42)
    m1 = get_embedder("deepwalk", dim=8, random_state=42, walk_length=10, num_walks=2, window=3).fit_transform(graph).matrix
    m2 = get_embedder("deepwalk", dim=8, random_state=43, walk_length=10, num_walks=2, window=3).fit_transform(graph).matrix
    assert not np.allclose(m1, m2)
