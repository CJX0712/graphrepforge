"""随机游走与 skipgram 单测：确定性、形状、统计性质。"""

from __future__ import annotations

import numpy as np

from graphforge.core.errors import EmbeddingFitError


def test_uniform_walk_deterministic(tiny_sbm: object) -> None:
    """同 seed 均匀游走完全一致。"""
    from graphforge.core.utils import make_rng
    from graphforge.graph.embeddings.walks import uniform_random_walk

    adjacency = tiny_sbm.adjacency  # type: ignore[attr-defined]
    w1 = uniform_random_walk(adjacency, start=0, length=10, rng=make_rng(42, "walk"))
    w2 = uniform_random_walk(adjacency, start=0, length=10, rng=make_rng(42, "walk"))
    assert w1 == w2
    assert len(w1) == 10
    assert w1[0] == 0


def test_biased_walk_returns_nodes(tiny_sbm: object) -> None:
    """偏置游走长度正确、节点合法。"""
    from graphforge.core.utils import make_rng
    from graphforge.graph.embeddings.walks import biased_random_walk

    adjacency = tiny_sbm.adjacency  # type: ignore[attr-defined]
    walk = biased_random_walk(adjacency, start=0, length=12, p=1.0, q=0.5, rng=make_rng(42, "walk"))
    assert len(walk) == 12
    assert all(0 <= node < adjacency.shape[0] for node in walk)


def test_alias_table_matches_distribution() -> None:
    """alias 采样分布与目标概率一致（大数定律）。"""
    from graphforge.graph.embeddings.walks import alias_draw, alias_setup

    probs = [0.1, 0.6, 0.3]
    jump, prob_q = alias_setup(probs)
    rng = np.random.default_rng(0)
    counts = np.zeros(3)
    trials = 60000
    for _ in range(trials):
        counts[alias_draw(jump, prob_q, rng)] += 1
    empirical = counts / trials
    assert np.allclose(empirical, probs, atol=0.01)


def test_generate_walks_shape_and_cover(tiny_sbm: object) -> None:
    """语料条数 = num_walks × nodes，节点全覆盖。"""
    from graphforge.graph.embeddings.walks import generate_walks

    adjacency = tiny_sbm.adjacency  # type: ignore[attr-defined]
    num_nodes = adjacency.shape[0]
    walks = generate_walks(adjacency, num_walks=2, walk_length=8, random_state=42)
    assert len(walks) == 2 * num_nodes
    visited = {node for walk in walks for node in walk}
    assert visited == set(range(num_nodes))


def test_skipgram_invalid_params_raise() -> None:
    """非法超参 → E302。"""
    from graphforge.graph.embeddings.skipgram import SkipGramTrainer

    for kwargs in (
        {"dim": 0},
        {"window": 0},
        {"negative": 0},
        {"epochs": 0},
        {"learning_rate": 0.0},
    ):
        try:
            SkipGramTrainer(**kwargs)  # type: ignore[arg-type]
        except EmbeddingFitError:
            continue
        raise AssertionError(f"expected EmbeddingFitError for {kwargs}")


def test_skipgram_output_shape_and_finite(tiny_sbm: object) -> None:
    """训练输出 (n, dim) 且无 NaN。"""
    from graphforge.graph.embeddings.skipgram import SkipGramTrainer
    from graphforge.graph.embeddings.walks import generate_walks

    adjacency = tiny_sbm.adjacency  # type: ignore[attr-defined]
    num_nodes = adjacency.shape[0]
    walks = generate_walks(adjacency, num_walks=2, walk_length=10, random_state=42)
    trainer = SkipGramTrainer(dim=8, window=3, negative=2, epochs=1, random_state=42)
    matrix = trainer.train(walks, num_nodes=num_nodes)
    assert matrix.shape == (num_nodes, 8)
    assert np.all(np.isfinite(matrix))


def test_skipgram_deterministic(tiny_sbm: object) -> None:
    """同 seed 训练结果逐元素一致。"""
    from graphforge.graph.embeddings.skipgram import SkipGramTrainer
    from graphforge.graph.embeddings.walks import generate_walks

    adjacency = tiny_sbm.adjacency  # type: ignore[attr-defined]
    walks = generate_walks(adjacency, num_walks=2, walk_length=10, random_state=7)
    m1 = SkipGramTrainer(dim=8, window=3, random_state=7).train(walks, num_nodes=adjacency.shape[0])
    m2 = SkipGramTrainer(dim=8, window=3, random_state=7).train(walks, num_nodes=adjacency.shape[0])
    assert np.array_equal(m1, m2)
