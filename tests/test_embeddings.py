"""嵌入器单测：四个内置方法 + 注册表 + 参数防泄漏。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import UnknownMethodError


def test_spectral_shape_and_finite(tiny_sbm: object) -> None:
    """Spectral：形状 (n, dim)、有限值、同 seed 可复现。"""
    from graphforge.graph.embeddings.registry import get_embedder

    graph = tiny_sbm
    emb1 = get_embedder("spectral", dim=8, random_state=42).fit_transform(graph)  # type: ignore[attr-defined]
    emb2 = get_embedder("spectral", dim=8, random_state=42).fit_transform(graph)  # type: ignore[attr-defined]
    assert emb1.matrix.shape == (graph.num_nodes, 8)  # type: ignore[attr-defined]
    assert np.all(np.isfinite(emb1.matrix))
    assert np.array_equal(emb1.matrix, emb2.matrix)
    assert emb1.backend == "internal"


def test_deepwalk_shape(tiny_sbm: object) -> None:
    """DeepWalk：形状 (n, dim)。"""
    from graphforge.graph.embeddings.registry import get_embedder

    emb = get_embedder("deepwalk", dim=8, random_state=42, walk_length=10, num_walks=2, window=3).fit_transform(tiny_sbm)  # type: ignore[attr-defined]
    assert emb.matrix.shape == (tiny_sbm.num_nodes, 8)  # type: ignore[attr-defined]


def test_node2vec_shape(tiny_sbm: object) -> None:
    """Node2Vec（p≠q）：形状 (n, dim)。"""
    from graphforge.graph.embeddings.registry import get_embedder

    emb = get_embedder("node2vec", dim=8, random_state=42, walk_length=10, num_walks=2, window=3, p=1.0, q=0.5).fit_transform(tiny_sbm)  # type: ignore[attr-defined]
    assert emb.matrix.shape == (tiny_sbm.num_nodes, 8)  # type: ignore[attr-defined]


def test_grarep_shape(tiny_sbm: object) -> None:
    """GraRep：总维度恒为 dim（k 步分量均分后截断）。"""
    from graphforge.graph.embeddings.registry import get_embedder

    emb = get_embedder("grarep", dim=8, random_state=42, k_steps=2).fit_transform(tiny_sbm)  # type: ignore[attr-defined]
    assert emb.matrix.shape[0] == tiny_sbm.num_nodes  # type: ignore[attr-defined]
    assert emb.matrix.shape[1] == 8
    assert np.all(np.isfinite(emb.matrix))


def test_registry_unknown_method() -> None:
    """未知方法 → E305。"""
    from graphforge.graph.embeddings.registry import get_embedder

    with pytest.raises(UnknownMethodError):
        get_embedder("no-such-method")


def test_set_params_reserved_rejected() -> None:
    """保留参数泄漏 → E303。"""
    from graphforge.graph.embeddings.registry import get_embedder

    embedder = get_embedder("node2vec", dim=8)
    with pytest.raises(Exception) as info:
        embedder.set_params(n_classes=3)
    assert getattr(info.value, "code", "") == "E303"


def test_get_params_roundtrip() -> None:
    """get_params 可用于重建等价嵌入器。"""
    from graphforge.graph.embeddings.registry import get_embedder

    embedder = get_embedder("node2vec", dim=16, walk_length=20)
    params = embedder.get_params()
    clone = get_embedder("node2vec", **params)
    assert clone.get_params() == params


def test_default_space_all_embedders() -> None:
    """各嵌入器默认空间可采样（SearchSpace 自洽）。"""
    import numpy as np

    from graphforge.graph.embeddings.registry import list_embedders, get_embedder
    from graphforge.core.utils import make_rng

    rng = make_rng(42, "hpo", "space")
    for name in list_embedders():
        space = get_embedder(name).default_space()
        space.validate()
        sampled = space.sample(rng)
        assert set(sampled) == set(space.names)
