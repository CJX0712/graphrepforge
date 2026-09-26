"""GraphForge 图嵌入层：四种内置经典实现 + 可选后端探测 + 注册表。

内置实现（**经典论文复现，非 SOTA**）：
    * :class:`SpectralEmbedder` —— Laplacian Eigenmaps（Belkin & Niyogi 2003）
    * :class:`DeepWalkEmbedder` —— DeepWalk（Perozzi et al. 2014）
    * :class:`Node2VecEmbedder` —— node2vec（Grover & Leskovec 2016）
    * :class:`GraRepEmbedder` —— GraRep（Cao et al. 2015）

作者：晨星
"""

from __future__ import annotations

from graphforge.graph.embeddings import backends as backends
from graphforge.graph.embeddings import base as base
from graphforge.graph.embeddings import deepwalk as deepwalk
from graphforge.graph.embeddings import grarep as grarep
from graphforge.graph.embeddings import node2vec as node2vec
from graphforge.graph.embeddings import registry as registry
from graphforge.graph.embeddings import skipgram as skipgram
from graphforge.graph.embeddings import spectral as spectral
from graphforge.graph.embeddings import walks as walks

from graphforge.graph.embeddings.backends import (
    available_backends,
    backend_report,
    build_karateclub_embedder,
    karateclub_status,
    node2vec_status,
    resolve_backend,
)
from graphforge.graph.embeddings.base import BaseEmbedder
from graphforge.graph.embeddings.deepwalk import DeepWalkEmbedder
from graphforge.graph.embeddings.grarep import GraRepEmbedder
from graphforge.graph.embeddings.node2vec import Node2VecEmbedder
from graphforge.graph.embeddings.registry import (
    EMBEDDER_REGISTRY,
    get_embedder,
    get_embedder_class,
    has_embedder,
    list_embedders,
    register_embedder,
)
from graphforge.graph.embeddings.skipgram import SkipGramTrainer, build_negative_table
from graphforge.graph.embeddings.spectral import SpectralEmbedder
from graphforge.graph.embeddings.walks import (
    WalkSampler,
    alias_draw,
    alias_setup,
    biased_random_walk,
    build_alias_tables,
    generate_walks,
    uniform_random_walk,
)

__all__ = [
    # 子模块
    "base",
    "backends",
    "walks",
    "skipgram",
    "spectral",
    "deepwalk",
    "node2vec",
    "grarep",
    "registry",
    # 类
    "BaseEmbedder",
    "SpectralEmbedder",
    "DeepWalkEmbedder",
    "Node2VecEmbedder",
    "GraRepEmbedder",
    "SkipGramTrainer",
    "WalkSampler",
    # 游走
    "alias_setup",
    "alias_draw",
    "build_alias_tables",
    "uniform_random_walk",
    "biased_random_walk",
    "generate_walks",
    "build_negative_table",
    # 后端
    "available_backends",
    "karateclub_status",
    "node2vec_status",
    "build_karateclub_embedder",
    "resolve_backend",
    "backend_report",
    # 注册表
    "EMBEDDER_REGISTRY",
    "register_embedder",
    "list_embedders",
    "get_embedder",
    "get_embedder_class",
    "has_embedder",
]
