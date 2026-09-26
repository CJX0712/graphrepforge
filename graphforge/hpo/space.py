"""超参搜索空间（L2，依赖 core）。

:class:`ParamSpec` / :class:`SearchSpace` 的权威定义在
:mod:`graphforge.core.types`（保证 core 可独立验证）；本模块做**再导出**并
提供「嵌入器默认空间」的便捷访问。

作者：晨星
"""

from __future__ import annotations

from graphforge.core.errors import UnknownMethodError
from graphforge.core.types import ParamSpec, SearchSpace

__all__ = [
    "ParamSpec",
    "SearchSpace",
    "embedder_space",
]


def embedder_space(method: str) -> SearchSpace:
    """取指定嵌入器的默认搜索空间。

    Args:
        method: 嵌入方法名（``spectral`` / ``deepwalk`` / ``node2vec`` / ``grarep``）。

    Returns:
        :class:`SearchSpace`。

    Raises:
        UnknownMethodError: 方法名未注册。
    """
    from graphforge.graph.embeddings.registry import get_embedder

    try:
        embedder = get_embedder(method)
    except UnknownMethodError:
        raise
    return embedder.default_space()
