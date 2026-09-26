"""嵌入器注册表：``EMBEDDER_REGISTRY`` + ``list_embedders()`` / ``get_embedder()``。

所有内置实现均标注 ``backend = "internal"``（经典论文复现，非 SOTA）。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from graphforge.core.errors import UnknownMethodError
from graphforge.core.logging import get_logger

from graphforge.graph.embeddings.base import BaseEmbedder
from graphforge.graph.embeddings.deepwalk import DeepWalkEmbedder
from graphforge.graph.embeddings.grarep import GraRepEmbedder
from graphforge.graph.embeddings.node2vec import Node2VecEmbedder
from graphforge.graph.embeddings.spectral import SpectralEmbedder

__all__ = [
    "EMBEDDER_REGISTRY",
    "register_embedder",
    "list_embedders",
    "get_embedder",
    "get_embedder_class",
    "has_embedder",
]

_LOGGER = get_logger(__name__)

#: 方法名 → 嵌入器类。
EMBEDDER_REGISTRY: Dict[str, Type[BaseEmbedder]] = {
    "spectral": SpectralEmbedder,
    "deepwalk": DeepWalkEmbedder,
    "node2vec": Node2VecEmbedder,
    "grarep": GraRepEmbedder,
}


def register_embedder(
    name: str, cls: Type[BaseEmbedder], force: bool = False
) -> None:
    """注册（或覆盖）一个嵌入器。

    Args:
        name: 方法名。
        cls: 嵌入器类。
        force: 允许覆盖已存在条目。

    Raises:
        UnknownMethodError: 同名已存在且 ``force=False``。
    """
    if name in EMBEDDER_REGISTRY and not force:
        raise UnknownMethodError(
            f"嵌入器 {name!r} 已注册", hint="如需覆盖，请传 force=True"
        )
    EMBEDDER_REGISTRY[name] = cls
    _LOGGER.debug("注册嵌入器：%s -> %s", name, cls.__name__)


def list_embedders() -> List[str]:
    """返回已注册嵌入器名（升序）。"""
    return sorted(EMBEDDER_REGISTRY)


def has_embedder(name: str) -> bool:
    """判断嵌入器是否已注册。"""
    return name in EMBEDDER_REGISTRY


def get_embedder_class(name: str) -> Type[BaseEmbedder]:
    """按名字取嵌入器类。

    Args:
        name: 方法名。

    Returns:
        嵌入器类。

    Raises:
        UnknownMethodError: 方法名未注册。
    """
    if name not in EMBEDDER_REGISTRY:
        raise UnknownMethodError(
            f"未知嵌入方法 {name!r}", hint=f"可选值：{list_embedders()}"
        )
    return EMBEDDER_REGISTRY[name]


def get_embedder(
    name: str,
    dim: Optional[int] = None,
    random_state: Optional[int] = None,
    **params: Any,
) -> BaseEmbedder:
    """构造一个嵌入器实例。

    Args:
        name: 方法名（``spectral`` / ``deepwalk`` / ``node2vec`` / ``grarep``）。
        dim: 嵌入维度；None 时取全局配置 ``GRAPHFORGE_DIM``。
        random_state: 随机种子；None 时取全局配置 ``GRAPHFORGE_RANDOM_STATE``。
        **params: 透传给嵌入器构造器的超参（**不得**含保留参数）。

    Returns:
        嵌入器实例。

    Raises:
        UnknownMethodError: 方法名未注册。
    """
    from graphforge.core.config import get_config

    cls = get_embedder_class(name)
    config = get_config()
    resolved_dim = int(dim if dim is not None else config.dim)
    resolved_seed = int(random_state if random_state is not None else config.random_state)
    reserved = cls.RESERVED & set(params)
    if reserved:
        raise UnknownMethodError(
            f"参数 {sorted(reserved)} 为保留参数，不能通过 get_embedder 传入",
            hint="dim / random_state 请使用独立参数位",
        )
    return cls(dim=resolved_dim, random_state=resolved_seed, **params)
