"""数据集注册表：``DATASET_REGISTRY`` + ``list_datasets()`` / ``get_dataset()``。

设计要点：
    * 只注册**合成图**（离线环境不下载 Cora/Citeseer 等外部数据集，见架构 §11 A4）；
    * 名字 → 生成函数，调用方通过 ``get_dataset("sbm", random_state=...)`` 取图；
    * 未知名字抛 ``E101``（数据集不存在）。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from graphforge.core.errors import DatasetNotFoundError
from graphforge.core.logging import get_logger
from graphforge.core.types import GraphData

from graphforge.data.synthetic import (
    generate_grid,
    generate_karate,
    generate_lfr,
    generate_sbm,
)

__all__ = [
    "DATASET_REGISTRY",
    "register_dataset",
    "list_datasets",
    "get_dataset",
    "has_dataset",
]

_LOGGER = get_logger(__name__)

#: 名称 → 生成函数（惰性调用，注册时不建图）。
DATASET_REGISTRY: Dict[str, Callable[..., GraphData]] = {
    "sbm": generate_sbm,
    "karate": generate_karate,
    "grid": generate_grid,
    "lfr": generate_lfr,
}


def register_dataset(name: str, factory: Callable[..., GraphData], force: bool = False) -> None:
    """注册（或覆盖）一个数据集生成器。

    Args:
        name: 数据集名。
        factory: 生成函数，签名兼容 ``(**kwargs) -> GraphData``。
        force: 为 True 时允许覆盖已存在条目。

    Raises:
        DatasetNotFoundError: 同名已存在且 ``force=False``（复用 E101 以简化调用方处理）。
    """
    if name in DATASET_REGISTRY and not force:
        raise DatasetNotFoundError(
            f"数据集 {name!r} 已注册", hint="如需覆盖，请传 force=True"
        )
    DATASET_REGISTRY[name] = factory
    _LOGGER.debug("注册数据集：%s", name)


def list_datasets() -> List[str]:
    """返回已注册数据集名（升序）。"""
    return sorted(DATASET_REGISTRY)


def has_dataset(name: str) -> bool:
    """判断数据集是否已注册。"""
    return name in DATASET_REGISTRY


def get_dataset(name: str, random_state: Optional[int] = None, **kwargs: Any) -> GraphData:
    """取一个数据集（默认是合成图，惰性生成）。

    Args:
        name: 数据集名（``sbm`` / ``karate`` / ``grid`` / ``lfr``）。
        random_state: 随机种子；None 时透传由生成器取默认（42），
            这里显式从全局配置取，保证"同 seed 同图"。
        **kwargs: 透传给生成器的其他参数。

    Returns:
        :class:`GraphData`。

    Raises:
        DatasetNotFoundError: 数据集名未注册。
    """
    if name not in DATASET_REGISTRY:
        raise DatasetNotFoundError(
            f"未知数据集 {name!r}", hint=f"可选值：{list_datasets()}"
        )
    if random_state is None:
        from graphforge.core.config import get_config

        random_state = get_config().random_state
    kwargs.setdefault("random_state", int(random_state))
    _LOGGER.info("载入数据集：%s（random_state=%s）", name, random_state)
    return DATASET_REGISTRY[name](**kwargs)
