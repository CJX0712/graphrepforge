"""嵌入器抽象基类：Template Method 固化通用流程，子类只实现 ``_fit``。

固化流程：``fit`` → ``_fit``（子类）→ 形状校验 → ``transform`` 产出 :class:`Embedding`。

**参数泄漏防护（架构 §10 坑 3）**：
``dim`` / ``random_state`` / ``backend`` 等内部参数由构造器单独接收，
:meth:`set_params` 一旦命中 :attr:`RESERVED` 立即抛 ``E303``。

作者：晨星
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from graphforge.core.errors import (
    EmbeddingFitError,
    InvalidEmbeddingParamError,
    PipelineStateError,
)
from graphforge.core.logging import get_logger
from graphforge.core.types import Embedding, GraphData, ParamSpec, SearchSpace
from graphforge.core.utils import derive_seed

__all__ = ["BaseEmbedder"]

_LOGGER = get_logger(__name__)


class BaseEmbedder(ABC):
    """图嵌入器抽象基类。

    Attributes:
        NAME: 方法名（子类必须覆盖，用于注册表与结果标记）。
        RESERVED: 保留参数集合，``set_params`` 命中即抛 ``E303``。
        dim: 嵌入维度（内部参数）。
        random_state: 随机种子（内部参数）。

    Examples:
        >>> class Dummy(BaseEmbedder):
        ...     NAME = "dummy"
        ...     def _fit(self, graph): return np.zeros((graph.num_nodes, self.dim))
        >>> emb = Dummy(dim=8).fit_transform(graph)  # doctest: +SKIP
    """

    #: 方法名；子类必须覆盖。
    NAME: str = "base"

    #: 保留参数：不允许通过 ``set_params`` 修改（防内部参数泄漏）。
    RESERVED = frozenset(
        {"dim", "n_classes", "num_nodes", "n_nodes", "random_state", "backend", "name"}
    )

    def __init__(self, dim: int = 128, random_state: int = 42, **hyper: Any) -> None:
        """构造嵌入器。

        Args:
            dim: 嵌入维度，必须 >= 1。
            random_state: 随机种子。
            **hyper: 方法自身的超参（会进入 :meth:`get_params`）。

        Raises:
            InvalidEmbeddingParamError: ``dim`` 非法或超参命中保留名。
        """
        if int(dim) < 1:
            raise InvalidEmbeddingParamError(
                f"dim={dim} 非法", hint="嵌入维度必须 >= 1"
            )
        leaked = sorted(set(hyper) & self.RESERVED)
        if leaked:
            raise InvalidEmbeddingParamError(
                f"构造参数 {leaked} 属于保留参数，不能作为超参传入",
                hint="保留参数请走构造器单独参数（如 dim=、random_state=）",
            )
        self.dim: int = int(dim)
        self.random_state: int = int(random_state)
        self._hyper: Dict[str, Any] = dict(hyper)
        self._matrix: Optional[np.ndarray] = None
        self._node_ids: Optional[List[str]] = None
        self._validate_hyper()

    # ---------------------------------------------------------------- 元信息
    @property
    def name(self) -> str:
        """方法名。"""
        return self.NAME

    @property
    def backend(self) -> str:
        """后端名；内置实现恒为 ``"internal"``。"""
        return "internal"

    def _validate_hyper(self) -> None:
        """超参自检钩子；子类可覆盖以加入自己的取值校验（违规抛 ``E303``）。"""
        return None

    def _hp(self, key: str, default: Any = None) -> Any:
        """读取超参（带默认值）。"""
        return self._hyper.get(key, default)

    def _seed(self, *tags: str) -> int:
        """派生本阶段的确定性子种子。

        Args:
            *tags: 阶段标签（``walk`` / ``svd`` / ``skipgram`` 等）。

        Returns:
            派生后的整数种子。
        """
        return derive_seed(self.random_state, *tags)

    # ---------------------------------------------------------------- 模板方法
    def fit(self, graph: GraphData) -> "BaseEmbedder":
        """在给定图上训练嵌入。

        Args:
            graph: 输入图（会先 ``validate()``）。

        Returns:
            ``self``，便于链式调用。

        Raises:
            EmbeddingFitError: ``_fit`` 返回矩阵形状/数值不合法。
        """
        graph.validate()
        matrix = self._fit(graph)
        matrix = np.asarray(matrix, dtype=np.float64)
        expected = (graph.num_nodes, self.dim)
        if matrix.ndim != 2 or matrix.shape != expected:
            raise EmbeddingFitError(
                f"{self.name} 产出矩阵形状 {matrix.shape} 与期望 {expected} 不符",
                hint="检查子类 _fit 的返回值",
            )
        if not np.all(np.isfinite(matrix)):
            raise EmbeddingFitError(
                f"{self.name} 产出矩阵含 NaN/Inf", hint="检查归一化与学习率"
            )
        self._matrix = matrix
        self._node_ids = list(graph.node_ids)
        _LOGGER.debug("%s 嵌入完成：%s", self.name, matrix.shape)
        return self

    def transform(self) -> Embedding:
        """返回 :meth:`fit` 产出的嵌入。

        Returns:
            :class:`Embedding`。

        Raises:
            PipelineStateError: 尚未调用 :meth:`fit`。
        """
        if self._matrix is None or self._node_ids is None:
            raise PipelineStateError(
                f"{self.name} 尚未 fit，无法 transform", hint="先调用 fit(graph) 或 fit_transform"
            )
        return Embedding(
            matrix=self._matrix.copy(),
            node_ids=list(self._node_ids),
            method=self.name,
            backend=self.backend,
            params=self.snapshot_params(),
        )

    def fit_transform(self, graph: GraphData) -> Embedding:
        """训练并直接返回嵌入。

        Args:
            graph: 输入图。

        Returns:
            :class:`Embedding`。
        """
        return self.fit(graph).transform()

    # ---------------------------------------------------------------- 参数
    def get_params(self) -> Dict[str, Any]:
        """返回**超参**字典（**不含** ``dim`` / ``random_state`` 等保留参数）。"""
        return dict(self._hyper)

    def snapshot_params(self) -> Dict[str, Any]:
        """返回运行快照参数（含 ``dim`` / ``random_state``，仅用于结果追溯）。"""
        return {**dict(self._hyper), "dim": self.dim, "random_state": self.random_state}

    def set_params(self, **params: Any) -> "BaseEmbedder":
        """就地修改超参。

        Args:
            **params: 待修改的超参。

        Returns:
            ``self``。

        Raises:
            InvalidEmbeddingParamError: 命中保留参数或未知参数。
        """
        reserved = sorted(set(params) & self.RESERVED)
        if reserved:
            raise InvalidEmbeddingParamError(
                f"参数 {reserved} 为保留参数，禁止通过 set_params 修改",
                hint="dim / random_state 等请在构造器中传入",
            )
        unknown = sorted(set(params) - set(self._hyper))
        if unknown:
            raise InvalidEmbeddingParamError(
                f"{self.name} 不支持参数 {unknown}",
                hint=f"可用超参：{sorted(self._hyper)}",
            )
        self._hyper.update(params)
        self._validate_hyper()
        return self

    def default_space(self) -> SearchSpace:
        """默认搜索空间（供 HPO 使用）；子类可覆盖。

        Returns:
            默认返回空 :class:`SearchSpace`（不参与搜索）。
        """
        return SearchSpace(params=())

    # ---------------------------------------------------------------- 抽象
    @abstractmethod
    def _fit(self, graph: GraphData) -> np.ndarray:
        """子类实现：返回 ``(n, dim)`` 的嵌入矩阵。

        Args:
            graph: 输入图（已校验）。

        Returns:
            ``(n, dim)`` 的 float 矩阵，行序与 ``graph.node_ids`` 一致。
        """
        raise NotImplementedError

    # ---------------------------------------------------------------- 辅助
    @staticmethod
    def _space(specs: List[ParamSpec]) -> SearchSpace:
        """用给定 :class:`ParamSpec` 列表构造搜索空间（子类便捷方法）。"""
        return SearchSpace(params=tuple(specs))

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        """返回形如 ``Node2VecEmbedder(dim=128, p=1.0, q=1.0)`` 的表示。"""
        inner = ", ".join(f"{key}={value!r}" for key, value in sorted(self._hyper.items()))
        return f"{type(self).__name__}(dim={self.dim}, random_state={self.random_state}{', ' + inner if inner else ''})"
