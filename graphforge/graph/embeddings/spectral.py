"""谱嵌入（Laplacian Eigenmaps）。

对应论文：**Belkin & Niyogi, "Laplacian Eigenmaps for Dimensionality Reduction
and Data Representation", Neural Computation 2003**。

数学等价：归一化图拉普拉斯 ``L = I - D^-1/2 A D^-1/2`` 的前 ``k`` 个**最小非平凡**
特征向量，等于归一化邻接矩阵 ``A_sym`` 的第 ``2..k+1`` 个**最大**特征向量。
本实现直接对 ``A_sym`` 求谱分解（数值更稳定），丢弃第 1 个（常数向量 / 平凡解）。

规模策略：
    * ``n <= 1500``：稠密 ``np.linalg.eigh``（精确、无需随机种子、天然可复现）；
    * ``n >  1500``：稀疏 ``scipy.sparse.linalg.svds``（**必须显式 random_state**）。

作者：晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import svds

from graphforge.core.errors import ConvergenceError, InvalidEmbeddingParamError
from graphforge.core.logging import get_logger
from graphforge.core.types import GraphData, ParamSpec, SearchSpace
from graphforge.core.utils import degree_vector, to_csr
from graphforge.preprocess.build import normalize_adjacency, symmetrize_adjacency

from graphforge.graph.embeddings.base import BaseEmbedder

__all__ = ["SpectralEmbedder", "SPECTRAL_NORMALIZATIONS"]

_LOGGER = get_logger(__name__)

#: 支持的归一化方式（对应是否使用归一化拉普拉斯）。
SPECTRAL_NORMALIZATIONS = ("symmetric", "none")

#: 稠密分解的规模上限。
DENSE_LIMIT: int = 1500


class SpectralEmbedder(BaseEmbedder):
    """拉普拉斯特征映射嵌入器。

    Attributes:
        NAME: ``"spectral"``。
    """

    NAME = "spectral"

    def __init__(
        self,
        dim: int = 128,
        normalize: str = "symmetric",
        random_state: int = 42,
    ) -> None:
        """构造谱嵌入器。

        Args:
            dim: 嵌入维度。
            normalize: ``"symmetric"`` 用归一化拉普拉斯（默认，对应原论文）；
                ``"none"`` 直接用邻接矩阵做特征映射。
            random_state: 随机种子（稠密路径不使用，稀疏路径传给 ``svds``）。
        """
        super().__init__(dim=dim, random_state=random_state, normalize=normalize)

    def _validate_hyper(self) -> None:
        """校验归一化方式。

        Raises:
            InvalidEmbeddingParamError: 取值非法。
        """
        mode = str(self._hp("normalize", "symmetric"))
        if mode not in SPECTRAL_NORMALIZATIONS:
            raise InvalidEmbeddingParamError(
                f"normalize={mode!r} 非法", hint=f"可选值：{list(SPECTRAL_NORMALIZATIONS)}"
            )

    def _fit(self, graph: GraphData) -> np.ndarray:
        """计算谱嵌入。

        Args:
            graph: 输入图。

        Returns:
            ``(n, dim)`` 嵌入矩阵。

        Raises:
            ConvergenceError: 稀疏 SVD 未收敛。
        """
        adjacency = symmetrize_adjacency(to_csr(graph.adjacency), method="max")
        num_nodes = int(graph.num_nodes)
        mode = str(self._hp("normalize", "symmetric"))
        target = normalize_adjacency(adjacency, "symmetric") if mode == "symmetric" else adjacency

        if num_nodes == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        if num_nodes == 1:
            return np.zeros((1, self.dim), dtype=np.float64)

        wanted = self.dim + 1  # +1 是为了拿到随后被丢弃的平凡向量
        if num_nodes <= DENSE_LIMIT:
            matrix = target.toarray()
            values, vectors = np.linalg.eigh(matrix)
            order = np.argsort(-values)  # 降序：取最大特征值
            selected = order[:wanted][1:]  # 丢弃第 1 个（常数向量）
            embedding = vectors[:, selected]
        else:
            rank = max(1, min(wanted, num_nodes - 1))
            try:
                left, singular, _ = svds(target, k=rank, random_state=self._seed("svd"))
            except Exception as exc:  # noqa: BLE001 - svds 在病态矩阵上可能抛任意异常
                raise ConvergenceError(
                    f"稀疏 SVD 失败：{exc}", hint="降低 dim 或先取最大连通分量"
                ) from exc
            if left.size == 0 or singular.size == 0:
                raise ConvergenceError(
                    "稀疏 SVD 返回空结果", hint="检查图规模与 dim"
                )
            order = np.argsort(-singular)
            embedding = left[:, order]
            if embedding.shape[1] > 1:
                embedding = embedding[:, 1:]
            else:
                embedding = embedding[:, :0]

        embedding = np.asarray(embedding, dtype=np.float64)
        if embedding.shape[1] < self.dim:
            padding = np.zeros((num_nodes, self.dim - embedding.shape[1]), dtype=np.float64)
            embedding = np.hstack([embedding, padding])
        return embedding[:, : self.dim]

    def default_space(self) -> SearchSpace:
        """默认搜索空间（谱方法只有归一化方式可调）。"""
        return self._space(
            [
                ParamSpec(
                    name="normalize",
                    kind="categorical",
                    choices=list(SPECTRAL_NORMALIZATIONS),
                    default="symmetric",
                )
            ]
        )
