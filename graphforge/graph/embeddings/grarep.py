"""GraRep 嵌入器。

对应论文：**Cao et al., "GraRep: Learning Graph Representations with Global
Structural Information", CIKM 2015**。

算法（按原论文 §4）：

1. 计算 k 步转移矩阵 ``A_hat^k``（``A_hat = D^-1 A`` 为行随机矩阵）；
2. 全局归一：``P = A_hat^k / sum(A_hat^k)``；
3. PPMI：``X = max(log(P / (row * col)) - log(beta), 0)``，
   其中 ``row`` / ``col`` 为 P 的行和/列和，``beta = 1 / |V|`` 为原论文的平移常数；
4. 对 PPMI 做截断 SVD，取 ``U * sqrt(S)`` 作为第 k 步的嵌入；
5. 把 k = 1..K 的嵌入**横向拼接**成最终表示。

实现细节：
    * PPMI 的 ``log(0)`` 项恒为 ``-inf`` → 被 ``max(., 0)`` 截断为 0，
      因此可以**只在非零元素上计算**，全程保持稀疏；
    * 每步维度 ``ceil(dim / K)``，拼接后裁剪/补零到 ``dim``。

作者：晨星
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import svds

from graphforge.core.errors import ConvergenceError, InvalidEmbeddingParamError
from graphforge.core.logging import get_logger
from graphforge.core.types import GraphData, ParamSpec, SearchSpace
from graphforge.core.utils import degree_vector, to_csr

from graphforge.graph.embeddings.base import BaseEmbedder

__all__ = ["GraRepEmbedder"]

_LOGGER = get_logger(__name__)


class GraRepEmbedder(BaseEmbedder):
    """GraRep 嵌入器（k 步 PPMI + SVD 拼接）。

    Attributes:
        NAME: ``"grarep"``。
    """

    NAME = "grarep"

    def __init__(
        self,
        dim: int = 128,
        k_steps: int = 4,
        random_state: int = 42,
    ) -> None:
        """构造 GraRep 嵌入器。

        Args:
            dim: 嵌入维度。
            k_steps: 最大转移步数 ``K``。
            random_state: 随机种子（传给 ``svds``）。
        """
        super().__init__(dim=dim, random_state=random_state, k_steps=k_steps)

    def _validate_hyper(self) -> None:
        """校验 ``k_steps``。

        Raises:
            InvalidEmbeddingParamError: ``k_steps`` < 1。
        """
        steps = int(self._hp("k_steps", 4))
        if steps < 1:
            raise InvalidEmbeddingParamError(
                f"k_steps={steps} 非法", hint="转移步数必须 >= 1"
            )

    def _fit(self, graph: GraphData) -> np.ndarray:
        """计算 GraRep 嵌入。

        Args:
            graph: 输入图。

        Returns:
            ``(n, dim)`` 嵌入矩阵。

        Raises:
            ConvergenceError: 所有步的 SVD 均失败且无可用分量。
        """
        num_nodes = int(graph.num_nodes)
        if num_nodes == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        steps = int(self._hp("k_steps", 4))
        per_step = max(1, int(math.ceil(self.dim / steps)))
        transition = self._transition_matrix(graph)
        blocks: List[np.ndarray] = []
        # 从单位阵出发，第一次乘完即为 A_hat^1（对应 k = 1）
        power = sp.identity(num_nodes, format="csr", dtype=np.float64)
        for _ in range(steps):
            power = (power @ transition).tocsr()
            ppmi = self._ppmi(power)
            blocks.append(self._svd_block(ppmi, per_step, num_nodes))
        if not blocks:
            raise ConvergenceError("GraRep 未产出任何嵌入分量", hint="检查 k_steps")
        embedding = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
        if embedding.shape[1] < self.dim:
            padding = np.zeros((num_nodes, self.dim - embedding.shape[1]), dtype=np.float64)
            embedding = np.hstack([embedding, padding])
        return np.asarray(embedding[:, : self.dim], dtype=np.float64)

    # ---------------------------------------------------------------- 内部
    def _transition_matrix(self, graph: GraphData) -> sp.csr_matrix:
        """构造行随机转移矩阵 ``D^-1 A``。

        Args:
            graph: 输入图。

        Returns:
            ``(n, n)`` CSR 转移矩阵（对称化、去自环后的邻接再行归一化）。
        """
        adjacency = to_csr(graph.adjacency)
        adjacency = adjacency.maximum(adjacency.T).tocsr()
        adjacency = adjacency - sp.diags(adjacency.diagonal())
        adjacency.eliminate_zeros()
        degrees = degree_vector(adjacency)
        safe = np.where(degrees > 0, degrees, 1.0)
        return (sp.diags(1.0 / safe) @ adjacency).tocsr()

    def _ppmi(self, power: sp.csr_matrix) -> sp.csr_matrix:
        """对 k 步转移矩阵做全局归一 + PPMI 变换（稀疏）。

        Args:
            power: ``A_hat^k``。

        Returns:
            PPMI 稀疏矩阵（非负）。
        """
        total = float(np.asarray(power.sum()).ravel()[0]) if power.nnz else 0.0
        if not np.isfinite(total) or total <= 0:
            return power
        data = power.data / total
        row_sums = np.asarray(power.sum(axis=1)).ravel() / total
        col_sums = np.asarray(power.sum(axis=0)).ravel() / total
        rows = np.repeat(np.arange(power.shape[0]), np.diff(power.indptr))
        cols = power.indices
        with np.errstate(divide="ignore", invalid="ignore"):
            values = (
                np.log(np.maximum(data, 1e-300))
                - np.log(np.maximum(row_sums[rows], 1e-300))
                - np.log(np.maximum(col_sums[cols], 1e-300))
                - math.log(1.0 / max(power.shape[0], 2))  # -log(beta), beta = 1/|V|
            )
        values = np.where(np.isfinite(values), values, 0.0)
        values = np.maximum(values, 0.0)
        matrix = sp.csr_matrix((values, cols, power.indptr), shape=power.shape)
        matrix.eliminate_zeros()
        return matrix

    def _svd_block(self, ppmi: sp.csr_matrix, rank: int, num_nodes: int) -> np.ndarray:
        """对 PPMI 矩阵做截断 SVD，返回 ``U * sqrt(S)``。

        Args:
            ppmi: PPMI 稀疏矩阵。
            rank: 目标维度。
            num_nodes: 节点数。

        Returns:
            ``(num_nodes, rank)`` 分块（不足补零）。
        """
        effective = max(1, min(int(rank), max(1, num_nodes - 1)))
        if ppmi.nnz == 0:
            block = np.zeros((num_nodes, effective), dtype=np.float64)
        elif num_nodes <= 2 or effective >= num_nodes - 1:
            dense = np.asarray(ppmi.todense(), dtype=np.float64)
            left, singular, _ = np.linalg.svd(dense, full_matrices=False)
            order = np.argsort(-singular)[:effective]
            block = left[:, order] * np.sqrt(np.maximum(singular[order], 0.0))
        else:
            try:
                left, singular, _ = svds(ppmi, k=effective, random_state=self._seed("svd"))
            except Exception as exc:  # noqa: BLE001 - svds 可能抛任意底层异常
                raise ConvergenceError(
                    f"GraRep 的 SVD 失败：{exc}", hint="降低 dim 或 k_steps"
                ) from exc
            order = np.argsort(-singular)
            block = left[:, order] * np.sqrt(np.maximum(singular[order], 0.0))
        if block.shape[1] < rank:
            padding = np.zeros((num_nodes, rank - block.shape[1]), dtype=np.float64)
            block = np.hstack([block, padding])
        return np.asarray(block[:, :rank], dtype=np.float64)

    def default_space(self) -> SearchSpace:
        """默认 HPO 搜索空间。"""
        return self._space(
            [
                ParamSpec("k_steps", "int", 1, 6, default=4),
            ]
        )
