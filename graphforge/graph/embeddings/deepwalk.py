"""DeepWalk 嵌入器。

对应论文：**Perozzi et al., "DeepWalk: Online Learning of Social
Representations", KDD 2014**；训练部分使用 **Mikolov et al., NIPS 2013** 的
负采样 skip-gram（原论文为 hierarchical softmax，此处为等价替代，见
:mod:`graphforge.graph.embeddings.skipgram` 的说明）。

流程：均匀随机游走（alias table O(1) 采样）→ skip-gram 训练 → ``(n, dim)``。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from graphforge.core.errors import InvalidEmbeddingParamError
from graphforge.core.logging import get_logger
from graphforge.core.types import GraphData, ParamSpec, SearchSpace
from graphforge.core.utils import degree_vector, make_rng, to_csr
from graphforge.preprocess.build import symmetrize_adjacency

from graphforge.graph.embeddings.base import BaseEmbedder
from graphforge.graph.embeddings.skipgram import SkipGramTrainer
from graphforge.graph.embeddings.walks import generate_walks

__all__ = ["DeepWalkEmbedder"]

_LOGGER = get_logger(__name__)


class DeepWalkEmbedder(BaseEmbedder):
    """DeepWalk 嵌入器（均匀游走 + 负采样 skip-gram）。

    Attributes:
        NAME: ``"deepwalk"``。
    """

    NAME = "deepwalk"

    def __init__(
        self,
        dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        window: int = 10,
        epochs: int = 1,
        learning_rate: float = 0.025,
        negative: int = 5,
        random_state: int = 42,
    ) -> None:
        """构造 DeepWalk 嵌入器。

        Args:
            dim: 嵌入维度。
            walk_length: 每条游走长度。
            num_walks: 每个节点的游走条数。
            window: skip-gram 上下文窗口半径。
            epochs: 训练轮数。
            learning_rate: 初始学习率。
            negative: 负样本数。
            random_state: 随机种子。
        """
        super().__init__(
            dim=dim,
            random_state=random_state,
            walk_length=walk_length,
            num_walks=num_walks,
            window=window,
            epochs=epochs,
            learning_rate=learning_rate,
            negative=negative,
        )

    def _validate_hyper(self) -> None:
        """校验超参取值（违规抛 ``E303``）。"""
        checks = {
            "walk_length": (2, 10000),
            "num_walks": (1, 1000),
            "window": (1, 100),
            "epochs": (1, 100),
            "negative": (1, 100),
        }
        for key, (low, high) in checks.items():
            value = int(self._hp(key, low))
            if value < low or value > high:
                raise InvalidEmbeddingParamError(
                    f"{key}={value} 超出合理范围 [{low}, {high}]",
                    hint="检查 DeepWalk 超参",
                )
        rate = float(self._hp("learning_rate", 0.025))
        if rate <= 0 or rate > 1:
            raise InvalidEmbeddingParamError(
                f"learning_rate={rate} 非法", hint="学习率应在 (0, 1] 内"
            )

    def _fit(self, graph: GraphData) -> np.ndarray:
        """生成游走并训练 skip-gram。

        Args:
            graph: 输入图。

        Returns:
            ``(n, dim)`` 嵌入矩阵。
        """
        adjacency = symmetrize_adjacency(to_csr(graph.adjacency), method="max")
        rng = make_rng(self.random_state, "walk")
        walks = generate_walks(
            adjacency,
            num_walks=int(self._hp("num_walks", 10)),
            walk_length=int(self._hp("walk_length", 80)),
            rng=rng,
        )
        trainer = SkipGramTrainer(
            dim=self.dim,
            window=int(self._hp("window", 10)),
            negative=int(self._hp("negative", 5)),
            epochs=int(self._hp("epochs", 1)),
            learning_rate=float(self._hp("learning_rate", 0.025)),
            random_state=self.random_state,
        )
        frequencies = degree_vector(adjacency)
        return trainer.train(walks, int(graph.num_nodes), frequencies=frequencies)

    def default_space(self) -> SearchSpace:
        """默认 HPO 搜索空间。"""
        return self._space(
            [
                ParamSpec("walk_length", "int", 20, 80, default=80),
                ParamSpec("num_walks", "int", 5, 20, default=10),
                ParamSpec("window", "int", 2, 10, default=10),
                ParamSpec("learning_rate", "log_float", 0.005, 0.1, default=0.025),
                ParamSpec("negative", "int", 3, 10, default=5),
            ]
        )
