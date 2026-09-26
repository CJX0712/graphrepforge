"""Skip-gram 训练器（纯 numpy 负采样 SGD）。

对标算法：**Mikolov et al., "Distributed Representations of Words and Phrases
and their Compositionality", NIPS 2013** 的负采样 skip-gram（NEG）。

说明（重要）：
    * 原 DeepWalk（KDD 2014）用的是 **hierarchical softmax**；本实现改用
      **负采样（k=5）**——这是原作者后续 word2vec 工作的标准等价替代，速度更快、
      实现更简洁。差异已在模块与 README 中显式标注。
    * 负样本按 ``degree ** 0.75``（节点频次）分布采样，与 word2vec 的
      ``freq ** 0.75`` 一致。
    * 学习率随已处理节点数线性衰减到 ``learning_rate * 1e-4``（word2vec 惯例）。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from graphforge.core.errors import EmbeddingFitError
from graphforge.core.logging import get_logger
from graphforge.core.utils import make_rng

__all__ = ["SkipGramTrainer", "build_negative_table", "sigmoid"]

_LOGGER = get_logger(__name__)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid（分段计算，避免大负数溢出）。

    Args:
        values: 输入数组。

    Returns:
        ``1 / (1 + exp(-x))``，形状同输入。
    """
    array = np.asarray(values, dtype=np.float64)
    out = np.empty_like(array)
    positive = array >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def build_negative_table(frequencies: np.ndarray) -> np.ndarray:
    """按 ``freq ** 0.75`` 构建负采样累积分布。

    Args:
        frequencies: 每个节点作为"词"的出现频次（通常用度）。

    Returns:
        ``(n,)`` 的累积概率数组，末元素为 1.0，供 ``np.searchsorted`` 使用。
    """
    array = np.asarray(frequencies, dtype=np.float64).ravel()
    if array.size == 0:
        return np.zeros(0, dtype=np.float64)
    powered = np.power(np.maximum(array, 0.0), 0.75)
    total = float(powered.sum())
    if not np.isfinite(total) or total <= 0.0:
        powered = np.ones(array.size, dtype=np.float64)
        total = float(powered.sum())
    cumulative = np.cumsum(powered) / total
    cumulative[-1] = 1.0
    return cumulative


class SkipGramTrainer:
    """负采样 skip-gram（SGD）训练器。

    Attributes:
        dim: 嵌入维度。
        window: 上下文窗口半径。
        negative: 每个正样本的负样本数。
        epochs: 训练轮数。
        learning_rate: 初始学习率。
        random_state: 随机种子。
        shrink_window: 是否使用 word2vec 的随机收缩窗口（默认 False，保证确定性与可解释性）。
    """

    def __init__(
        self,
        dim: int = 128,
        window: int = 10,
        negative: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.025,
        random_state: int = 42,
        shrink_window: bool = False,
    ) -> None:
        """构造训练器。

        Args:
            dim: 嵌入维度，必须 >= 1。
            window: 上下文窗口半径，必须 >= 1。
            negative: 负样本数，必须 >= 1。
            epochs: 训练轮数，必须 >= 1。
            learning_rate: 初始学习率，必须 > 0。
            random_state: 随机种子。
            shrink_window: 是否随机收缩窗口。

        Raises:
            EmbeddingFitError: 参数取值非法。
        """
        if dim < 1:
            raise EmbeddingFitError(f"dim={dim} 非法", hint="嵌入维度必须 >= 1")
        if window < 1:
            raise EmbeddingFitError(f"window={window} 非法", hint="窗口半径必须 >= 1")
        if negative < 1:
            raise EmbeddingFitError(f"negative={negative} 非法", hint="负样本数必须 >= 1")
        if epochs < 1:
            raise EmbeddingFitError(f"epochs={epochs} 非法", hint="训练轮数必须 >= 1")
        if learning_rate <= 0:
            raise EmbeddingFitError(
                f"learning_rate={learning_rate} 非法", hint="学习率必须 > 0"
            )
        self.dim = int(dim)
        self.window = int(window)
        self.negative = int(negative)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.shrink_window = bool(shrink_window)

    def train(
        self,
        walks: Sequence[Sequence[int]],
        num_nodes: int,
        frequencies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """训练并返回嵌入矩阵。

        Args:
            walks: 游走语料（每条为节点下标序列）。
            num_nodes: 节点总数（决定矩阵行数）。
            frequencies: 负采样频次；None 时退化为均匀分布。

        Returns:
            ``(num_nodes, dim)`` 的嵌入矩阵（float64）。

        Raises:
            EmbeddingFitError: 语料为空或节点下标越界。
        """
        if num_nodes <= 0:
            raise EmbeddingFitError("num_nodes 必须为正", hint="检查图规模")
        if not walks:
            raise EmbeddingFitError("游走语料为空", hint="检查 num_walks / walk_length")
        corpus: List[np.ndarray] = [np.asarray(walk, dtype=np.int64).ravel() for walk in walks]
        corpus = [walk for walk in corpus if walk.size >= 2]
        if not corpus:
            raise EmbeddingFitError(
                "全部游走长度均 < 2，无法训练", hint="增大 walk_length 或检查图连通性"
            )
        max_index = int(max(walk.max() for walk in corpus))
        if max_index >= int(num_nodes):
            raise EmbeddingFitError(
                f"游走中出现越界节点 {max_index} >= {num_nodes}", hint="检查节点编号"
            )

        rng = make_rng(self.random_state, "skipgram")
        if frequencies is None:
            counts = np.bincount(
                np.concatenate(corpus), minlength=int(num_nodes)
            ).astype(np.float64)
            frequencies = counts
        negative_cdf = build_negative_table(np.asarray(frequencies, dtype=np.float64))
        if negative_cdf.size == 0:
            negative_cdf = build_negative_table(np.ones(int(num_nodes)))

        scale = 0.5 / float(self.dim)
        input_vectors = rng.uniform(-scale, scale, size=(int(num_nodes), self.dim))
        output_vectors = np.zeros((int(num_nodes), self.dim), dtype=np.float64)

        total_centers = sum(int(walk.size) for walk in corpus) * self.epochs
        processed = 0
        base_lr = self.learning_rate
        for _ in range(self.epochs):
            order = rng.permutation(len(corpus))
            for position in order:
                walk = corpus[int(position)]
                length = int(walk.size)
                for center_pos in range(length):
                    center = int(walk[center_pos])
                    radius = self.window
                    if self.shrink_window:
                        radius = int(rng.integers(1, self.window + 1))
                    left = max(0, center_pos - radius)
                    right = min(length, center_pos + radius + 1)
                    if left >= center_pos and right <= center_pos + 1:
                        continue
                    contexts = np.concatenate(
                        [walk[left:center_pos], walk[center_pos + 1 : right]]
                    )
                    if contexts.size == 0:
                        continue
                    negative_count = int(contexts.size) * self.negative
                    negatives = np.searchsorted(
                        negative_cdf, rng.random(negative_count), side="right"
                    )
                    negatives = np.clip(negatives, 0, int(num_nodes) - 1).astype(np.int64)
                    targets = np.concatenate([contexts, negatives])
                    labels = np.concatenate(
                        [np.ones(contexts.size), np.zeros(negatives.size)]
                    )

                    learning_rate = base_lr * max(1.0 - processed / max(1, total_centers), 1e-4)
                    center_vector = input_vectors[center]
                    # 先取旧的输出向量副本：梯度必须用"更新前"的参数计算（严格 SGD）
                    target_vectors = output_vectors[targets]
                    scores = sigmoid(target_vectors @ center_vector)
                    grad = (labels - scores) * learning_rate
                    # np.add.at：负样本可能重复，普通 fancy-index += 只保留最后一次
                    np.add.at(output_vectors, targets, np.outer(grad, center_vector))
                    input_vectors[center] = center_vector + grad @ target_vectors
                    processed += 1
        _LOGGER.debug(
            "skip-gram 训练完成：%d 游走 / %d 中心节点 / dim=%d",
            len(corpus),
            processed,
            self.dim,
        )
        return input_vectors

    def fit_transform(
        self,
        walks: Sequence[Sequence[int]],
        num_nodes: int,
        frequencies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """:meth:`train` 的别名（对齐 sklearn 习惯）。"""
        return self.train(walks, num_nodes, frequencies=frequencies)

    def get_params(self) -> Dict[str, Any]:
        """返回训练器参数（不含 ``dim``，避免与嵌入器保留参数混淆）。"""
        return {
            "window": self.window,
            "negative": self.negative,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "shrink_window": self.shrink_window,
        }
