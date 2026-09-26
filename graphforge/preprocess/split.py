"""数据集划分：节点（分层 / 随机）与边（含负采样）。

统一约定（供任务层消费）：
    * ``Split.train_idx / val_idx / test_idx`` 是**样本下标**——
      节点分类指向节点，链路预测指向 ``Split.extra["edges"]`` 的行；
    * ``Split.y`` 与"全部样本"对齐；
    * 链路预测遵循**标准防泄漏做法**：训练图只含训练集正边，
      验证/测试的正负边都从训练图中移除后再评估（见 :class:`EdgeSplitter`）。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from graphforge.core.errors import (
    InsufficientSamplesError,
    InvalidSplitRatioError,
    NegativeSamplingError,
)
from graphforge.core.logging import get_logger
from graphforge.core.types import Split, SplitStrategy, TaskSpec, TaskType
from graphforge.core.utils import make_rng

from graphforge.preprocess.build import adjacency_from_edges, edge_list

__all__ = [
    "validate_ratios",
    "allocate_counts",
    "sample_negative_edges",
    "RandomNodeSplitter",
    "StratifiedNodeSplitter",
    "EdgeSplitter",
    "SPLITTER_REGISTRY",
    "get_splitter",
    "list_splitters",
]

_LOGGER = get_logger(__name__)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    """校验比例三元组。

    Args:
        train_ratio: 训练集比例。
        val_ratio: 验证集比例。
        test_ratio: 测试集比例。

    Returns:
        规范化后的 ``(train, val, test)``。

    Raises:
        InvalidSplitRatioError: 存在负值、和偏离 1.0、训练集为 0。
    """
    for name, value in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if value < 0:
            raise InvalidSplitRatioError(f"{name}={value} 不能为负", hint="比例必须在 [0, 1] 内")
    total = float(train_ratio + val_ratio + test_ratio)
    if abs(total - 1.0) > 1e-6:
        raise InvalidSplitRatioError(
            f"比例之和为 {total:.6f}，应为 1.0", hint="调整 train/val/test"
        )
    if train_ratio <= 0:
        raise InvalidSplitRatioError("train_ratio 必须为正", hint="训练集不能为空")
    return (float(train_ratio), float(val_ratio), float(test_ratio))


def allocate_counts(total: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    """把 ``total`` 个样本按 ``ratios`` 切成三份（先切 train，再在剩余中切 val/test）。

    总量 >= 3 时保证三份各至少 1 个；总量 < 3 时尽量保证 train >= 1。

    Args:
        total: 样本总数。
        ratios: ``(train, val, test)`` 比例。

    Returns:
        ``(n_train, n_val, n_test)``，三者之和恒为 ``total``。
    """
    if total <= 0:
        return (0, 0, 0)
    train_ratio, val_ratio, test_ratio = float(ratios[0]), float(ratios[1]), float(ratios[2])
    n_train = int(round(total * train_ratio))
    n_train = max(0, min(n_train, total))
    rest = total - n_train
    denominator = val_ratio + test_ratio
    n_val = 0 if denominator <= 0 else int(round(rest * val_ratio / denominator))
    n_val = max(0, min(n_val, rest))
    n_test = rest - n_val

    if total >= 3:
        buckets: List[int] = [n_train, n_val, n_test]
        for index in range(3):
            if buckets[index] >= 1:
                continue
            donor = int(np.argmax(buckets))
            if donor == index or buckets[donor] <= 1:
                continue
            buckets[donor] -= 1
            buckets[index] += 1
        n_train, n_val, n_test = buckets[0], buckets[1], buckets[2]
    elif n_train == 0:
        n_train = total
        n_val = 0
        n_test = 0
    return (n_train, n_val, n_test)


def sample_negative_edges(
    num_nodes: int,
    forbidden: Set[Tuple[int, int]],
    count: int,
    rng: "np.random.Generator",
    max_attempts_factor: int = 50,
) -> np.ndarray:
    """采样不存在的边作为负样本（同时写入 ``forbidden``，保证不重复）。

    Args:
        num_nodes: 节点数。
        forbidden: 已有边集合（键为 ``(min, max)``），**会被就地更新**。
        count: 需要的负边数。
        rng: 随机数发生器。
        max_attempts_factor: 尝试次数上限倍数（``count * factor + 1000``）。

    Returns:
        ``(count, 2)`` 的 int 数组。

    Raises:
        NegativeSamplingError: 候选耗尽仍不够。
    """
    if count <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    max_pairs = num_nodes * (num_nodes - 1) // 2
    if count > max_pairs - len(forbidden):
        raise NegativeSamplingError(
            f"需要 {count} 条负边，但可用候选仅 {max_pairs - len(forbidden)} 条",
            hint="降低 negative_ratio 或增大图规模",
        )
    collected: List[Tuple[int, int]] = []
    attempts = 0
    limit = count * int(max_attempts_factor) + 1000
    while len(collected) < count and attempts < limit:
        attempts += 1
        left = int(rng.integers(0, num_nodes))
        right = int(rng.integers(0, num_nodes))
        if left == right:
            continue
        pair = (min(left, right), max(left, right))
        if pair in forbidden:
            continue
        forbidden.add(pair)
        collected.append(pair)
    if len(collected) < count:
        raise NegativeSamplingError(
            f"负采样在 {attempts} 次尝试后只得到 {len(collected)}/{count} 条",
            hint="降低 negative_ratio 或增大图规模",
        )
    return np.asarray(collected, dtype=np.int64)


def _to_key_set(edges: np.ndarray) -> Set[Tuple[int, int]]:
    """把 ``(m, 2)`` 边数组转为 ``(min, max)`` 键集合。"""
    if edges.size == 0:
        return set()
    array = np.asarray(edges, dtype=np.int64)
    return {(int(min(row[0], row[1])), int(max(row[0], row[1]))) for row in array}


# ---------------------------------------------------------------- 节点划分
class RandomNodeSplitter:
    """随机节点划分（不保证类别比例）。"""

    #: 划分策略枚举值。
    strategy: SplitStrategy = SplitStrategy.RANDOM

    def __init__(self, random_state: int = 42, shuffle: bool = True) -> None:
        """构造划分器。

        Args:
            random_state: 随机种子。
            shuffle: 是否打乱节点顺序。
        """
        self.random_state = int(random_state)
        self.shuffle = bool(shuffle)

    def split(self, graph: Any, spec: TaskSpec) -> Split:
        """划分节点。

        Args:
            graph: 输入图。
            spec: 任务规格。

        Returns:
            :class:`Split`，``y`` 为节点标签（可能为 None）。
        """
        spec.validate()
        num_nodes = int(graph.num_nodes)
        if num_nodes < 3:
            raise InsufficientSamplesError(
                f"节点数 {num_nodes} 不足以划分", hint="至少需要 3 个节点"
            )
        rng = make_rng(spec.random_state, "split")
        order = rng.permutation(num_nodes) if self.shuffle else np.arange(num_nodes)
        n_train, n_val, n_test = allocate_counts(num_nodes, spec.ratios)
        train_idx = np.sort(order[:n_train].astype(np.int64))
        val_idx = np.sort(order[n_train : n_train + n_val].astype(np.int64))
        test_idx = np.sort(order[n_train + n_val :].astype(np.int64))
        labels = None if graph.node_labels is None else np.asarray(graph.node_labels).copy()
        return Split(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            y=labels,
            extra={"strategy": self.strategy.value},
        )


class StratifiedNodeSplitter(RandomNodeSplitter):
    """分层节点划分（**逐类别**按比例切，保证 train/val/test 类别分布一致）。

    图中无标签时自动退化为 :class:`RandomNodeSplitter` 的行为并记 WARNING。
    """

    strategy: SplitStrategy = SplitStrategy.STRATIFIED

    def split(self, graph: Any, spec: TaskSpec) -> Split:
        """按类别分层划分节点。

        Args:
            graph: 输入图。
            spec: 任务规格。

        Returns:
            :class:`Split`。
        """
        spec.validate()
        if graph.node_labels is None or not graph.has_labels:
            _LOGGER.warning("图 %s 无标签，分层划分退化为随机划分", graph.name)
            return super().split(graph, spec)

        labels = np.asarray(graph.node_labels)
        rng = make_rng(spec.random_state, "split")
        trains: List[np.ndarray] = []
        vals: List[np.ndarray] = []
        tests: List[np.ndarray] = []
        for label in np.unique(labels[labels >= 0]):
            members = np.flatnonzero(labels == label)
            if members.size == 0:
                continue
            order = members[rng.permutation(members.size)] if self.shuffle else members
            n_train, n_val, n_test = allocate_counts(int(order.size), spec.ratios)
            trains.append(order[:n_train])
            vals.append(order[n_train : n_train + n_val])
            tests.append(order[n_train + n_val :])
        unlabeled = np.flatnonzero(labels < 0)
        if unlabeled.size:
            trains.append(unlabeled)

        train_idx = np.sort(np.concatenate(trains).astype(np.int64)) if trains else np.zeros(0, np.int64)
        val_idx = np.sort(np.concatenate(vals).astype(np.int64)) if vals else np.zeros(0, np.int64)
        test_idx = np.sort(np.concatenate(tests).astype(np.int64)) if tests else np.zeros(0, np.int64)
        if train_idx.size == 0 or test_idx.size == 0:
            raise InsufficientSamplesError(
                "分层划分后训练集或测试集为空", hint="增大图规模或调整比例"
            )
        return Split(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            y=labels.copy(),
            extra={"strategy": self.strategy.value},
        )


# ---------------------------------------------------------------- 边划分
class EdgeSplitter:
    """边划分（含负采样）。

    流程（标准防泄漏做法）：

    1. 取无向图的全部无向边（上三角），随机打乱后按比例切成 train/val/test **正边**；
    2. 对每一份分别采样 ``negative_ratio`` 倍的**负边**（不存在的节点对），
       且负边不得与**任何**正边重叠；
    3. ``extra["train_adjacency"]`` 只由训练集正边构成——**嵌入/训练阶段应使用该图**，
       验证与测试边不出现在其中。
    """

    strategy: SplitStrategy = SplitStrategy.EDGE

    def __init__(
        self,
        negative_ratio: float = 1.0,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> None:
        """构造边划分器。

        Args:
            negative_ratio: 负样本倍率（相对正边数）。
            random_state: 随机种子。
            shuffle: 是否打乱边顺序。
        """
        self.negative_ratio = float(negative_ratio)
        self.random_state = int(random_state)
        self.shuffle = bool(shuffle)

    def split(self, graph: Any, spec: TaskSpec) -> Split:
        """划分边并采样负边。

        Args:
            graph: 输入图。
            spec: 任务规格（``negative_ratio`` 优先取 spec，其次取构造参数）。

        Returns:
            :class:`Split`；``train_idx/val_idx/test_idx`` 为 ``extra["edges"]`` 的行下标。

        Raises:
            InsufficientSamplesError: 边数太少。
        """
        spec.validate()
        edges = edge_list(graph)
        num_edges = int(edges.shape[0])
        if num_edges < 3:
            raise InsufficientSamplesError(
                f"边数 {num_edges} 不足以划分", hint="至少需要 3 条边"
            )
        ratio = float(spec.negative_ratio) if spec.negative_ratio > 0 else self.negative_ratio
        rng = make_rng(spec.random_state, "split", "edge")
        order = rng.permutation(num_edges) if self.shuffle else np.arange(num_edges)
        edges = edges[order]
        n_train, n_val, n_test = allocate_counts(num_edges, spec.ratios)

        train_pos = edges[:n_train]
        val_pos = edges[n_train : n_train + n_val]
        test_pos = edges[n_train + n_val :]

        forbidden = _to_key_set(edges)
        neg_rng = make_rng(spec.random_state, "neg")
        train_neg = sample_negative_edges(
            int(graph.num_nodes), forbidden, int(round(n_train * ratio)), neg_rng
        )
        val_neg = sample_negative_edges(
            int(graph.num_nodes), forbidden, int(round(n_val * ratio)), neg_rng
        )
        test_neg = sample_negative_edges(
            int(graph.num_nodes), forbidden, int(round(n_test * ratio)), neg_rng
        )

        all_edges = np.vstack([train_pos, train_neg, val_pos, val_neg, test_pos, test_neg])
        labels = np.concatenate(
            [
                np.ones(train_pos.shape[0], dtype=np.int64),
                np.zeros(train_neg.shape[0], dtype=np.int64),
                np.ones(val_pos.shape[0], dtype=np.int64),
                np.zeros(val_neg.shape[0], dtype=np.int64),
                np.ones(test_pos.shape[0], dtype=np.int64),
                np.zeros(test_neg.shape[0], dtype=np.int64),
            ]
        )
        train_stop = train_pos.shape[0] + train_neg.shape[0]
        val_stop = train_stop + val_pos.shape[0] + val_neg.shape[0]
        split = Split(
            train_idx=np.arange(0, train_stop, dtype=np.int64),
            val_idx=np.arange(train_stop, val_stop, dtype=np.int64),
            test_idx=np.arange(val_stop, all_edges.shape[0], dtype=np.int64),
            y=labels,
            extra={
                "strategy": self.strategy.value,
                "edges": all_edges,
                "positive_edges": np.vstack([train_pos, val_pos, test_pos]),
                "negative_edges": np.vstack([train_neg, val_neg, test_neg]),
                "train_edges": train_pos,
                # 训练期邻接矩阵：只含训练集正边（防泄漏，见类 docstring）
                "train_adjacency": adjacency_from_edges(int(graph.num_nodes), train_pos),
                "edge_op": spec.edge_op,
            },
        )
        _LOGGER.debug(
            "边划分：正边 %d（train/val/test = %d/%d/%d），负边 %d",
            num_edges,
            n_train,
            n_val,
            n_test,
            train_neg.shape[0] + val_neg.shape[0] + test_neg.shape[0],
        )
        return split


# ---------------------------------------------------------------- 注册表
SPLITTER_REGISTRY: Dict[str, Any] = {
    "stratified": StratifiedNodeSplitter,
    "random": RandomNodeSplitter,
    "edge": EdgeSplitter,
}


def list_splitters() -> List[str]:
    """返回划分器名（升序）。"""
    return sorted(SPLITTER_REGISTRY)


def get_splitter(
    name: Union[str, TaskType, SplitStrategy], random_state: int = 42, **kwargs: Any
) -> Any:
    """按名字或任务类型取划分器实例。

    Args:
        name: ``"stratified"`` / ``"random"`` / ``"edge"``，或 :class:`TaskType` /
            :class:`SplitStrategy`（自动映射）。
        random_state: 随机种子。
        **kwargs: 透传给划分器构造器。

    Returns:
        划分器实例。

    Raises:
        UnknownMethodError: 名字未注册。
    """
    from graphforge.core.errors import UnknownMethodError

    if isinstance(name, TaskType):
        name = "edge" if name is TaskType.LINK_PREDICTION else "stratified"
    if isinstance(name, SplitStrategy):
        name = name.value
    key = str(name)
    if key not in SPLITTER_REGISTRY:
        raise UnknownMethodError(
            f"未知划分器 {name!r}", hint=f"可选值：{list_splitters()}"
        )
    kwargs.setdefault("random_state", int(random_state))
    return SPLITTER_REGISTRY[key](**kwargs)
