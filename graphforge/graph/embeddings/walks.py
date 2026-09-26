"""随机游走（DeepWalk 的均匀游走 / node2vec 的 p·q 偏置二阶游走）。

性能要求：采样必须 O(1)。本模块用 **alias table（Vose 1991）** 实现，
先 O(k) 建表、后 O(1) 采样，避免每步做 ``np.random.choice``（O(k)）。

node2vec 偏置语义（Grover & Leskovec, KDD 2016 §3）：
从 ``t`` 走到 ``v`` 后，下一步走到 ``x`` 的**未归一化**概率为
``w(v, x) * alpha``，其中
    * ``alpha = 1/p``（``x == t``，走回上一个节点）
    * ``alpha = 1``  （``x`` 与 ``t`` 相邻，BFS 式）
    * ``alpha = 1/q``（``x`` 与 ``t`` 不相邻，DFS 式）

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp

from graphforge.core.logging import get_logger
from graphforge.core.utils import to_csr

__all__ = [
    "alias_setup",
    "alias_draw",
    "build_alias_tables",
    "WalkSampler",
    "uniform_random_walk",
    "biased_random_walk",
    "generate_walks",
]

_LOGGER = get_logger(__name__)


def alias_setup(probs: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """为给定概率分布建立 alias table（O(k) 建表）。

    Args:
        probs: 非负的未归一化权重。

    Returns:
        ``(J, q)``：``J`` 为 int 索引数组，``q`` 为 float 概率数组。
        分布全零/含 NaN 时退化为均匀分布（避免除零）。
    """
    values = np.asarray(probs, dtype=np.float64).ravel()
    size = int(values.size)
    if size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        values = np.ones(size, dtype=np.float64)
        total = float(size)
    prob_q = values * (size / total)
    jump = np.zeros(size, dtype=np.int64)
    smaller: List[int] = []
    larger: List[int] = []
    for index in range(size):
        if prob_q[index] < 1.0:
            smaller.append(index)
        else:
            larger.append(index)
    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        jump[small] = large
        prob_q[large] = prob_q[large] + prob_q[small] - 1.0
        if prob_q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return jump, np.minimum(prob_q, 1.0)


def alias_draw(jump: np.ndarray, prob_q: np.ndarray, rng: "np.random.Generator") -> int:
    """从 alias table 中 O(1) 采样一个下标。

    Args:
        jump: :func:`alias_setup` 返回的 ``J``。
        prob_q: :func:`alias_setup` 返回的 ``q``。
        rng: ``numpy`` 随机数发生器。

    Returns:
        采样得到的下标。

    Raises:
        ValueError: 表为空。
    """
    size = int(jump.size)
    if size == 0:
        raise ValueError("alias table 为空，无法采样")
    index = int(rng.integers(0, size))
    if float(rng.random()) < float(prob_q[index]):
        return index
    return int(jump[index])


def build_alias_tables(
    adjacency: Any, p: float = 1.0, q: float = 1.0
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """为 node2vec 预计算**每条有向边** ``(t, v)`` 的 alias table。

    Args:
        adjacency: 邻接矩阵（CSR）。
        p: 返回参数（return parameter）。
        q: 出入参数（in-out parameter）。

    Returns:
        ``{(t, v): (J, q)}`` 字典。
    """
    sampler = WalkSampler(adjacency, p=p, q=q)
    tables: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for previous in range(sampler.num_nodes):
        for current in sampler.neighbors[previous]:
            tables[(previous, int(current))] = sampler.biased_table(previous, int(current))
    return tables


class WalkSampler:
    """基于 alias table 的随机游走采样器（均匀 / 偏置二合一）。

    Attributes:
        num_nodes: 节点数。
        neighbors: 每个节点的邻居下标数组（CSR 切片缓存）。
        p: node2vec 返回参数。
        q: node2vec 出入参数。
    """

    def __init__(self, adjacency: Any, p: float = 1.0, q: float = 1.0) -> None:
        """构造采样器（**不预先展开所有偏置表**，按边惰性缓存）。

        Args:
            adjacency: 邻接矩阵（任意可转 CSR 的对象）。
            p: 返回参数，必须 > 0。
            q: 出入参数，必须 > 0。

        Raises:
            ValueError: ``p`` / ``q`` 非正。
        """
        if p <= 0 or q <= 0:
            raise ValueError(f"p / q 必须为正，实际 p={p}, q={q}")
        matrix = to_csr(adjacency)
        self.num_nodes: int = int(matrix.shape[0])
        self.indptr: np.ndarray = matrix.indptr
        self.indices: np.ndarray = matrix.indices
        self.data: np.ndarray = matrix.data.astype(np.float64)
        self.neighbors: List[np.ndarray] = [
            self.indices[self.indptr[node] : self.indptr[node + 1]].astype(np.int64)
            for node in range(self.num_nodes)
        ]
        self.weights: List[np.ndarray] = [
            self.data[self.indptr[node] : self.indptr[node + 1]] for node in range(self.num_nodes)
        ]
        self.neighbor_sets: List[frozenset] = [
            frozenset(int(value) for value in bucket) for bucket in self.neighbors
        ]
        self.p: float = float(p)
        self.q: float = float(q)
        self._uniform_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._biased_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    # ---------------------------------------------------------------- 表
    def uniform_table(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
        """返回节点 ``node`` 的**均匀** alias table（忽略边权，纯 DeepWalk 语义）。"""
        cached = self._uniform_cache.get(node)
        if cached is None:
            size = int(self.neighbors[node].size)
            cached = alias_setup(np.ones(size, dtype=np.float64))
            self._uniform_cache[node] = cached
        return cached

    def weighted_table(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
        """返回节点 ``node`` 的**按边权** alias table。"""
        weights = self.weights[node]
        if weights.size == 0:
            return alias_setup(np.zeros(0))
        return alias_setup(weights)

    def biased_table(self, previous: int, current: int) -> Tuple[np.ndarray, np.ndarray]:
        """返回给定上一步 ``previous``、当前 ``current`` 的 node2vec 偏置 alias table。

        Args:
            previous: 上一步节点 ``t``。
            current: 当前节点 ``v``。

        Returns:
            ``(J, q)``。
        """
        key = (int(previous), int(current))
        cached = self._biased_cache.get(key)
        if cached is not None:
            return cached
        bucket = self.neighbors[current]
        weights = self.weights[current]
        previous_set = self.neighbor_sets[previous]
        size = int(bucket.size)
        scaled = np.empty(size, dtype=np.float64)
        for position in range(size):
            neighbour = int(bucket[position])
            if neighbour == previous:
                alpha = 1.0 / self.p
            elif neighbour in previous_set:
                alpha = 1.0
            else:
                alpha = 1.0 / self.q
            scaled[position] = float(weights[position]) * alpha
        cached = alias_setup(scaled)
        self._biased_cache[key] = cached
        return cached

    # ---------------------------------------------------------------- 游走
    def uniform_walk(self, start: int, length: int, rng: "np.random.Generator") -> List[int]:
        """均匀随机游走（DeepWalk）。

        Args:
            start: 起点节点。
            length: 游走长度（节点数，含起点）。
            rng: 随机数发生器。

        Returns:
            节点下标列表；遇到孤立节点时提前终止。
        """
        walk: List[int] = [int(start)]
        current = int(start)
        for _ in range(max(0, int(length) - 1)):
            if self.neighbors[current].size == 0:
                break
            jump, prob_q = self.uniform_table(current)
            offset = alias_draw(jump, prob_q, rng)
            current = int(self.neighbors[current][offset])
            walk.append(current)
        return walk

    def biased_walk(self, start: int, length: int, rng: "np.random.Generator") -> List[int]:
        """node2vec 二阶偏置随机游走。

        Args:
            start: 起点节点。
            length: 游走长度（节点数，含起点）。
            rng: 随机数发生器。

        Returns:
            节点下标列表。
        """
        walk: List[int] = [int(start)]
        if self.neighbors[start].size == 0 or length <= 1:
            return walk
        previous = -1
        current = int(start)
        for _ in range(int(length) - 1):
            if previous < 0:
                jump, prob_q = self.weighted_table(current)
            else:
                jump, prob_q = self.biased_table(previous, current)
            offset = alias_draw(jump, prob_q, rng)
            nxt = int(self.neighbors[current][offset])
            if nxt < 0:
                break
            previous, current = current, nxt
            walk.append(current)
            if self.neighbors[current].size == 0:
                break
        return walk


def uniform_random_walk(
    adjacency: Any,
    start: int,
    length: int,
    rng: "np.random.Generator",
    sampler: Optional[WalkSampler] = None,
) -> List[int]:
    """均匀随机游走（函数式入口，DeepWalk 用）。

    Args:
        adjacency: 邻接矩阵。
        start: 起点节点。
        length: 游走长度。
        rng: 随机数发生器。
        sampler: 复用已有采样器（可选）。

    Returns:
        节点下标列表。
    """
    engine = sampler if sampler is not None else WalkSampler(adjacency)
    return engine.uniform_walk(int(start), int(length), rng)


def biased_random_walk(
    adjacency: Any,
    start: int,
    length: int,
    p: float,
    q: float,
    rng: "np.random.Generator",
    sampler: Optional[WalkSampler] = None,
) -> List[int]:
    """node2vec 偏置随机游走（函数式入口）。

    Args:
        adjacency: 邻接矩阵。
        start: 起点节点。
        length: 游走长度。
        p: 返回参数。
        q: 出入参数。
        rng: 随机数发生器。
        sampler: 复用已有采样器（可选；为 None 时按 p/q 新建）。

    Returns:
        节点下标列表。
    """
    engine = sampler if sampler is not None else WalkSampler(adjacency, p=p, q=q)
    return engine.biased_walk(int(start), int(length), rng)


def generate_walks(
    adjacency: Any,
    num_walks: int = 10,
    walk_length: int = 80,
    p: Optional[float] = None,
    q: Optional[float] = None,
    rng: Optional["np.random.Generator"] = None,
    nodes: Optional[Sequence[int]] = None,
    random_state: int = 42,
) -> List[List[int]]:
    """为全部节点生成随机游走语料。

    ``p`` / ``q`` 均为 None（或都等于 1）时使用均匀游走（DeepWalk）；
    否则使用二阶偏置游走（node2vec）。

    Args:
        adjacency: 邻接矩阵。
        num_walks: 每个节点的游走条数。
        walk_length: 每条游走的长度。
        p: 返回参数；None 表示均匀游走。
        q: 出入参数；None 表示均匀游走。
        rng: 随机数发生器；None 时由 ``random_state`` 派生（tag = ``walk``）。
        nodes: 起点节点序列；None 表示全部节点。
        random_state: 未显式传 ``rng`` 时的种子。

    Returns:
        游走列表（每条为节点下标列表）。
    """
    from graphforge.core.utils import make_rng

    generator = rng if rng is not None else make_rng(random_state, "walk")
    biased = p is not None and q is not None and (float(p) != 1.0 or float(q) != 1.0)
    sampler = WalkSampler(adjacency, p=float(p or 1.0), q=float(q or 1.0))
    starts = list(range(sampler.num_nodes)) if nodes is None else [int(node) for node in nodes]
    walks: List[List[int]] = []
    for _ in range(max(1, int(num_walks))):
        for start in starts:
            if biased:
                walks.append(sampler.biased_walk(start, int(walk_length), generator))
            else:
                walks.append(sampler.uniform_walk(start, int(walk_length), generator))
    _LOGGER.debug(
        "生成游走 %d 条（length=%d, biased=%s）", len(walks), walk_length, biased
    )
    return walks
