"""合成图生成器（全部吃 ``random_state``，结果确定可复现）。

提供的生成器：
    * :func:`generate_sbm` —— 随机块模型（**默认合成图**，节点带 ``block`` 属性直接当标签）
    * :func:`generate_karate` —— Zachary 空手道俱乐部（34 节点）
    * :func:`generate_grid` —— 2D 网格图
    * :func:`generate_lfr` —— LFR 基准图（**危险**：会挂起，带重试 + 时间预算 + 回退 SBM）

> ⚠️ LFR 实测会抛 ``ExceededMaxIterations`` 或长时间不返回（>120s）。
> 本模块对 LFR 做了三重保护：规模上限、重试次数、**守护线程 + 时间预算**，
> 超时/失败一律回退到 SBM 并在 ``metadata`` 中留下 ``lfr_fallback=True``。
> 测试与 demo **默认不走 LFR**。

作者：晨星
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from graphforge.core.errors import EmptyGraphError
from graphforge.core.logging import get_logger
from graphforge.core.types import GraphData
from graphforge.core.utils import (
    derive_seed,
    remove_self_loops,
    symmetrize_adjacency,
    to_csr,
)

__all__ = [
    "generate_sbm",
    "generate_karate",
    "generate_grid",
    "generate_lfr",
    "generate",
    "SYNTHETIC_GENERATORS",
    "list_synthetic",
]

_LOGGER = get_logger(__name__)

#: LFR 允许尝试的最大节点数（超过则直接回退，实测 >200 极易挂起）。
LFR_MAX_NODES: int = 200


def _import_networkx() -> Any:
    """延迟导入 networkx（data 层不强制 core 依赖它）。

    Returns:
        ``networkx`` 模块。
    """
    import networkx as nx  # noqa: WPS433 - 延迟导入是刻意的

    return nx


# ---------------------------------------------------------------- SBM
def generate_sbm(
    n_blocks: int = 4,
    block_size: int = 100,
    p_in: float = 0.15,
    p_out: float = 0.02,
    random_state: int = 42,
    name: str = "sbm",
) -> GraphData:
    """生成随机块模型（Stochastic Block Model）图。

    Args:
        n_blocks: 社区数。
        block_size: 每个社区的节点数。
        p_in: 社区内连边概率。
        p_out: 社区间连边概率。
        random_state: 随机种子。
        name: 图名称。

    Returns:
        :class:`GraphData`，标签即社区编号（``block`` 属性）。

    Raises:
        EmptyGraphError: 生成的图无边（概率过低）。
    """
    nx = _import_networkx()
    sizes: List[int] = [int(block_size)] * int(n_blocks)
    probs = np.full((n_blocks, n_blocks), float(p_out), dtype=np.float64)
    np.fill_diagonal(probs, float(p_in))
    seed = derive_seed(random_state, "data", "sbm")
    graph = nx.stochastic_block_model(sizes, probs.tolist(), seed=seed, directed=False)
    data = GraphData.from_networkx(graph, label_attr="block", name=name)
    adjacency = remove_self_loops(symmetrize_adjacency(data.adjacency, method="max"))
    data = GraphData(
        adjacency=adjacency,
        node_ids=list(data.node_ids),
        node_labels=data.node_labels,
        node_features=data.node_features,
        is_directed=False,
        name=name,
        metadata={
            "generator": "sbm",
            "n_blocks": int(n_blocks),
            "block_size": int(block_size),
            "p_in": float(p_in),
            "p_out": float(p_out),
            "random_state": int(random_state),
        },
    )
    if data.num_edges == 0:
        raise EmptyGraphError(
            f"SBM 生成结果无边（p_in={p_in}, p_out={p_out}）", hint="提高连边概率"
        )
    _LOGGER.debug("SBM 生成完成：%d 节点 / %d 边", data.num_nodes, data.num_edges)
    return data


# ---------------------------------------------------------------- Karate
def generate_karate(name: str = "karate", random_state: int = 42) -> GraphData:
    """生成 Zachary 空手道俱乐部图（34 节点 / 78 边，确定性图）。

    Args:
        name: 图名称。
        random_state: 保留参数（该图为确定性图，种子不影响结构，仅写入 metadata）。

    Returns:
        :class:`GraphData`，标签为 ``club`` 属性映射出的 0/1。
    """
    nx = _import_networkx()
    graph = nx.karate_club_graph()
    data = GraphData.from_networkx(graph, label_attr="club", name=name)
    adjacency = remove_self_loops(symmetrize_adjacency(data.adjacency, method="max"))
    return GraphData(
        adjacency=adjacency,
        node_ids=list(data.node_ids),
        node_labels=data.node_labels,
        node_features=None,
        is_directed=False,
        name=name,
        metadata={
            "generator": "karate",
            "random_state": int(random_state),
            "deterministic": True,
        },
    )


# ---------------------------------------------------------------- Grid
def generate_grid(
    m: int = 10,
    n: int = 10,
    label_by: str = "row",
    random_state: int = 42,
    name: str = "grid",
) -> GraphData:
    """生成 2D 网格图。

    Args:
        m: 行数。
        n: 列数。
        label_by: 标签来源；``"row"`` 用行号（共 ``m`` 类），``"none"`` 不生成标签。
        random_state: 保留参数（网格图为确定性图）。
        name: 图名称。

    Returns:
        :class:`GraphData`，节点 id 为 ``"0".."m*n-1"``。

    Raises:
        ValueError: ``label_by`` 取值非法。
    """
    nx = _import_networkx()
    graph = nx.grid_2d_graph(int(m), int(n))
    nodes = list(graph.nodes())
    adjacency = to_csr(
        nx.to_scipy_sparse_array(graph, nodelist=nodes, format="csr", dtype=np.float64)
    )
    adjacency = remove_self_loops(symmetrize_adjacency(adjacency, method="max"))

    labels: Optional[np.ndarray] = None
    if label_by == "row":
        labels = np.asarray([int(node[0]) for node in nodes], dtype=np.int64)
    elif label_by == "none":
        labels = None
    else:
        raise ValueError(f"未知的 label_by：{label_by!r}（可选：row / none）")

    return GraphData(
        adjacency=adjacency,
        node_ids=[str(index) for index in range(len(nodes))],
        node_labels=labels,
        node_features=None,
        is_directed=False,
        name=name,
        metadata={
            "generator": "grid",
            "m": int(m),
            "n": int(n),
            "label_by": label_by,
            "random_state": int(random_state),
            "deterministic": True,
        },
    )


# ---------------------------------------------------------------- LFR
def _lfr_worker(result: Dict[str, Any], **kwargs: Any) -> None:
    """在守护线程内执行 LFR 生成，把图或异常写入 ``result``。"""
    try:
        nx = _import_networkx()
        result["graph"] = nx.LFR_benchmark_graph(**kwargs)
    except BaseException as exc:  # noqa: BLE001 - 必须捕获 nx 的任意失败
        result["error"] = exc


def generate_lfr(
    n: int = 100,
    tau1: float = 3.0,
    tau2: float = 1.5,
    mu: float = 0.1,
    average_degree: float = 5.0,
    min_community: int = 10,
    random_state: int = 42,
    max_retries: int = 2,
    time_budget_sec: float = 5.0,
    fallback: str = "sbm",
    name: str = "lfr",
) -> GraphData:
    """生成 LFR 基准图（**带超时回退**，见模块 docstring 的危险提示）。

    Args:
        n: 节点数（> :data:`LFR_MAX_NODES` 时直接回退）。
        tau1: 度分布指数。
        tau2: 社区规模分布指数。
        mu: 混合参数。
        average_degree: 平均度。
        min_community: 最小社区规模。
        random_state: 随机种子。
        max_retries: 最大重试次数。
        time_budget_sec: 每次尝试的时间预算（秒）。
        fallback: 回退生成器名（``"sbm"``）。
        name: 图名称。

    Returns:
        :class:`GraphData`；回退时 ``metadata["lfr_fallback"] = True``。
    """
    reason: Optional[str] = None
    if int(n) > LFR_MAX_NODES:
        reason = f"n={n} 超过 LFR 安全上限 {LFR_MAX_NODES}"

    graph: Any = None
    if reason is None:
        for attempt in range(max(1, int(max_retries))):
            seed = derive_seed(random_state, "data", "lfr", str(attempt))
            kwargs: Dict[str, Any] = {
                "n": int(n),
                "tau1": float(tau1),
                "tau2": float(tau2),
                "mu": float(mu),
                "average_degree": float(average_degree),
                "min_community": int(min_community),
                "seed": int(seed),
            }
            result: Dict[str, Any] = {}
            worker = threading.Thread(
                target=_lfr_worker, args=(result,), kwargs=kwargs, daemon=True
            )
            worker.start()
            worker.join(timeout=max(0.1, float(time_budget_sec)))
            if worker.is_alive():
                reason = (
                    f"LFR 超时（>{time_budget_sec}s，attempt={attempt}）：已放弃该守护线程"
                )
                break
            if "graph" in result:
                graph = result["graph"]
                break
            reason = f"LFR 生成失败（attempt={attempt}）：{type(result.get('error')).__name__}"

    if graph is None:
        _LOGGER.warning("LFR 不可用（%s），回退到 %s", reason, fallback)
        data = SYNTHETIC_GENERATORS[fallback](random_state=random_state)
        return GraphData(
            adjacency=data.adjacency,
            node_ids=list(data.node_ids),
            node_labels=data.node_labels,
            node_features=data.node_features,
            is_directed=data.is_directed,
            name=name,
            metadata={
                **dict(data.metadata),
                "generator": "lfr_fallback",
                "lfr_fallback": True,
                "lfr_reason": reason or "unknown",
                "lfr_requested_n": int(n),
            },
        )

    # LFR 的社区属性是 set，需手工展开为标签
    communities = nx_communities(graph)
    node_ids = [str(node) for node in graph.nodes()]
    labels = np.full(len(node_ids), -1, dtype=np.int64)
    for community_index, members in enumerate(communities):
        position = {str(node): index for index, node in enumerate(graph.nodes())}
        for member in members:
            labels[position[str(member)]] = community_index
    adjacency = to_csr(
        _import_networkx().to_scipy_sparse_array(
            graph, nodelist=list(graph.nodes()), format="csr", dtype=np.float64
        )
    )
    adjacency = remove_self_loops(symmetrize_adjacency(adjacency, method="max"))
    return GraphData(
        adjacency=adjacency,
        node_ids=node_ids,
        node_labels=labels,
        node_features=None,
        is_directed=False,
        name=name,
        metadata={
            "generator": "lfr",
            "n": int(n),
            "mu": float(mu),
            "num_communities": int(len(communities)),
            "random_state": int(random_state),
            "lfr_fallback": False,
        },
    )


def nx_communities(graph: Any) -> List[Sequence[Any]]:
    """从 LFR 图的 ``community`` 节点属性抽出社区列表。

    Args:
        graph: 带 ``community`` 节点属性的 networkx 图。

    Returns:
        社区列表，每个社区为节点序列。
    """
    buckets: Dict[int, List[Any]] = {}
    for node, attrs in graph.nodes(data=True):
        for community_index in attrs.get("community", ()):
            buckets.setdefault(int(community_index), []).append(node)
    return [buckets[key] for key in sorted(buckets)]


# ---------------------------------------------------------------- 注册表
SYNTHETIC_GENERATORS: Dict[str, Callable[..., GraphData]] = {
    "sbm": generate_sbm,
    "karate": generate_karate,
    "grid": generate_grid,
    "lfr": generate_lfr,
}


def list_synthetic() -> List[str]:
    """返回全部合成图生成器名（升序）。"""
    return sorted(SYNTHETIC_GENERATORS)


def generate(name: str, **kwargs: Any) -> GraphData:
    """按名字调用合成图生成器。

    Args:
        name: 生成器名（``sbm`` / ``karate`` / ``grid`` / ``lfr``）。
        **kwargs: 透传给生成器的参数。

    Returns:
        :class:`GraphData`。

    Raises:
        UnknownMethodError: 生成器名未注册。
    """
    from graphforge.core.errors import UnknownMethodError

    if name not in SYNTHETIC_GENERATORS:
        raise UnknownMethodError(
            f"未知合成图生成器 {name!r}", hint=f"可选值：{list_synthetic()}"
        )
    return SYNTHETIC_GENERATORS[name](**kwargs)
