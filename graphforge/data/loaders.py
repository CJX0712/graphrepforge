"""文件载入器：edgelist / CSV 边表 / 标签表 / 特征表，统一产出 :class:`GraphData`。

设计要点：
    * **不依赖 pandas 也能工作**：优先用 pandas，失败或缺失时回退标准库 ``csv``；
    * CSV 读取后显式 ``astype``，避免 pandas 3.x 的推断差异；
    * 所有载入结果都做"对称化 + 去自环 + CSR 化"，保证下游只看到规范图。

作者：晨星
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from graphforge.core.errors import (
    DatasetNotFoundError,
    EmptyGraphError,
    InvalidGraphFormatError,
    LabelMismatchError,
    UnsupportedFileFormatError,
)
from graphforge.core.logging import get_logger
from graphforge.core.types import GraphData
from graphforge.core.utils import remove_self_loops, symmetrize_adjacency, to_csr

__all__ = [
    "load_edgelist",
    "load_edge_csv",
    "load_label_csv",
    "load_feature_csv",
    "load_graph",
    "attach_labels",
    "attach_features",
    "EDGELIST_SUFFIXES",
    "CSV_SUFFIXES",
]

_LOGGER = get_logger(__name__)

#: 视为 edgelist 的文件后缀。
EDGELIST_SUFFIXES: Tuple[str, ...] = (".edgelist", ".edges", ".txt", ".tsv", ".dat")

#: 视为 CSV 的文件后缀。
CSV_SUFFIXES: Tuple[str, ...] = (".csv",)


def _resolve_path(path: Union[str, Path]) -> Path:
    """解析并校验文件路径。

    Args:
        path: 输入路径。

    Returns:
        存在的 :class:`Path`。

    Raises:
        DatasetNotFoundError: 文件不存在。
    """
    target = Path(path)
    if not target.exists():
        raise DatasetNotFoundError(
            f"文件不存在：{target}", hint="检查路径拼写或使用已注册的数据集名"
        )
    if not target.is_file():
        raise DatasetNotFoundError(f"不是文件：{target}", hint="请传入文件路径")
    return target


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """读取 CSV 为 ``(表头, 行字典列表)``。

    优先使用 pandas；pandas 不可用或读取失败时回退标准库 ``csv``。

    Args:
        path: CSV 文件路径。

    Returns:
        ``(header, rows)``。

    Raises:
        EmptyGraphError: 文件无表头或无数据行。
    """
    try:
        import pandas as pd  # 延迟导入，失败不影响系统

        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        header = [str(column) for column in frame.columns]
        rows = [
            {key: ("" if value is None else str(value)) for key, value in zip(header, record)}
            for record in frame.itertuples(index=False, name=None)
        ]
        _LOGGER.debug("pandas 读取 %s：%d 行", path.name, len(rows))
        return header, rows
    except ImportError:  # pragma: no cover - pandas 为核心依赖，此处仅兜底
        _LOGGER.debug("pandas 不可用，回退标准库 csv：%s", path.name)
    except Exception as exc:  # noqa: BLE001 - pandas 读取失败也要走兜底
        _LOGGER.warning("pandas 读取 %s 失败（%s），回退标准库 csv", path.name, exc)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = _csv.reader(handle)
        records = [list(record) for record in reader]
    if not records:
        raise EmptyGraphError(f"CSV 文件为空：{path}", hint="检查文件内容")
    header = [str(column).strip() for column in records[0]]
    rows = [
        {header[index]: value.strip() for index, value in enumerate(record)}
        for record in records[1:]
    ]
    if not rows:
        raise EmptyGraphError(f"CSV 无数据行：{path}", hint="检查文件内容")
    return header, rows


def _build_graph(
    sources: Sequence[str],
    targets: Sequence[str],
    weights: Optional[Sequence[float]],
    directed: bool,
    name: str,
) -> GraphData:
    """由边序列构造规范化的 :class:`GraphData`。

    Args:
        sources: 源节点 id。
        targets: 目标节点 id。
        weights: 边权；None 时全为 1.0。
        directed: 是否保留方向。
        name: 图名称。

    Returns:
        :class:`GraphData`（无向图已对称化、去自环）。

    Raises:
        EmptyGraphError: 没有任何有效边。
        InvalidGraphFormatError: 源/目标数量不一致。
    """
    if len(sources) != len(targets):
        raise InvalidGraphFormatError(
            f"源节点数 {len(sources)} 与目标节点数 {len(targets)} 不一致",
            hint="每行必须是 'u v' 或 'u v w'",
        )
    if not sources:
        raise EmptyGraphError(f"图 {name} 没有任何边", hint="检查文件内容")

    index: Dict[str, int] = {}
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []

    def _node_id(raw: str) -> int:
        if raw not in index:
            index[raw] = len(index)
        return index[raw]

    for position, (source, target) in enumerate(zip(sources, targets)):
        source_id = _node_id(str(source).strip())
        target_id = _node_id(str(target).strip())
        weight = 1.0 if weights is None else float(weights[position])
        rows.append(source_id)
        cols.append(target_id)
        values.append(weight)
        if not directed and source_id != target_id:
            rows.append(target_id)
            cols.append(source_id)
            values.append(weight)

    from scipy import sparse as sp  # 局部导入：本模块只在需要时触碰 scipy.sparse

    num_nodes = len(index)
    adjacency = to_csr(
        sp.coo_matrix((np.asarray(values, dtype=np.float64),
                       (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
                      shape=(num_nodes, num_nodes))
    )
    if not directed:
        adjacency = symmetrize_adjacency(adjacency, method="max")
    adjacency = remove_self_loops(adjacency)
    node_ids = [node for node, _ in sorted(index.items(), key=lambda item: item[1])]
    if adjacency.nnz == 0:
        raise EmptyGraphError(f"图 {name} 去自环后无边", hint="检查文件内容")
    return GraphData(
        adjacency=adjacency,
        node_ids=node_ids,
        node_labels=None,
        node_features=None,
        is_directed=bool(directed),
        name=name,
        metadata={"source": "file", "num_edges_raw": len(sources)},
    )


# ---------------------------------------------------------------- edgelist
def load_edgelist(
    path: Union[str, Path],
    delimiter: Optional[str] = None,
    weighted: bool = False,
    directed: bool = False,
    comment: str = "#",
    name: Optional[str] = None,
) -> GraphData:
    """载入 edgelist 文本（每行 ``u v`` 或 ``u v w``）。

    Args:
        path: 文件路径。
        delimiter: 分隔符；None 表示任意空白。
        weighted: 是否读取第三列作为权重。
        directed: 是否按有向图处理。
        comment: 注释行前缀。
        name: 图名称；None 时取文件名主干。

    Returns:
        :class:`GraphData`。

    Raises:
        DatasetNotFoundError: 文件不存在。
        EmptyGraphError: 文件无有效边。
        InvalidGraphFormatError: 行格式错误。
    """
    target = _resolve_path(path)
    sources: List[str] = []
    targets: List[str] = []
    weights: List[float] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or (comment and line.startswith(comment)):
                continue
            tokens = line.split(delimiter) if delimiter else line.split()
            tokens = [token.strip() for token in tokens if token.strip()]
            expected = 3 if weighted else 2
            if len(tokens) < expected:
                raise InvalidGraphFormatError(
                    f"{target.name}:{line_number} 字段数为 {len(tokens)}，需要 {expected}",
                    hint="每行应为 'u v'（或 'u v w'）",
                )
            sources.append(tokens[0])
            targets.append(tokens[1])
            weights.append(float(tokens[2]) if weighted else 1.0)
    graph_name = name or target.stem
    return _build_graph(sources, targets, weights if weighted else None, directed, graph_name)


# ---------------------------------------------------------------- CSV 边表
def load_edge_csv(
    path: Union[str, Path],
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = None,
    directed: bool = False,
    name: Optional[str] = None,
) -> GraphData:
    """载入 CSV 边表。

    Args:
        path: 文件路径。
        source_col: 源节点列名。
        target_col: 目标节点列名。
        weight_col: 权重列名；None 表示无权。
        directed: 是否按有向图处理。
        name: 图名称；None 时取文件名主干。

    Returns:
        :class:`GraphData`。

    Raises:
        InvalidGraphFormatError: 缺少必需列。
    """
    target = _resolve_path(path)
    header, rows = _read_csv_rows(target)
    for column in (source_col, target_col):
        if column not in header:
            raise InvalidGraphFormatError(
                f"CSV 缺少列 {column!r}（实际表头：{header}）",
                hint="用 source_col / target_col 指定列名",
            )
    if weight_col is not None and weight_col not in header:
        raise InvalidGraphFormatError(
            f"CSV 缺少权重列 {weight_col!r}", hint="确认 weight_col 与表头一致"
        )
    sources = [row[source_col] for row in rows]
    targets = [row[target_col] for row in rows]
    weights = [float(row[weight_col]) for row in rows] if weight_col else None
    graph_name = name or target.stem
    return _build_graph(sources, targets, weights, directed, graph_name)


# ---------------------------------------------------------------- 标签 / 特征
def load_label_csv(
    path: Union[str, Path],
    node_col: str = "node",
    label_col: str = "label",
) -> Tuple[List[str], np.ndarray]:
    """载入节点标签 CSV。

    Args:
        path: 文件路径。
        node_col: 节点 id 列名。
        label_col: 标签列名。

    Returns:
        ``(node_ids, labels)``；labels 为 ``(n,)`` int 数组（非数值标签按字典序映射）。

    Raises:
        InvalidGraphFormatError: 缺少必需列或无数据行。
    """
    target = _resolve_path(path)
    header, rows = _read_csv_rows(target)
    for column in (node_col, label_col):
        if column not in header:
            raise InvalidGraphFormatError(
                f"标签 CSV 缺少列 {column!r}（实际表头：{header}）",
                hint="用 node_col / label_col 指定列名",
            )
    node_ids = [str(row[node_col]) for row in rows]
    raw_labels = [str(row[label_col]) for row in rows]
    try:
        labels = np.asarray([int(float(value)) for value in raw_labels], dtype=np.int64)
    except ValueError:
        unique = sorted(set(raw_labels))
        mapping = {value: index for index, value in enumerate(unique)}
        labels = np.asarray([mapping[value] for value in raw_labels], dtype=np.int64)
    return node_ids, labels


def load_feature_csv(
    path: Union[str, Path],
    node_col: str = "node",
) -> Tuple[List[str], np.ndarray]:
    """载入节点特征 CSV（除 ``node_col`` 外全部列作为特征）。

    Args:
        path: 文件路径。
        node_col: 节点 id 列名。

    Returns:
        ``(node_ids, features)``；features 为 ``(n, d)`` float 数组。

    Raises:
        InvalidGraphFormatError: 缺少节点列或无特征列。
    """
    target = _resolve_path(path)
    header, rows = _read_csv_rows(target)
    if node_col not in header:
        raise InvalidGraphFormatError(
            f"特征 CSV 缺少列 {node_col!r}（实际表头：{header}）", hint="用 node_col 指定列名"
        )
    feature_cols = [column for column in header if column != node_col]
    if not feature_cols:
        raise InvalidGraphFormatError(
            f"特征 CSV 只有 {node_col!r} 一列", hint="至少提供 1 个特征列"
        )
    node_ids = [str(row[node_col]) for row in rows]
    features = np.asarray(
        [[float(row[column]) for column in feature_cols] for row in rows], dtype=np.float64
    )
    return node_ids, features


def attach_labels(graph: GraphData, node_ids: Sequence[str], labels: np.ndarray) -> GraphData:
    """把外部标签对齐到图（缺失节点填 ``-1``）。

    Args:
        graph: 目标图。
        node_ids: 标签文件中的节点 id。
        labels: 与 ``node_ids`` 对齐的标签数组。

    Returns:
        新的 :class:`GraphData`。

    Raises:
        LabelMismatchError: 标签长度与节点 id 数量不一致。
    """
    array = np.asarray(labels)
    if len(node_ids) != array.shape[0]:
        raise LabelMismatchError(
            f"标签数 {array.shape[0]} 与节点数 {len(node_ids)} 不一致", hint="检查标签文件"
        )
    position = {str(node): index for index, node in enumerate(node_ids)}
    aligned = np.full(graph.num_nodes, -1, dtype=np.int64)
    for index, node_id in enumerate(graph.node_ids):
        if node_id in position:
            aligned[index] = int(array[position[node_id]])
    return GraphData(
        adjacency=graph.adjacency,
        node_ids=list(graph.node_ids),
        node_labels=aligned,
        node_features=graph.node_features,
        is_directed=graph.is_directed,
        name=graph.name,
        metadata={**dict(graph.metadata), "label_source": "file"},
    )


def attach_features(
    graph: GraphData, node_ids: Sequence[str], features: np.ndarray
) -> GraphData:
    """把外部特征对齐到图（缺失节点填 0）。

    Args:
        graph: 目标图。
        node_ids: 特征文件中的节点 id。
        features: ``(n, d)`` 特征矩阵。

    Returns:
        新的 :class:`GraphData`。

    Raises:
        LabelMismatchError: 特征行数与节点 id 数量不一致。
    """
    array = np.asarray(features, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if len(node_ids) != array.shape[0]:
        raise LabelMismatchError(
            f"特征行数 {array.shape[0]} 与节点数 {len(node_ids)} 不一致", hint="检查特征文件"
        )
    position = {str(node): index for index, node in enumerate(node_ids)}
    aligned = np.zeros((graph.num_nodes, array.shape[1]), dtype=np.float64)
    for index, node_id in enumerate(graph.node_ids):
        if node_id in position:
            aligned[index] = array[position[node_id]]
    return GraphData(
        adjacency=graph.adjacency,
        node_ids=list(graph.node_ids),
        node_labels=graph.node_labels,
        node_features=aligned,
        is_directed=graph.is_directed,
        name=graph.name,
        metadata={**dict(graph.metadata), "feature_source": "file"},
    )


# ---------------------------------------------------------------- 分发
def load_graph(path: Union[str, Path], **kwargs: Any) -> GraphData:
    """按后缀自动选择载入器。

    Args:
        path: 文件路径。
        **kwargs: 透传给具体载入器。

    Returns:
        :class:`GraphData`。

    Raises:
        UnsupportedFileFormatError: 后缀不受支持。
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix in CSV_SUFFIXES:
        return load_edge_csv(target, **kwargs)
    if suffix in EDGELIST_SUFFIXES:
        if suffix == ".tsv":
            kwargs.setdefault("delimiter", "\t")
        return load_edgelist(target, **kwargs)
    raise UnsupportedFileFormatError(
        f"不支持的文件后缀 {suffix!r}：{target.name}",
        hint=f"edgelist 支持 {list(EDGELIST_SUFFIXES)}，CSV 支持 {list(CSV_SUFFIXES)}",
    )
