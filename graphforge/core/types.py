"""核心数据模型（L0）。

本模块只允许 import stdlib / typing / numpy / scipy；``networkx`` 一律在方法内部
**延迟导入**，以保证 core 层不因可选依赖失败而不可用。

主要类型：
    * 枚举：``TaskType`` / ``BackendStatus`` / ``SplitStrategy``
    * 图：``GraphData``
    * 嵌入：``Embedding``
    * 任务与划分：``TaskSpec`` / ``Split``
    * 结果：``EvalResult`` / ``BenchmarkResult``
    * HPO：``ParamSpec`` / ``SearchSpace`` / ``HPOSpec`` / ``TrialResult``

作者：晨星
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse as sp

from graphforge.core.errors import (
    EmbeddingFitError,
    EmptyGraphError,
    InsufficientSamplesError,
    InvalidAdjacencyError,
    InvalidGraphFormatError,
    InvalidSearchSpaceError,
    InvalidSplitRatioError,
    LabelMismatchError,
    MissingMetricError,
    SerializationError,
)

__all__ = [
    "TaskType",
    "BackendStatus",
    "SplitStrategy",
    "GraphData",
    "Embedding",
    "TaskSpec",
    "Split",
    "EvalResult",
    "BenchmarkResult",
    "ParamSpec",
    "SearchSpace",
    "HPOSpec",
    "TrialResult",
    "DEFAULT_PRIMARY_METRIC",
    "PARAM_KINDS",
    "HPO_BACKENDS",
    "EDGE_OPS",
]


# ==================================================================== 枚举
class TaskType(str, Enum):
    """任务类型枚举（str 子类，便于直接序列化）。"""

    NODE_CLASSIFICATION = "node_classification"
    LINK_PREDICTION = "link_prediction"
    EMBEDDING_BENCHMARK = "embedding_benchmark"

    def __str__(self) -> str:  # pragma: no cover - 序列化辅助
        return self.value


class BackendStatus(str, Enum):
    """后端状态枚举。"""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"

    def __str__(self) -> str:  # pragma: no cover - 序列化辅助
        return self.value


class SplitStrategy(str, Enum):
    """划分策略枚举。"""

    STRATIFIED = "stratified"
    RANDOM = "random"
    EDGE = "edge"

    def __str__(self) -> str:  # pragma: no cover - 序列化辅助
        return self.value


#: 各任务的默认 primary 指标（全部"越大越好"）。
DEFAULT_PRIMARY_METRIC: Dict[TaskType, str] = {
    TaskType.NODE_CLASSIFICATION: "macro_f1",
    TaskType.LINK_PREDICTION: "roc_auc",
    TaskType.EMBEDDING_BENCHMARK: "macro_f1",
}

#: 链路预测的边特征算子。
EDGE_OPS: Tuple[str, ...] = ("hadamard", "average", "l2", "concat")

#: 搜索空间允许的参数类型。
PARAM_KINDS: Tuple[str, ...] = ("float", "int", "log_float", "categorical")

#: HPO 后端枚举值。
HPO_BACKENDS: Tuple[str, ...] = ("auto", "optuna", "random", "grid", "none")


# ==================================================================== 图
@dataclass(frozen=True)
class GraphData:
    """图的统一表示。

    Attributes:
        adjacency: ``(n, n)`` 的 CSR 稀疏邻接矩阵；无向图应已对称化、无自环。
        node_ids: 长度 ``n`` 的稳定顺序节点标识（字符串）。
        node_labels: ``(n,)`` 的 int 标签；``-1`` 表示该节点无标签；可为 None。
        node_features: ``(n, d)`` 的 float 特征；可为 None。
        is_directed: 是否为有向图。
        name: 数据集/图名称。
        metadata: 附加元信息（生成参数、来源路径等）。
    """

    adjacency: sp.csr_matrix
    node_ids: List[str]
    node_labels: Optional[np.ndarray] = None
    node_features: Optional[np.ndarray] = None
    is_directed: bool = False
    name: str = "graph"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- 属性
    @property
    def num_nodes(self) -> int:
        """节点数。"""
        return int(self.adjacency.shape[0])

    @property
    def num_edges(self) -> int:
        """边数（无向图按"无向边"计数，即 nnz // 2）。"""
        nnz = int(self.adjacency.nnz)
        if self.is_directed:
            return nnz
        return nnz // 2

    @property
    def has_labels(self) -> bool:
        """是否存在有效标签（至少一个非 ``-1`` 的标签）。"""
        if self.node_labels is None:
            return False
        labels = np.asarray(self.node_labels)
        if labels.size == 0:
            return False
        return bool(np.any(labels >= 0))

    @property
    def degrees(self) -> np.ndarray:
        """返回 ``(n,)`` 的出度（无向图即度）。"""
        return np.asarray(self.adjacency.sum(axis=1)).ravel()

    # ---------------------------------------------------------------- 转换
    def to_networkx(self) -> Any:
        """转换为 ``networkx`` 图对象（延迟 import networkx）。

        Returns:
            ``nx.Graph`` 或 ``nx.DiGraph``，节点名为 :attr:`node_ids`，
            边属性带 ``weight``；标签写入节点属性 ``label``，特征写入 ``feature``。

        Raises:
            InvalidGraphFormatError: 图未通过 :meth:`validate` 校验。
        """
        import networkx as nx  # 延迟导入：core 层不强制依赖 networkx

        self.validate()
        graph = nx.DiGraph() if self.is_directed else nx.Graph()
        graph.add_nodes_from(self.node_ids)
        coo = self.adjacency.tocoo()
        max_weight = float(coo.data.max()) if coo.data.size else 1.0
        for row, col, value in zip(coo.row, coo.col, coo.data):
            if not self.is_directed and row > col:
                continue  # 无向图只保留上三角，避免重复加边
            weight = float(value)
            if max_weight > 0 and weight > 0:
                graph.add_edge(
                    self.node_ids[int(row)],
                    self.node_ids[int(col)],
                    weight=weight,
                )
        if self.node_labels is not None:
            labels = np.asarray(self.node_labels)
            for index, node_id in enumerate(self.node_ids):
                graph.nodes[node_id]["label"] = int(labels[index])
        if self.node_features is not None:
            features = np.asarray(self.node_features)
            for index, node_id in enumerate(self.node_ids):
                graph.nodes[node_id]["feature"] = features[index].tolist()
        graph.graph["name"] = self.name
        graph.graph.update({key: value for key, value in self.metadata.items()})
        return graph

    @classmethod
    def from_networkx(
        cls,
        graph: Any,
        label_attr: Optional[str] = None,
        feature_attr: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "GraphData":
        """从 ``networkx`` 图构造 :class:`GraphData`（延迟 import networkx）。

        Args:
            graph: networkx 图对象。
            label_attr: 作为标签的节点属性名；为 None 则不抽取标签。
            feature_attr: 作为特征的节点属性名；为 None 则不抽取特征。
            name: 图名称；None 时取 ``graph.graph.get("name")`` 或 ``"graph"``。

        Returns:
            新的 :class:`GraphData`。
        """
        import networkx as nx  # 延迟导入

        nodes: List[Any] = list(graph.nodes())
        if not nodes:
            raise EmptyGraphError("networkx 图为空", hint="请检查输入图或改用合成图生成器")
        node_ids: List[str] = [str(node) for node in nodes]
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, format="csr", dtype=np.float64)
        adjacency = sp.csr_matrix(adjacency)

        labels: Optional[np.ndarray] = None
        if label_attr is not None:
            raw = [graph.nodes[node].get(label_attr) for node in nodes]
            if any(value is None for value in raw):
                raise LabelMismatchError(
                    f"节点缺少属性 {label_attr!r}", hint="确认 label_attr 与图属性名一致"
                )
            unique = sorted({str(value) for value in raw})
            mapping = {value: index for index, value in enumerate(unique)}
            labels = np.asarray([mapping[str(value)] for value in raw], dtype=np.int64)

        features: Optional[np.ndarray] = None
        if feature_attr is not None:
            raw_features = [graph.nodes[node].get(feature_attr) for node in nodes]
            if any(value is None for value in raw_features):
                raise LabelMismatchError(
                    f"节点缺少属性 {feature_attr!r}", hint="确认 feature_attr 与图属性名一致"
                )
            features = np.asarray(raw_features, dtype=np.float64)
            if features.ndim == 1:
                features = features.reshape(-1, 1)

        graph_name = name or str(graph.graph.get("name", "graph"))
        return cls(
            adjacency=adjacency,
            node_ids=node_ids,
            node_labels=labels,
            node_features=features,
            is_directed=bool(graph.is_directed()),
            name=graph_name,
            metadata={"source": "networkx"},
        )

    # ---------------------------------------------------------------- 校验
    def validate(self) -> None:
        """校验图结构，违规抛 E1xx / E2xx。

        Raises:
            InvalidAdjacencyError: 邻接矩阵非稀疏 / 非方阵。
            EmptyGraphError: 节点数为 0。
            InvalidGraphFormatError: 节点 id 数量或重复、特征形状错误。
            LabelMismatchError: 标签长度与节点数不一致。
        """
        adjacency = self.adjacency
        if not sp.issparse(adjacency):
            raise InvalidAdjacencyError("邻接矩阵必须是 scipy 稀疏矩阵", hint="使用 sp.csr_matrix")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise InvalidAdjacencyError(
                f"邻接矩阵必须是方阵，实际 shape={adjacency.shape}", hint="检查构图逻辑"
            )
        num_nodes = int(adjacency.shape[0])
        if num_nodes == 0:
            raise EmptyGraphError("图节点数为 0", hint="检查数据集或生成参数")
        if len(self.node_ids) != num_nodes:
            raise InvalidGraphFormatError(
                f"node_ids 长度 {len(self.node_ids)} 与节点数 {num_nodes} 不一致",
                hint="节点 id 必须与邻接矩阵行序一一对应",
            )
        if len(set(self.node_ids)) != num_nodes:
            raise InvalidGraphFormatError("node_ids 存在重复值", hint="节点 id 必须唯一")
        if self.node_labels is not None:
            labels = np.asarray(self.node_labels)
            if labels.ndim != 1 or labels.shape[0] != num_nodes:
                raise LabelMismatchError(
                    f"node_labels 形状 {labels.shape} 与节点数 {num_nodes} 不匹配",
                    hint="标签应为 (n,) 的一维数组",
                )
        if self.node_features is not None:
            features = np.asarray(self.node_features)
            if features.ndim != 2 or features.shape[0] != num_nodes:
                raise InvalidGraphFormatError(
                    f"node_features 形状 {features.shape} 与节点数 {num_nodes} 不匹配",
                    hint="特征应为 (n, d) 的二维数组",
                )

    def subgraph(self, node_index: Sequence[int]) -> "GraphData":
        """按节点索引取子图（保持顺序，同步裁剪标签与特征）。

        Args:
            node_index: 节点下标序列（允许乱序，不允许越界）。

        Returns:
            新的 :class:`GraphData`。

        Raises:
            InvalidGraphFormatError: 索引越界或为空。
        """
        index = np.asarray(node_index, dtype=np.int64).ravel()
        if index.size == 0:
            raise InvalidGraphFormatError("子图节点索引为空", hint="至少选择 1 个节点")
        if index.min() < 0 or index.max() >= self.num_nodes:
            raise InvalidGraphFormatError(
                f"节点索引越界：[{index.min()}, {index.max()}] 超出 [0, {self.num_nodes - 1}]",
                hint="索引必须落在节点范围内",
            )
        sub_adjacency = sp.csr_matrix(self.adjacency[index][:, index])
        labels = None
        if self.node_labels is not None:
            labels = np.asarray(self.node_labels)[index].copy()
        features = None
        if self.node_features is not None:
            features = np.asarray(self.node_features)[index].copy()
        return GraphData(
            adjacency=sub_adjacency,
            node_ids=[self.node_ids[int(i)] for i in index],
            node_labels=labels,
            node_features=features,
            is_directed=self.is_directed,
            name=f"{self.name}:subgraph",
            metadata=dict(self.metadata),
        )


# ==================================================================== 嵌入
@dataclass(frozen=True)
class Embedding:
    """图嵌入结果。

    Attributes:
        matrix: ``(n, dim)`` 的嵌入矩阵，行序与 :attr:`node_ids` 一致。
        node_ids: 节点标识。
        method: 方法名（``"node2vec"`` / ``"deepwalk"`` / ...）。
        backend: ``"internal"`` 或 ``"karateclub"``。
        params: 生成该嵌入时使用的超参快照。
    """

    matrix: np.ndarray
    node_ids: List[str]
    method: str
    backend: str = "internal"
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        """嵌入维度。"""
        return int(self.matrix.shape[1])

    @property
    def num_nodes(self) -> int:
        """嵌入覆盖的节点数。"""
        return int(self.matrix.shape[0])

    def validate(self) -> None:
        """校验嵌入矩阵。

        Raises:
            InvalidGraphFormatError: 矩阵非二维、行数与 ``node_ids`` 不一致。
            EmbeddingFitError: 矩阵含 NaN / Inf。
            LabelMismatchError: ``node_ids`` 长度与矩阵行数不一致。
        """
        matrix = np.asarray(self.matrix)
        if matrix.ndim != 2:
            raise InvalidGraphFormatError(
                f"嵌入矩阵必须是二维，实际 ndim={matrix.ndim}", hint="检查 _fit 返回值"
            )
        if len(self.node_ids) != matrix.shape[0]:
            raise LabelMismatchError(
                f"node_ids 长度 {len(self.node_ids)} 与矩阵行数 {matrix.shape[0]} 不一致",
                hint="嵌入行序必须与 node_ids 对齐",
            )
        if matrix.shape[1] == 0:
            raise InvalidGraphFormatError("嵌入维度为 0", hint="dim 必须 >= 1")
        if not np.all(np.isfinite(matrix)):
            raise EmbeddingFitError("嵌入矩阵含 NaN 或 Inf", hint="检查学习率与归一化步骤")

    def align(self, node_ids: Sequence[str]) -> "Embedding":
        """按目标节点顺序重排嵌入行。

        Args:
            node_ids: 目标节点 id 序列（必须是当前 :attr:`node_ids` 的子集）。

        Returns:
            重排后的新 :class:`Embedding`。

        Raises:
            InvalidGraphFormatError: 目标 id 存在缺失。
        """
        target = list(node_ids)
        position = {node_id: index for index, node_id in enumerate(self.node_ids)}
        missing = [node_id for node_id in target if node_id not in position]
        if missing:
            raise InvalidGraphFormatError(
                f"嵌入缺少 {len(missing)} 个节点：{missing[:5]}", hint="先对图与嵌入做节点对齐"
            )
        rows = [position[node_id] for node_id in target]
        return Embedding(
            matrix=np.asarray(self.matrix)[rows].copy(),
            node_ids=target,
            method=self.method,
            backend=self.backend,
            params=dict(self.params),
        )


# ==================================================================== 任务
@dataclass(frozen=True)
class TaskSpec:
    """任务规格。

    Attributes:
        task: 任务类型。
        train_ratio / val_ratio / test_ratio: 划分比例，三者之和应为 1.0。
        n_splits: 交叉验证折数。
        random_state: 随机种子。
        scoring: 需要计算的指标名序列。
        primary_metric: 主指标；None 时按任务取默认值。
        negative_ratio: 链路预测负样本倍率。
        edge_op: 链路预测边特征算子。
        estimator: 下游估计器名。
        estimator_params: 下游估计器超参。
        use_cv: 是否使用交叉验证。
    """

    task: TaskType
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    scoring: Tuple[str, ...] = ("accuracy", "macro_f1")
    primary_metric: Optional[str] = None
    negative_ratio: float = 1.0
    edge_op: str = "hadamard"
    estimator: str = "logistic_regression"
    estimator_params: Dict[str, Any] = field(default_factory=dict)
    use_cv: bool = False

    # ---------------------------------------------------------------- 派生
    @property
    def ratios(self) -> Tuple[float, float, float]:
        """返回 ``(train, val, test)`` 比例三元组。"""
        return (self.train_ratio, self.val_ratio, self.test_ratio)

    def resolve_primary(self) -> str:
        """解析主指标。

        Returns:
            显式指定时返回 :attr:`primary_metric`；否则返回任务默认主指标
            （节点分类 ``macro_f1``、链路预测 ``roc_auc``、嵌入基准 ``macro_f1``）。
        """
        if self.primary_metric:
            return self.primary_metric
        return DEFAULT_PRIMARY_METRIC[self.task]

    def resolve_scoring(self) -> Tuple[str, ...]:
        """解析指标列表；为空时退化为 ``(primary,)``。"""
        if self.scoring:
            return tuple(self.scoring)
        return (self.resolve_primary(),)

    def copy_with(self, **changes: Any) -> "TaskSpec":
        """返回带修改的副本（基于 ``dataclasses.replace``）。"""
        return replace(self, **changes)

    # ---------------------------------------------------------------- 校验
    def validate(self) -> None:
        """校验任务规格。

        Raises:
            InvalidSplitRatioError: 比例非正、和偏离 1.0、折数 < 2。
            InvalidSearchSpaceError: ``edge_op`` 不在 :data:`EDGE_OPS` 内。
        """
        train, val, test = self.ratios
        for name, value in (("train_ratio", train), ("val_ratio", val), ("test_ratio", test)):
            if value < 0:
                raise InvalidSplitRatioError(
                    f"{name}={value} 不能为负", hint="比例必须在 [0, 1] 内"
                )
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            raise InvalidSplitRatioError(
                f"划分比例之和为 {total:.6f}，应为 1.0", hint="调整 train/val/test 比例"
            )
        if train <= 0:
            raise InvalidSplitRatioError("train_ratio 必须大于 0", hint="训练集不能为空")
        if self.n_splits < 2:
            raise InvalidSplitRatioError(
                f"n_splits={self.n_splits} 不合法", hint="交叉验证折数必须 >= 2"
            )
        if self.edge_op not in EDGE_OPS:
            raise InvalidSearchSpaceError(
                f"edge_op={self.edge_op!r} 不支持", hint=f"可选值：{list(EDGE_OPS)}"
            )
        if self.negative_ratio <= 0:
            raise InvalidSplitRatioError(
                f"negative_ratio={self.negative_ratio} 必须为正", hint="负样本倍率需 > 0"
            )


@dataclass(frozen=True)
class Split:
    """划分结果。

    统一约定：``train_idx / val_idx / test_idx`` 是**样本下标**——
    节点分类任务下标指向节点，链路预测任务下标指向 :attr:`extra` 中的边数组。

    Attributes:
        train_idx: 训练集样本下标。
        val_idx: 验证集样本下标。
        test_idx: 测试集样本下标。
        y: 与"全部样本"对齐的标签（节点分类为类别，链路预测为 0/1）。
        extra: 附加数据（边数组、训练期邻接矩阵等）。
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    y: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        """训练集样本数。"""
        return int(np.asarray(self.train_idx).size)

    @property
    def n_val(self) -> int:
        """验证集样本数。"""
        return int(np.asarray(self.val_idx).size)

    @property
    def n_test(self) -> int:
        """测试集样本数。"""
        return int(np.asarray(self.test_idx).size)

    @property
    def sizes(self) -> Tuple[int, int, int]:
        """返回 ``(n_train, n_val, n_test)``。"""
        return (self.n_train, self.n_val, self.n_test)

    def validate(self, min_total: int = 1) -> None:
        """校验划分结果。

        Args:
            min_total: 样本总量下限。

        Raises:
            InsufficientSamplesError: 样本总量不足或某子集为空。
        """
        total = self.n_train + self.n_val + self.n_test
        if total < min_total:
            raise InsufficientSamplesError(
                f"划分后样本总量 {total} < {min_total}", hint="降低比例要求或增大图规模"
            )
        if self.n_train == 0:
            raise InsufficientSamplesError("训练集为空", hint="提高 train_ratio")


# ==================================================================== 结果
@dataclass
class EvalResult:
    """单次评估结果。

    Attributes:
        task: 任务类型。
        dataset: 数据集名。
        method: 嵌入/方法名；无嵌入时为 ``"none"``。
        backend: 后端名（``internal`` / ``karateclub``）。
        metrics: 指标名 → 数值（全部"越大越好"）。
        primary_metric: 主指标名。
        primary_value: 主指标数值。
        n_samples: 参与评估的样本数。
        elapsed_sec: 评估耗时（秒）。
        params: 该次运行的参数快照。
    """

    task: TaskType
    dataset: str
    method: str
    backend: str
    metrics: Dict[str, float]
    primary_metric: str
    primary_value: float
    n_samples: int
    elapsed_sec: float
    params: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        """展平为一行（供表格 / CSV / DataFrame 使用）。

        Returns:
            字段字典；指标以 ``metric_<name>`` 为键展开，避免与主字段冲突。
        """
        row: Dict[str, Any] = {
            "task": self.task.value if isinstance(self.task, TaskType) else str(self.task),
            "dataset": self.dataset,
            "method": self.method,
            "backend": self.backend,
            "primary_metric": self.primary_metric,
            "primary_value": float(self.primary_value),
            "n_samples": int(self.n_samples),
            "elapsed_sec": round(float(self.elapsed_sec), 4),
        }
        for name, value in self.metrics.items():
            row[f"metric_{name}"] = float(value)
        return row

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（保留嵌套结构）。"""
        return {
            "task": self.task.value if isinstance(self.task, TaskType) else str(self.task),
            "dataset": self.dataset,
            "method": self.method,
            "backend": self.backend,
            "metrics": {key: float(value) for key, value in self.metrics.items()},
            "primary_metric": self.primary_metric,
            "primary_value": float(self.primary_value),
            "n_samples": int(self.n_samples),
            "elapsed_sec": round(float(self.elapsed_sec), 4),
            "params": dict(self.params),
        }


@dataclass
class BenchmarkResult:
    """基准评测汇总结果。

    Attributes:
        name: 基准名称。
        created_at: 创建时间（ISO-8601，UTC）。
        config: 运行配置快照。
        rows: 各次评估的 :class:`EvalResult`。
        skipped: 被跳过的方法，形如 ``[{"method": ..., "reason": ...}]``。
    """

    name: str = "benchmark"
    created_at: str = field(default_factory=lambda: _utc_now_iso())
    config: Dict[str, Any] = field(default_factory=dict)
    rows: List[EvalResult] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)

    # ---------------------------------------------------------------- 查询
    def best(self, metric: Optional[str] = None) -> EvalResult:
        """返回指定指标上最优的一行（**越大越好**）。

        Args:
            metric: 指标名；None 时使用各行的主指标（多行主指标必须一致）。

        Returns:
            最优的 :class:`EvalResult`。

        Raises:
            InsufficientSamplesError: 结果为空。
            MissingMetricError: 没有任何一行包含该指标。
        """
        if not self.rows:
            raise InsufficientSamplesError("基准结果为空，无法取最优", hint="先运行 benchmark")
        name = metric or self._consistent_primary()
        candidates = [row for row in self.rows if name in row.metrics]
        if not candidates:
            raise MissingMetricError(
                f"所有结果均缺少指标 {name!r}",
                hint=f"可用指标：{sorted(self.rows[0].metrics)}",
            )
        return max(candidates, key=lambda row: float(row.metrics[name]))

    def _consistent_primary(self) -> str:
        """返回多行一致的主指标名；不一致时抛 E406。"""
        names = {row.primary_metric for row in self.rows}
        if len(names) != 1:
            raise MissingMetricError(
                f"结果主指标不一致：{sorted(names)}", hint="显式传入 metric 参数"
            )
        return self.rows[0].primary_metric

    # ---------------------------------------------------------------- 输出
    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "config": dict(self.config),
            "rows": [row.to_dict() for row in self.rows],
            "skipped": [dict(item) for item in self.skipped],
        }

    def to_json(self, path: Union[str, Path]) -> None:
        """落盘 JSON。

        Args:
            path: 目标文件路径（父目录不存在时自动创建）。

        Raises:
            SerializationError: 写入失败。
        """
        target = Path(path)
        try:
            if target.parent and not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
        except OSError as exc:
            raise SerializationError(
                f"写入 {target} 失败：{exc}", hint="检查目录权限与磁盘空间"
            ) from exc

    def to_table(self, metric: Optional[str] = None) -> str:
        """渲染为固定宽度表格（列宽 12，避免中英文错位）。

        Args:
            metric: 展示的指标列；None 时使用各行主指标。

        Returns:
            多行文本表格。
        """
        headers = ("dataset", "method", "backend", "task", "metric", "value", "elapsed")
        lines = ["".join(f"{header:<12}" for header in headers), "-" * 12 * len(headers)]

        def _fmt(cell: str) -> str:
            return f"{cell[:12]:<12}"

        for row in self.rows:
            name = metric or row.primary_metric
            value = row.metrics.get(name, row.primary_value)
            cells = (
                str(row.dataset),
                str(row.method),
                str(row.backend),
                row.task.value if isinstance(row.task, TaskType) else str(row.task),
                str(name),
                f"{float(value):.4f}",
                f"{float(row.elapsed_sec):.3f}",
            )
            lines.append("".join(_fmt(cell) for cell in cells))
        for item in self.skipped:
            cells = ("-", str(item.get("method", "unknown")), "skipped", "-", "-",
                     str(item.get("reason", ""))[:11], "-")
            lines.append("".join(_fmt(cell) for cell in cells))
        return "\n".join(lines)


def _utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO-8601 字符串（秒精度）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ==================================================================== HPO
@dataclass(frozen=True)
class ParamSpec:
    """单个超参的搜索定义。

    Attributes:
        name: 参数名。
        kind: ``float`` / ``int`` / ``log_float`` / ``categorical``。
        low: 下界（数值型）。
        high: 上界（数值型，int 取闭区间）。
        choices: 离散取值（categorical）。
        default: 默认值。
    """

    name: str
    kind: str = "float"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Sequence[Any]] = None
    default: Optional[Any] = None

    def __post_init__(self) -> None:
        """校验参数定义。

        Raises:
            InvalidSearchSpaceError: kind 非法或边界缺失/倒置。
        """
        if self.kind not in PARAM_KINDS:
            raise InvalidSearchSpaceError(
                f"参数 {self.name} 的 kind={self.kind!r} 非法", hint=f"可选值：{list(PARAM_KINDS)}"
            )
        if self.kind == "categorical":
            if self.choices is None or len(self.choices) == 0:
                raise InvalidSearchSpaceError(
                    f"categorical 参数 {self.name} 缺少 choices", hint="至少给出 1 个候选值"
                )
        else:
            if self.low is None or self.high is None:
                raise InvalidSearchSpaceError(
                    f"数值参数 {self.name} 缺少 low/high", hint="float/int 必须给出上下界"
                )
            if self.low > self.high:
                raise InvalidSearchSpaceError(
                    f"参数 {self.name} 的 low={self.low} 大于 high={self.high}",
                    hint="下界必须不大于上界",
                )
            if self.kind == "log_float" and (self.low is not None and self.low <= 0):
                raise InvalidSearchSpaceError(
                    f"log_float 参数 {self.name} 的下界必须为正", hint="log 域要求 low > 0"
                )

    def values(self, n: int) -> List[Any]:
        """返回该参数在网格搜索中的 ``n`` 个取值。

        Args:
            n: 期望取值个数（>= 1）。

        Returns:
            取值列表。categorical 直接返回候选；数值型按等距/对数等距采样。
        """
        count = max(1, int(n))
        if self.kind == "categorical":
            return list(self.choices or [])
        low = float(self.low if self.low is not None else 0.0)
        high = float(self.high if self.high is not None else 1.0)
        if self.kind == "int":
            grid = np.unique(np.linspace(low, high, count).round().astype(np.int64))
            return [int(value) for value in grid]
        if self.kind == "log_float":
            grid = np.geomspace(max(low, 1e-12), max(high, 1e-12), count)
            return [float(value) for value in grid]
        return [float(value) for value in np.linspace(low, high, count)]

    def sample(self, rng: "np.random.Generator") -> Any:
        """用给定随机数发生器采样一个取值。

        Args:
            rng: ``numpy`` 随机数发生器。

        Returns:
            采样值。
        """
        if self.kind == "categorical":
            choices = list(self.choices or [])
            index = int(rng.integers(0, len(choices)))
            return choices[index]
        low = float(self.low if self.low is not None else 0.0)
        high = float(self.high if self.high is not None else 1.0)
        if self.kind == "int":
            return int(rng.integers(int(round(low)), int(round(high)) + 1))
        if self.kind == "log_float":
            log_low = math.log(max(low, 1e-12))
            log_high = math.log(max(high, 1e-12))
            return float(math.exp(float(rng.uniform(log_low, log_high))))
        return float(rng.uniform(low, high))


@dataclass(frozen=True)
class SearchSpace:
    """超参搜索空间。

    Attributes:
        params: :class:`ParamSpec` 元组。
    """

    params: Tuple[ParamSpec, ...] = ()

    @property
    def names(self) -> Tuple[str, ...]:
        """返回全部参数名。"""
        return tuple(spec.name for spec in self.params)

    def __len__(self) -> int:
        """参数个数。"""
        return len(self.params)

    def __iter__(self) -> Iterator[ParamSpec]:
        """迭代参数定义。"""
        return iter(self.params)

    def sample(self, rng: "np.random.Generator") -> Dict[str, Any]:
        """随机采样一组参数。

        Args:
            rng: ``numpy`` 随机数发生器。

        Returns:
            ``{参数名: 采样值}``。
        """
        return {spec.name: spec.sample(rng) for spec in self.params}

    def grid(self, n: int = 10) -> List[Dict[str, Any]]:
        """生成至多 ``n`` 组网格参数。

        先按每参数 ``ceil(n ** (1/k))`` 个取值做笛卡尔积；若组合数超过 ``n``，
        则按等距下标抽样，保证结果确定且覆盖两端。

        Args:
            n: 期望组合数上限（>= 1）。

        Returns:
            参数字典列表。
        """
        if not self.params:
            return [{}]
        budget = max(1, int(n))
        per_param = max(1, int(math.ceil(budget ** (1.0 / len(self.params)))))
        axes: List[List[Any]] = [spec.values(per_param) for spec in self.params]
        combos: List[Dict[str, Any]] = [{}]
        for spec, axis in zip(self.params, axes):
            combos = [
                {**combo, spec.name: value} for combo in combos for value in axis
            ]
        if len(combos) <= budget:
            return combos
        idx = np.unique(np.linspace(0, len(combos) - 1, budget).round().astype(np.int64))
        return [combos[int(i)] for i in idx]

    def validate(self) -> None:
        """校验搜索空间。

        Raises:
            InvalidSearchSpaceError: 参数名重复或单个参数定义非法。
        """
        names = self.names
        if len(set(names)) != len(names):
            raise InvalidSearchSpaceError("搜索空间存在重复参数名", hint="参数名必须唯一")
        for spec in self.params:
            spec.values(1)  # 触发 ParamSpec 自校验逻辑


@dataclass(frozen=True)
class HPOSpec:
    """超参搜索配置。

    Attributes:
        backend: ``auto`` / ``optuna`` / ``random`` / ``grid`` / ``none``。
        n_trials: 试验次数。
        direction: 优化方向；本项目统一 ``maximize``（指标越大越好）。
        random_state: 随机种子。
        timeout_sec: 时间预算（秒）；None 表示不限。
    """

    backend: str = "auto"
    n_trials: int = 20
    direction: str = "maximize"
    random_state: int = 42
    timeout_sec: Optional[float] = None

    def validate(self) -> None:
        """校验 HPO 配置。

        Raises:
            InvalidSearchSpaceError: backend 非法、次数 < 1、方向非 maximize。
        """
        if self.backend not in HPO_BACKENDS:
            raise InvalidSearchSpaceError(
                f"backend={self.backend!r} 非法", hint=f"可选值：{list(HPO_BACKENDS)}"
            )
        if self.n_trials < 1:
            raise InvalidSearchSpaceError(
                f"n_trials={self.n_trials} 非法", hint="试验次数必须 >= 1"
            )
        if self.direction != "maximize":
            raise InvalidSearchSpaceError(
                f"direction={self.direction!r} 不支持", hint="本项目统一使用 maximize"
            )


@dataclass
class TrialResult:
    """超参搜索结果。

    Attributes:
        best_params: 最优参数组合。
        best_value: 最优目标值（越大越好）。
        n_trials: 实际完成的试验次数。
        backend: 实际使用的后端名。
        history: 每次试验的记录，形如 ``[{"params": {...}, "value": float}]``。
    """

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_value: float = float("-inf")
    n_trials: int = 0
    backend: str = "grid"
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "best_params": dict(self.best_params),
            "best_value": float(self.best_value),
            "n_trials": int(self.n_trials),
            "backend": self.backend,
            "history": [dict(item) for item in self.history],
        }
