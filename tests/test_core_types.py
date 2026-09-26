"""core.types 数据模型单测。

作者：晨星
"""

from __future__ import annotations

import numpy as np
import pytest
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
)
from graphforge.core.types import (
    BenchmarkResult,
    Embedding,
    EvalResult,
    GraphData,
    HPOSpec,
    ParamSpec,
    SearchSpace,
    Split,
    TaskSpec,
    TaskType,
    TrialResult,
)


def _graph() -> GraphData:
    """构造一个 4 节点链图（0-1-2-3）。"""
    adjacency = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )
    return GraphData(
        adjacency=adjacency,
        node_ids=["0", "1", "2", "3"],
        node_labels=np.array([0, 0, 1, 1], dtype=np.int64),
        name="chain",
    )


# ---------------------------------------------------------------- GraphData
def test_graph_basic_properties() -> None:
    """num_nodes / num_edges / has_labels / degrees 正确。"""
    graph = _graph()
    assert graph.num_nodes == 4
    assert graph.num_edges == 3  # 无向边：3 条
    assert graph.has_labels is True
    assert graph.degrees.tolist() == [1.0, 2.0, 2.0, 1.0]


def test_graph_validate_rejects_non_square() -> None:
    """非方阵邻接 → E203。"""
    graph = GraphData(
        adjacency=sp.csr_matrix(np.ones((2, 3))), node_ids=["a", "b"], name="bad"
    )
    with pytest.raises(InvalidAdjacencyError) as info:
        graph.validate()
    assert info.value.code == "E203"


def test_graph_validate_rejects_empty() -> None:
    """0 节点 → E103。"""
    graph = GraphData(adjacency=sp.csr_matrix((0, 0)), node_ids=[], name="empty")
    with pytest.raises(EmptyGraphError) as info:
        graph.validate()
    assert info.value.code == "E103"


def test_graph_validate_rejects_id_count_mismatch() -> None:
    """node_ids 数量与矩阵不一致 → E102。"""
    graph = GraphData(adjacency=sp.csr_matrix(np.zeros((3, 3))), node_ids=["a"], name="bad")
    with pytest.raises(InvalidGraphFormatError) as info:
        graph.validate()
    assert info.value.code == "E102"


def test_graph_validate_rejects_label_mismatch() -> None:
    """标签长度不匹配 → E104。"""
    graph = _graph()
    broken = GraphData(
        adjacency=graph.adjacency,
        node_ids=graph.node_ids,
        node_labels=np.array([0, 1], dtype=np.int64),
        name="bad",
    )
    with pytest.raises(LabelMismatchError) as info:
        broken.validate()
    assert info.value.code == "E104"


def test_graph_networkx_roundtrip() -> None:
    """to_networkx → from_networkx 往返后结构一致。"""
    graph = _graph()
    nx_graph = graph.to_networkx()
    assert nx_graph.number_of_nodes() == 4
    assert nx_graph.number_of_edges() == 3
    restored = GraphData.from_networkx(nx_graph, label_attr="label", name="chain")
    assert restored.num_nodes == graph.num_nodes
    assert restored.num_edges == graph.num_edges
    assert sorted(restored.node_ids) == sorted(graph.node_ids)
    assert sorted(np.asarray(restored.node_labels).tolist()) == [0, 0, 1, 1]


def test_graph_subgraph() -> None:
    """subgraph 同步裁剪邻接、id 与标签。"""
    graph = _graph()
    sub = graph.subgraph([0, 1, 2])
    assert sub.num_nodes == 3
    assert sub.node_ids == ["0", "1", "2"]
    assert np.asarray(sub.node_labels).tolist() == [0, 0, 1]
    assert sub.num_edges == 2


def test_graph_subgraph_rejects_out_of_range() -> None:
    """越界索引 → E102。"""
    graph = _graph()
    with pytest.raises(InvalidGraphFormatError):
        graph.subgraph([0, 99])


# ---------------------------------------------------------------- Embedding
def test_embedding_basic_and_align() -> None:
    """dim / num_nodes / align 重排 / validate。"""
    matrix = np.arange(6, dtype=np.float64).reshape(3, 2)
    embedding = Embedding(matrix=matrix, node_ids=["a", "b", "c"], method="dummy")
    assert embedding.dim == 2
    assert embedding.num_nodes == 3
    aligned = embedding.align(["c", "a"])
    assert aligned.node_ids == ["c", "a"]
    assert aligned.matrix.tolist() == matrix[[2, 0]].tolist()
    embedding.validate()


def test_embedding_validate_rejects_nan() -> None:
    """含 NaN → E302。"""
    matrix = np.array([[0.0, 1.0], [np.nan, 1.0]])
    embedding = Embedding(matrix=matrix, node_ids=["a", "b"], method="dummy")
    with pytest.raises(EmbeddingFitError) as info:
        embedding.validate()
    assert info.value.code == "E302"


def test_embedding_align_rejects_missing_node() -> None:
    """align 目标含未知节点 → E102。"""
    embedding = Embedding(matrix=np.zeros((2, 2)), node_ids=["a", "b"], method="dummy")
    with pytest.raises(InvalidGraphFormatError):
        embedding.align(["a", "zzz"])


# ---------------------------------------------------------------- TaskSpec
def test_task_spec_default_primary() -> None:
    """未显式指定 primary 时按任务取默认值（越大越好口径一致）。"""
    assert TaskSpec(task=TaskType.NODE_CLASSIFICATION).resolve_primary() == "macro_f1"
    assert TaskSpec(task=TaskType.LINK_PREDICTION).resolve_primary() == "roc_auc"
    assert TaskSpec(task=TaskType.EMBEDDING_BENCHMARK).resolve_primary() == "macro_f1"


def test_task_spec_validate_rejects_bad_ratio() -> None:
    """比例和不为 1 → E201。"""
    spec = TaskSpec(task=TaskType.NODE_CLASSIFICATION, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)
    with pytest.raises(InvalidSplitRatioError) as info:
        spec.validate()
    assert info.value.code == "E201"


def test_task_spec_validate_rejects_bad_edge_op() -> None:
    """未知 edge_op → E403。"""
    spec = TaskSpec(task=TaskType.LINK_PREDICTION, edge_op="unknown")
    with pytest.raises(InvalidSearchSpaceError):
        spec.validate()


def test_task_spec_copy_with() -> None:
    """copy_with 返回新对象且不改原对象。"""
    spec = TaskSpec(task=TaskType.NODE_CLASSIFICATION)
    other = spec.copy_with(random_state=7)
    assert spec.random_state == 42
    assert other.random_state == 7


# ---------------------------------------------------------------- Split
def test_split_sizes() -> None:
    """Split 的规模属性正确。"""
    split = Split(
        train_idx=np.arange(6), val_idx=np.arange(6, 8), test_idx=np.arange(8, 10)
    )
    assert split.sizes == (6, 2, 2)
    split.validate()


def test_split_validate_rejects_empty_train() -> None:
    """空训练集 → E202。"""
    split = Split(train_idx=np.zeros(0, dtype=int), val_idx=np.arange(2), test_idx=np.arange(2, 4))
    with pytest.raises(InsufficientSamplesError):
        split.validate()


# ---------------------------------------------------------------- 结果类型
def test_eval_result_as_row() -> None:
    """as_row 展平且指标带 ``metric_`` 前缀。"""
    result = EvalResult(
        task=TaskType.NODE_CLASSIFICATION,
        dataset="sbm",
        method="node2vec",
        backend="internal",
        metrics={"accuracy": 0.9, "macro_f1": 0.8},
        primary_metric="macro_f1",
        primary_value=0.8,
        n_samples=10,
        elapsed_sec=0.123456,
    )
    row = result.as_row()
    assert row["task"] == "node_classification"
    assert row["metric_accuracy"] == 0.9
    assert row["elapsed_sec"] == 0.1235


def test_benchmark_result_best_and_io(tmp_path) -> None:
    """best 取最大；to_json 可落盘；to_table 各行等宽。"""
    def _row(name: str, value: float) -> EvalResult:
        return EvalResult(
            task=TaskType.NODE_CLASSIFICATION,
            dataset="sbm",
            method=name,
            backend="internal",
            metrics={"macro_f1": value},
            primary_metric="macro_f1",
            primary_value=value,
            n_samples=5,
            elapsed_sec=0.01,
        )

    benchmark = BenchmarkResult(
        name="demo", rows=[_row("a", 0.5), _row("b", 0.9)], skipped=[{"method": "c", "reason": "unavailable"}]
    )
    assert benchmark.best().method == "b"
    assert benchmark.best("macro_f1").primary_value == 0.9

    target = tmp_path / "benchmark.json"
    benchmark.to_json(target)
    assert target.exists() and target.read_text(encoding="utf-8").strip().startswith("{")

    table = benchmark.to_table()
    lines = table.splitlines()
    assert len(lines) == 5  # 表头 + 分隔线 + 2 数据行 + 1 行 skipped
    assert len({len(line) for line in lines}) == 1  # 定宽：所有行长度一致


def test_benchmark_result_best_rejects_missing_metric() -> None:
    """不存在的指标 → E406。"""
    benchmark = BenchmarkResult(
        name="demo",
        rows=[
            EvalResult(
                task=TaskType.NODE_CLASSIFICATION,
                dataset="sbm",
                method="a",
                backend="internal",
                metrics={"accuracy": 0.5},
                primary_metric="macro_f1",
                primary_value=0.5,
                n_samples=1,
                elapsed_sec=0.0,
            )
        ],
    )
    with pytest.raises(MissingMetricError):
        benchmark.best("macro_f1")


def test_benchmark_result_empty_rejects() -> None:
    """空结果取 best → E202。"""
    with pytest.raises(InsufficientSamplesError):
        BenchmarkResult(name="empty").best()


# ---------------------------------------------------------------- HPO 类型
def test_search_space_sample_and_grid() -> None:
    """sample 落在区间内；grid 组合数受预算约束且确定。"""
    space = SearchSpace(
        params=(
            ParamSpec("p", "log_float", 0.25, 4.0, default=1.0),
            ParamSpec("k", "int", 1, 5, default=2),
            ParamSpec("op", "categorical", choices=("a", "b"), default="a"),
        )
    )
    rng = np.random.default_rng(0)
    sample = space.sample(rng)
    assert 0.25 <= sample["p"] <= 4.0
    assert 1 <= sample["k"] <= 5
    assert sample["op"] in ("a", "b")

    grid = space.grid(8)
    assert len(grid) <= 8
    assert grid == space.grid(8)  # 确定性


def test_search_space_rejects_bad_spec() -> None:
    """非法 kind / 缺 choices → E403。"""
    with pytest.raises(InvalidSearchSpaceError):
        ParamSpec("x", "weird", 0, 1)
    with pytest.raises(InvalidSearchSpaceError):
        ParamSpec("x", "categorical")
    with pytest.raises(InvalidSearchSpaceError):
        ParamSpec("x", "log_float", 0.0, 1.0)


def test_hpo_spec_validate() -> None:
    """HPOSpec 合法/非法分支。"""
    HPOSpec().validate()
    with pytest.raises(InvalidSearchSpaceError):
        HPOSpec(backend="nope").validate()
    with pytest.raises(InvalidSearchSpaceError):
        HPOSpec(n_trials=0).validate()
    with pytest.raises(InvalidSearchSpaceError):
        HPOSpec(direction="minimize").validate()


def test_trial_result_to_dict() -> None:
    """TrialResult 可序列化。"""
    trial = TrialResult(best_params={"p": 1.0}, best_value=0.8, n_trials=3, backend="grid")
    payload = trial.to_dict()
    assert payload["n_trials"] == 3 and payload["best_params"]["p"] == 1.0
