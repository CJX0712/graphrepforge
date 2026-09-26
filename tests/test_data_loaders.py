"""data.loaders 单测：edgelist / CSV 读写与挂载。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graphforge.core.errors import (
    DatasetNotFoundError,
    EmptyGraphError,
    LabelMismatchError,
    UnsupportedFileFormatError,
)


def test_load_edgelist(tmp_path: Path) -> None:
    """edgelist 文本文件正确建图。"""
    from graphforge.data.loaders import load_edgelist

    path = tmp_path / "edges.txt"
    path.write_text("a b 1.0\nb c 1.0\nc a 1.0\n", encoding="utf-8")
    graph = load_edgelist(path)
    assert graph.num_nodes == 3
    assert graph.num_edges == 3
    assert not graph.has_labels


def test_load_edge_csv_and_labels(tmp_path: Path) -> None:
    """CSV 建图 + 标签 CSV 挂载（默认列 source/target）。"""
    from graphforge.data.loaders import attach_labels, load_edge_csv, load_label_csv

    edge_path = tmp_path / "edges.csv"
    edge_path.write_text("source,target\n1,2\n2,3\n3,1\n", encoding="utf-8")
    graph = load_edge_csv(edge_path)
    assert graph.num_nodes == 3

    label_path = tmp_path / "labels.csv"
    label_path.write_text("node,label\n1,0\n2,1\n3,0\n", encoding="utf-8")
    ids, labels = load_label_csv(label_path)
    labeled = attach_labels(graph, ids, labels)
    assert labeled.has_labels
    assert int(np.asarray(labeled.node_labels).sum()) == 1


def test_attach_labels_length_mismatch(tmp_path: Path) -> None:
    """标签数组与节点 id 数量不一致 → E104。"""
    from graphforge.data.loaders import attach_labels, load_edgelist

    path = tmp_path / "edges.txt"
    path.write_text("a b\n", encoding="utf-8")
    graph = load_edgelist(path)
    with pytest.raises(LabelMismatchError):
        attach_labels(graph, ["a"], np.asarray([0, 1]))


def test_attach_labels_missing_node_filled(tmp_path: Path) -> None:
    """标签文件缺失的节点填 -1。"""
    from graphforge.data.loaders import attach_labels, load_edgelist

    path = tmp_path / "edges.txt"
    path.write_text("a b\nb c\n", encoding="utf-8")
    graph = load_edgelist(path)
    labeled = attach_labels(graph, ["a"], np.asarray([1]))
    labels = np.asarray(labeled.node_labels)
    assert labels[0] == 1
    assert (labels[1:] == -1).all()


def test_empty_file_raises(tmp_path: Path) -> None:
    """空 CSV → E103。"""
    from graphforge.data.loaders import load_edge_csv

    path = tmp_path / "empty.csv"
    path.write_text("source,target\n", encoding="utf-8")
    with pytest.raises(EmptyGraphError):
        load_edge_csv(path)


def test_unsupported_format(tmp_path: Path) -> None:
    """未知后缀 → E105。"""
    from graphforge.data.loaders import load_graph

    path = tmp_path / "graph.parquet"
    path.write_text("dummy", encoding="utf-8")
    with pytest.raises(UnsupportedFileFormatError):
        load_graph(path)


def test_load_graph_dispatch(tmp_path: Path) -> None:
    """load_graph 按后缀分派到 edgelist / edge_csv。"""
    from graphforge.data.loaders import load_graph

    edge_path = tmp_path / "g.txt"
    edge_path.write_text("x y\n", encoding="utf-8")
    assert load_graph(edge_path).num_nodes == 2
    csv_path = tmp_path / "g.csv"
    csv_path.write_text("source,target\n1,2\n", encoding="utf-8")
    assert load_graph(csv_path).num_nodes == 2


def test_features_csv_and_attach(tmp_path: Path) -> None:
    """特征 CSV 读取 + attach_features 形状正确。"""
    from graphforge.data.loaders import attach_features, load_edgelist, load_feature_csv

    edge_path = tmp_path / "edges.txt"
    edge_path.write_text("a b\nb c\n", encoding="utf-8")
    graph = load_edgelist(edge_path)
    feat_path = tmp_path / "feats.csv"
    feat_path.write_text("node,f1,f2\na,0.5,1.5\nb,1.0,2.0\nc,2.0,3.0\n", encoding="utf-8")
    ids, features = load_feature_csv(feat_path)
    fused = attach_features(graph, ids, features)
    assert fused.node_features is not None
    assert np.asarray(fused.node_features).shape == (3, 2)


def test_registry_unknown_dataset() -> None:
    """未注册数据集 → E101。"""
    from graphforge.data.registry import get_dataset

    with pytest.raises(DatasetNotFoundError):
        get_dataset("no-such-dataset")
