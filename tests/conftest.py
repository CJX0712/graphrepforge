"""pytest 公共 fixture。

约定：
    * 每个用例前后都 ``reset_config()``，避免环境变量互相污染；
    * 图 fixture 全部**惰性构造**（函数体内 import），保证 core 层单测不依赖上层。

作者：晨星
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

from graphforge.core.config import reset_config

#: 项目根（tests/ 的上一级）。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_config() -> Iterator[None]:
    """每个用例前后重置全局配置单例。"""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def fixed_seed() -> int:
    """固定随机种子。"""
    return 42


@pytest.fixture
def project_root() -> Path:
    """项目根目录。"""
    return PROJECT_ROOT


@pytest.fixture
def tmp_artifacts(tmp_path: Path) -> Path:
    """临时产物目录（等价于 ``artifacts/``，但不污染仓库）。"""
    target = tmp_path / "artifacts"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def small_sbm() -> Any:
    """小型 SBM：3 社区 × 25 节点，带标签。"""
    from graphforge.data.synthetic import generate_sbm

    return generate_sbm(
        n_blocks=3, block_size=25, p_in=0.25, p_out=0.02, random_state=42, name="sbm_small"
    )


@pytest.fixture
def tiny_sbm() -> Any:
    """极小 SBM：2 社区 × 15 节点，用于嵌入类用例（保证秒级完成）。"""
    from graphforge.data.synthetic import generate_sbm

    return generate_sbm(
        n_blocks=2, block_size=15, p_in=0.35, p_out=0.05, random_state=42, name="sbm_tiny"
    )


@pytest.fixture
def karate() -> Any:
    """Zachary 空手道俱乐部图（34 节点 / 78 边）。"""
    from graphforge.data.synthetic import generate_karate

    return generate_karate()


@pytest.fixture
def grid() -> Any:
    """10×10 网格图。"""
    from graphforge.data.synthetic import generate_grid

    return generate_grid(m=10, n=10, random_state=42)


@pytest.fixture
def spec_node_clf() -> Any:
    """节点分类任务规格。"""
    from graphforge.core.types import TaskSpec, TaskType

    return TaskSpec(task=TaskType.NODE_CLASSIFICATION, random_state=42)


@pytest.fixture
def spec_link_pred() -> Any:
    """链路预测任务规格。"""
    from graphforge.core.types import TaskSpec, TaskType

    return TaskSpec(
        task=TaskType.LINK_PREDICTION,
        scoring=("roc_auc", "average_precision"),
        random_state=42,
    )
