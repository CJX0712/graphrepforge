"""HPO 子包：搜索空间 / 后端探测 / 优化器 / 执行入口。"""

from __future__ import annotations

from graphforge.core.types import HPOSpec, ParamSpec, SearchSpace, TrialResult
from graphforge.hpo.backends import available_optimizers, backend_report, resolve_optimizer
from graphforge.hpo.optimizer import (
    BaseOptimizer,
    GridSearchOptimizer,
    OptunaOptimizer,
    RandomSearchOptimizer,
    make_optimizer,
)
from graphforge.hpo.runner import run_hpo
from graphforge.hpo.space import embedder_space

__all__ = [
    "HPOSpec",
    "ParamSpec",
    "SearchSpace",
    "TrialResult",
    "BaseOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "OptunaOptimizer",
    "make_optimizer",
    "available_optimizers",
    "resolve_optimizer",
    "backend_report",
    "run_hpo",
    "embedder_space",
]
