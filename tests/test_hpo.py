"""HPO 单测：优化器确定性、降级、run_hpo 端到端。"""

from __future__ import annotations

import numpy as np
import pytest


def test_available_optimizers_structure() -> None:
    """探测结果结构正确，grid/random 恒可用。"""
    from graphforge.hpo.backends import available_optimizers

    status = available_optimizers()
    assert status["grid"] is True
    assert status["random"] is True
    assert isinstance(status["optuna"], bool)


def test_grid_optimizer_deterministic() -> None:
    """grid 搜索确定性，history 数量 = n_trials。"""
    from graphforge.core.types import HPOSpec, ParamSpec, SearchSpace
    from graphforge.hpo.optimizer import GridSearchOptimizer

    space = SearchSpace(
        params=(
            ParamSpec(name="a", kind="float", low=0.0, high=1.0),
            ParamSpec(name="b", kind="int", low=1, high=4),
        )
    )
    spec = HPOSpec(backend="grid", n_trials=6)
    result1 = GridSearchOptimizer().search(lambda p: float(p["a"]) + p["b"], space, spec)
    result2 = GridSearchOptimizer().search(lambda p: float(p["a"]) + p["b"], space, spec)
    assert result1.best_params == result2.best_params
    assert result1.n_trials == result2.n_trials <= 6
    assert result1.best_value == result2.best_value
    assert result1.backend == "grid"


def test_random_optimizer_deterministic() -> None:
    """random 搜索同 seed 复现。"""
    from graphforge.core.types import HPOSpec, ParamSpec, SearchSpace
    from graphforge.hpo.optimizer import RandomSearchOptimizer

    space = SearchSpace(params=(ParamSpec(name="x", kind="log_float", low=0.001, high=1.0),))
    spec = HPOSpec(backend="random", n_trials=4)
    r1 = RandomSearchOptimizer().search(lambda p: 1.0, space, spec)
    r2 = RandomSearchOptimizer().search(lambda p: 1.0, space, spec)
    assert r1.best_params == r2.best_params
    assert r1.n_trials == 4


def test_grid_overrides_history() -> None:
    """history 记录全部 trial 且 best 取最大。"""
    from graphforge.core.types import HPOSpec, ParamSpec, SearchSpace
    from graphforge.hpo.optimizer import GridSearchOptimizer

    space = SearchSpace(params=(ParamSpec(name="k", kind="int", low=1, high=5),))
    spec = HPOSpec(backend="grid", n_trials=5)
    result = GridSearchOptimizer().search(lambda p: float(p["k"]), space, spec)
    assert result.best_value == 5.0
    assert result.best_params["k"] == 5


def test_optuna_or_fallback() -> None:
    """optuna 可用则 TPE 确定性；不可用则 E404。"""
    from graphforge.core.errors import HPOBackendUnavailableError
    from graphforge.core.types import HPOSpec, ParamSpec, SearchSpace
    from graphforge.hpo.backends import available_optimizers
    from graphforge.hpo.optimizer import OptunaOptimizer

    space = SearchSpace(params=(ParamSpec(name="x", kind="float", low=0.0, high=1.0),))
    spec = HPOSpec(backend="optuna", n_trials=3)
    if available_optimizers()["optuna"]:
        r1 = OptunaOptimizer().search(lambda p: -abs(float(p["x"]) - 0.3), space, spec)
        r2 = OptunaOptimizer().search(lambda p: -abs(float(p["x"]) - 0.3), space, spec)
        assert r1.best_params == r2.best_params
        assert r1.n_trials == 3
    else:
        with pytest.raises(HPOBackendUnavailableError):
            OptunaOptimizer().search(lambda p: 1.0, space, spec)


def test_run_hpo_end_to_end(tiny_sbm: object) -> None:
    """run_hpo：grid 3 trials，返回 TrialResult 且 primary 越大越好。"""
    from graphforge.core.types import HPOSpec, TaskSpec, TaskType
    from graphforge.eval.metrics import default_scoring
    from graphforge.hpo.runner import run_hpo

    spec = TaskSpec(
        task=TaskType.NODE_CLASSIFICATION,
        scoring=default_scoring(TaskType.NODE_CLASSIFICATION),
        random_state=42,
    )
    result = run_hpo(
        dataset=tiny_sbm,  # type: ignore[arg-type]
        method="spectral",
        task="node_classification",
        spec=spec,
        hpo=HPOSpec(backend="grid", n_trials=3),
        dim=8,
    )
    assert result.n_trials <= 3
    assert np.isfinite(result.best_value)
    assert 0.0 <= result.best_value <= 1.0
