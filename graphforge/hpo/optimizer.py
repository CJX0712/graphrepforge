"""HPO 优化器：grid / random / optuna 三后端统一 ``search()`` 接口。

约定：
    * ``objective(params) -> float`` 返回**越大越好**的目标值；
    * 所有采样确定性：random 用 :func:`graphforge.core.utils.make_rng`，
      optuna 用 ``TPESampler(seed=...)``；
    * 失败的 trial 记为 ``-inf`` 继续，不中断搜索。

作者：晨星
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import numpy as np

from graphforge.core.errors import HPOBackendUnavailableError
from graphforge.core.types import HPOSpec, SearchSpace, TrialResult
from graphforge.core.utils import make_rng

__all__ = [
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "OptunaOptimizer",
    "make_optimizer",
]

_LOGGER = logging.getLogger(__name__)

Objective = Callable[[Dict[str, Any]], float]


class BaseOptimizer(ABC):
    """优化器基类：统一 ``search`` 签名。"""

    #: 优化器名（子类覆盖）。
    name: str = "base"

    @abstractmethod
    def search(self, objective: Objective, space: SearchSpace, spec: HPOSpec) -> TrialResult:
        """执行搜索。

        Args:
            objective: 目标函数（越大越好）。
            space: 搜索空间。
            spec: HPO 配置。

        Returns:
            :class:`TrialResult`。
        """


def _evaluate(objective: Objective, params: Dict[str, Any]) -> float:
    """安全评估单组参数：异常记为 -inf 并告警。"""
    try:
        return float(objective(params))
    except Exception as exc:  # noqa: BLE001 - 单 trial 失败不拖垮搜索
        _LOGGER.warning("trial 失败 params=%s: %s", params, exc)
        return float("-inf")


def _pack(best_value: float, history: List[Dict[str, Any]], backend: str) -> TrialResult:
    """从 history 汇总 :class:`TrialResult`（best 取 history 最大）。"""
    if history:
        best_item = max(history, key=lambda item: float(item["value"]))
        return TrialResult(
            best_params=dict(best_item["params"]),
            best_value=float(best_item["value"]),
            n_trials=len(history),
            backend=backend,
            history=list(history),
        )
    return TrialResult(best_params={}, best_value=float("-inf"), n_trials=0, backend=backend, history=[])


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索：至多 ``n_trials`` 组确定性组合。"""

    name = "grid"

    def search(self, objective: Objective, space: SearchSpace, spec: HPOSpec) -> TrialResult:
        """遍历 :meth:`SearchSpace.grid` 组合并评估。"""
        space.validate()
        combos = space.grid(spec.n_trials)
        history: List[Dict[str, Any]] = []
        for params in combos:
            value = _evaluate(objective, params)
            history.append({"params": dict(params), "value": value})
        return _pack(max((h["value"] for h in history), default=float("-inf")), history, self.name)


class RandomSearchOptimizer(BaseOptimizer):
    """随机搜索：确定性采样 ``n_trials`` 组。"""

    name = "random"

    def search(self, objective: Objective, space: SearchSpace, spec: HPOSpec) -> TrialResult:
        """用 ``make_rng(random_state, "hpo", "random")`` 采样评估。"""
        space.validate()
        rng = make_rng(spec.random_state, "hpo", "random")
        history: List[Dict[str, Any]] = []
        for _ in range(int(spec.n_trials)):
            params = space.sample(rng)
            value = _evaluate(objective, params)
            history.append({"params": dict(params), "value": value})
        return _pack(max((h["value"] for h in history), default=float("-inf")), history, self.name)


class OptunaOptimizer(BaseOptimizer):
    """Optuna TPE 搜索（懒 import，不可用时抛 E404）。"""

    name = "optuna"

    def search(self, objective: Objective, space: SearchSpace, spec: HPOSpec) -> TrialResult:
        """用 ``optuna.create_study`` 做 TPE 搜索。"""
        space.validate()
        try:
            import optuna
            from optuna.samplers import TPESampler
        except Exception as exc:  # noqa: BLE001 - 依赖缺失统一 E404
            raise HPOBackendUnavailableError(
                "optuna 不可用", hint="pip install optuna 或改用 backend=random"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = TPESampler(seed=int(spec.random_state))
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _suggest(trial: "optuna.Trial") -> Dict[str, Any]:
            params: Dict[str, Any] = {}
            for item in space.params:
                if item.kind == "categorical":
                    params[item.name] = trial.suggest_categorical(item.name, list(item.choices or []))
                elif item.kind == "int":
                    params[item.name] = trial.suggest_int(
                        item.name, int(item.low or 0), int(item.high or 0)
                    )
                elif item.kind == "log_float":
                    params[item.name] = trial.suggest_float(
                        item.name, float(item.low or 1e-12), float(item.high or 1.0), log=True
                    )
                else:
                    params[item.name] = trial.suggest_float(
                        item.name, float(item.low or 0.0), float(item.high or 1.0)
                    )
            return params

        def _optuna_objective(trial: "optuna.Trial") -> float:
            params = _suggest(trial)
            value = _evaluate(objective, params)
            if not np.isfinite(value):
                value = float("-1e12")  # optuna 不接受 -inf
            return float(value)

        study.optimize(
            _optuna_objective,
            n_trials=int(spec.n_trials),
            timeout=spec.timeout_sec,
            show_progress_bar=False,
        )
        history: List[Dict[str, Any]] = [
            {"params": dict(trial.params), "value": float(trial.value if trial.value is not None else float("-inf"))}
            for trial in study.trials
        ]
        return _pack(0.0, history, self.name)


_OPTIMIZERS: Dict[str, type] = {
    "grid": GridSearchOptimizer,
    "random": RandomSearchOptimizer,
    "optuna": OptunaOptimizer,
}


def make_optimizer(name: str = "auto") -> BaseOptimizer:
    """按名字构造优化器（``auto`` 解析到实际可用后端）。

    Args:
        name: ``auto`` / ``optuna`` / ``random`` / ``grid``。

    Returns:
        优化器实例。

    Raises:
        HPOBackendUnavailableError: 名字非法或后端不可用（E404）。
    """
    from graphforge.hpo.backends import resolve_optimizer

    resolved = resolve_optimizer(name) if str(name).lower() == "auto" else str(name).lower()
    if resolved not in _OPTIMIZERS:
        raise HPOBackendUnavailableError(
            f"未知 HPO 后端 {name!r}", hint=f"可选值：{sorted(set(_OPTIMIZERS) | {'auto'})}"
        )
    return _OPTIMIZERS[resolved]()
