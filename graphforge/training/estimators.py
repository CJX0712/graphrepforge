"""下游估计器工厂（**三参分离，杜绝参数泄漏**）。

铁律（架构 §10 坑 3）：
    * ``random_state`` 等内部参数由工厂注入，**不混进 ``params``**；
    * ``name`` / ``params`` / ``random_state`` 三个参数位分离；
    * 未知参数在**构造前**用签名检查拦下，统一抛 ``E401``。

sklearn 类一律以 ``_sk_`` 前缀别名导入（防递归坑）。

作者：晨星
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type

# --- sklearn 别名导入 ---
from sklearn.ensemble import RandomForestClassifier as _sk_RandomForestClassifier
from sklearn.linear_model import LogisticRegression as _sk_LogisticRegression

from graphforge.core.errors import EstimatorBuildError
from graphforge.core.logging import get_logger

__all__ = [
    "ESTIMATOR_REGISTRY",
    "register_estimator",
    "list_estimators",
    "make_estimator",
    "validate_estimator_params",
    "DEFAULT_ESTIMATOR",
]

_LOGGER = get_logger(__name__)

#: 默认估计器名。
DEFAULT_ESTIMATOR: str = "logistic_regression"


def _build_logistic(params: Dict[str, Any], random_state: int) -> Any:
    """构造 LogisticRegression（``max_iter`` 默认 1000，避免收敛告警）。"""
    merged: Dict[str, Any] = {"max_iter": 1000, **params}
    merged["random_state"] = int(random_state)
    return _sk_LogisticRegression(**merged)


def _build_random_forest(params: Dict[str, Any], random_state: int) -> Any:
    """构造 RandomForestClassifier（``n_estimators`` 默认 200）。"""
    merged: Dict[str, Any] = {"n_estimators": 200, **params}
    merged["random_state"] = int(random_state)
    merged.setdefault("n_jobs", 1)
    return _sk_RandomForestClassifier(**merged)


#: 估计器名 → 工厂函数 ``(params, random_state) -> estimator``。
ESTIMATOR_REGISTRY: Dict[str, Callable[[Dict[str, Any], int], Any]] = {
    "logistic_regression": _build_logistic,
    "random_forest": _build_random_forest,
}


def register_estimator(
    name: str, factory: Callable[[Dict[str, Any], int], Any], force: bool = False
) -> None:
    """注册（或覆盖）一个估计器工厂。

    Args:
        name: 估计器名。
        factory: ``(params, random_state) -> estimator``。
        force: 允许覆盖已存在条目。

    Raises:
        EstimatorBuildError: 同名已存在且 ``force=False``。
    """
    if name in ESTIMATOR_REGISTRY and not force:
        raise EstimatorBuildError(
            f"估计器 {name!r} 已注册", hint="如需覆盖，请传 force=True"
        )
    ESTIMATOR_REGISTRY[name] = factory
    _LOGGER.debug("注册估计器：%s", name)


def list_estimators() -> List[str]:
    """返回已注册估计器名（升序）。"""
    return sorted(ESTIMATOR_REGISTRY)


def validate_estimator_params(name: str, params: Dict[str, Any]) -> None:
    """校验参数名是否被目标估计器接受。

    Args:
        name: 估计器名。
        params: 待校验参数字典。

    Raises:
        EstimatorBuildError: 估计器名未注册或存在未知参数。
    """
    if name not in ESTIMATOR_REGISTRY:
        raise EstimatorBuildError(
            f"未知估计器 {name!r}", hint=f"可选值：{list_estimators()}"
        )
    allowed = _allowed_params(name)
    if allowed is None:
        return
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise EstimatorBuildError(
            f"估计器 {name!r} 不接受参数 {unknown}", hint=f"可用参数：{sorted(allowed)}"
        )


def _allowed_params(name: str) -> Optional[set]:
    """返回估计器构造器可接受的参数名集合；无法推断时返回 None（跳过校验）。"""
    mapping: Dict[str, Type[Any]] = {
        "logistic_regression": _sk_LogisticRegression,
        "random_forest": _sk_RandomForestClassifier,
    }
    cls = mapping.get(name)
    if cls is None:
        return None
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - C 扩展类可能无签名
        return None
    return {param for param in signature.parameters if param != "self"}


def make_estimator(
    name: str = DEFAULT_ESTIMATOR,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Any:
    """构造下游估计器。

    Args:
        name: 估计器名（``logistic_regression`` / ``random_forest``）。
        params: 估计器超参（**不含** ``random_state`` 等内部参数）。
        random_state: 随机种子，由本函数注入。

    Returns:
        sklearn 兼容估计器。

    Raises:
        EstimatorBuildError: 名字未注册、参数非法或构造失败（E401）。
    """
    payload: Dict[str, Any] = dict(params or {})
    validate_estimator_params(name, payload)
    factory = ESTIMATOR_REGISTRY[name]
    try:
        return factory(payload, int(random_state))
    except EstimatorBuildError:
        raise
    except Exception as exc:  # noqa: BLE001 - sklearn 构造异常统一包成 E401
        raise EstimatorBuildError(
            f"构造估计器 {name!r} 失败：{exc}", hint="检查 estimator_params"
        ) from exc
