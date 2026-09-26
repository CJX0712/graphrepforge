"""HPO 后端探测与解析（**永不抛异常**的探测入口）。

后端优先级：``optuna``（TPE）> ``random`` > ``grid``；
``optuna`` 不可用时自动降级，显式指定不可用后端才抛 E404。

作者：晨星
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from graphforge.core.errors import HPOBackendUnavailableError

__all__ = [
    "available_optimizers",
    "resolve_optimizer",
    "backend_report",
]

_LOGGER = logging.getLogger(__name__)


def _optuna_available() -> bool:
    """探测 optuna 是否可导入（吃掉一切异常）。"""
    try:
        import optuna  # noqa: F401, WPS433 - 探测性导入

        return True
    except Exception:  # noqa: BLE001 - 任何导入失败都视为不可用
        return False


def available_optimizers() -> Dict[str, bool]:
    """返回 ``{"optuna": bool, "random": True, "grid": True}``。

    ``random`` / ``grid`` 为内置实现，恒可用；``optuna`` 按探测结果。
    """
    return {"optuna": _optuna_available(), "random": True, "grid": True}


def resolve_optimizer(preferred: str = "auto") -> str:
    """按偏好解析实际可用的优化器。

    Args:
        preferred: ``auto`` / ``optuna`` / ``random`` / ``grid``。

    Returns:
        实际优化器名。

    Raises:
        HPOBackendUnavailableError: 显式指定的后端不可用（E404）。
    """
    choice = str(preferred or "auto").lower()
    status = available_optimizers()
    if choice == "auto":
        if status["optuna"]:
            return "optuna"
        _LOGGER.warning("optuna 不可用，HPO 自动降级为 random search")
        return "random"
    if not status.get(choice, False):
        raise HPOBackendUnavailableError(
            f"HPO 后端 {choice!r} 不可用", hint=f"当前可用：{sorted(k for k, v in status.items() if v)}"
        )
    return choice


def backend_report() -> List[Dict[str, str]]:
    """返回 HPO 后端状态表（供 ``cli doctor`` 渲染）。

    Returns:
        每行 ``{"backend": ..., "status": ..., "detail": ...}``。
    """
    status = available_optimizers()
    rows: List[Dict[str, str]] = []
    detail = ""
    if status["optuna"]:
        try:
            import optuna  # WPS433 - 展示版本号

            detail = f"optuna=={getattr(optuna, '__version__', '?')}"
        except Exception:  # pragma: no cover - 理论不可达
            detail = ""
    else:
        detail = "未安装（可用 pip install optuna 启用 TPE 搜索）"
    rows.append(
        {"backend": "optuna", "status": "available" if status["optuna"] else "unavailable", "detail": detail}
    )
    rows.append({"backend": "random", "status": "available", "detail": "内置随机搜索"})
    rows.append({"backend": "grid", "status": "available", "detail": "内置网格搜索"})
    return rows
