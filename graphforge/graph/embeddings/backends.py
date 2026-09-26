"""可选后端探测（karateclub / node2vec）——**永不抛异常**。

设计铁律（架构 §10 坑 5）：
    1. 只在**本模块内部** try/except 懒加载重型可选依赖；
    2. :func:`available_backends` 恒定返回字典，**任何失败都退化为 False**；
    3. 只有"全后端不可用"才由调用方决定是否抛 ``E504``（本模块不抛）。

当前环境实测结论（Windows / py3.13）：``karateclub`` 与 ``node2vec`` **均不可用**
（依赖链要求 numpy<2 → 触发源码构建失败）。因此 :func:`available_backends`
恒返回 ``{"karateclub": False, "internal": True}``，benchmark 表 ``backend`` 列恒为
``internal``。探测代码保留，换环境后可自动启用官方实现。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from graphforge.core.errors import BackendUnavailableError
from graphforge.core.logging import get_logger
from graphforge.core.types import BackendStatus

__all__ = [
    "available_backends",
    "karateclub_status",
    "node2vec_status",
    "build_karateclub_embedder",
    "resolve_backend",
    "backend_report",
    "BACKEND_INTERNAL",
    "BACKEND_KARATECLUB",
]

_LOGGER = get_logger(__name__)

#: 内置后端名。
BACKEND_INTERNAL: str = "internal"

#: karateclub 后端名。
BACKEND_KARATECLUB: str = "karateclub"

#: karateclub 方法名 → 其构造器名（仅在 karateclub 可用时使用）。
KARATECLUB_MODELS: Dict[str, str] = {
    "deepwalk": "DeepWalk",
    "node2vec": "Node2Vec",
    "diff2vec": "Diff2Vec",
    "grarep": "GraRep",
    "hope": "HOPE",
    "netmf": "NetMF",
}


def _probe(module_name: str) -> Dict[str, Any]:
    """探测一个可选模块是否可导入。

    Args:
        module_name: 模块名（如 ``"karateclub"``）。

    Returns:
        ``{"available": bool, "version": str | None, "error": str | None}``。
    """
    import importlib

    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        return {"available": True, "version": None if version is None else str(version), "error": None}
    except Exception as exc:  # noqa: BLE001 - 任何失败都视为不可用
        return {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}".split("\n")[0][:200],
        }


def karateclub_status() -> Dict[str, Any]:
    """返回 karateclub 后端状态（**不抛异常**）。

    Returns:
        ``{"available": bool, "version": str | None, "error": str | None}``。
    """
    return _probe("karateclub")


def node2vec_status() -> Dict[str, Any]:
    """返回 node2vec 后端状态（**不抛异常**）。"""
    return _probe("node2vec")


def available_backends() -> Dict[str, bool]:
    """探测全部嵌入后端可用性（**永不抛异常**）。

    Returns:
        ``{"karateclub": bool, "internal": bool}``；``internal`` 恒为 True。
    """
    try:
        karateclub = bool(karateclub_status().get("available", False))
    except Exception:  # pragma: no cover - 双保险
        karateclub = False
    return {BACKEND_KARATECLUB: karateclub, BACKEND_INTERNAL: True}


def build_karateclub_embedder(
    method: str, dim: int = 128, **params: Any
) -> Any:
    """构造 karateclub 官方嵌入器（不可用时抛 ``E301``）。

    Args:
        method: 方法名（``deepwalk`` / ``node2vec`` / ...）。
        dim: 嵌入维度。
        **params: 透传给 karateclub 构造器。

    Returns:
        karateclub 估计器实例。

    Raises:
        BackendUnavailableError: karateclub 不可用或方法名未映射。
    """
    status = karateclub_status()
    if not status["available"]:
        raise BackendUnavailableError(
            "karateclub 不可用，无法构造官方嵌入器",
            hint="改用内置实现（backend=internal）或安装 karateclub",
        )
    if method not in KARATECLUB_MODELS:
        raise BackendUnavailableError(
            f"karateclub 未映射方法 {method!r}", hint=f"可选值：{sorted(KARATECLUB_MODELS)}"
        )
    try:
        import karateclub  # noqa: WPS433 - 已确认可用，此处才真正导入

        factory = getattr(karateclub, KARATECLUB_MODELS[method])
        estimator = factory(dimensions=int(dim), **params)
        _LOGGER.info("使用 karateclub 后端：%s（dim=%d）", method, dim)
        return estimator
    except Exception as exc:  # noqa: BLE001 - 官方实现参数不兼容也要转成 E301
        raise BackendUnavailableError(
            f"构造 karateclub {method} 失败：{exc}", hint="检查参数兼容性"
        ) from exc


def resolve_backend(preferred: str = "auto") -> str:
    """按偏好解析实际可用的后端。

    Args:
        preferred: ``auto`` / ``internal`` / ``karateclub``。

    Returns:
        实际后端名；karateclub 不可用时一律 ``internal``。
    """
    choice = str(preferred or "auto").lower()
    if choice in (BACKEND_KARATECLUB,):
        if available_backends().get(BACKEND_KARATECLUB, False):
            return BACKEND_KARATECLUB
        _LOGGER.warning("偏好 karateclub 但后端不可用，降级为 internal")
        return BACKEND_INTERNAL
    return BACKEND_INTERNAL


def backend_report() -> List[Dict[str, str]]:
    """返回后端可用性表（供 ``cli doctor`` 渲染，**不抛异常**）。

    Returns:
        每行形如 ``{"backend": ..., "status": ..., "version": ..., "detail": ...}``。
    """
    rows: List[Dict[str, str]] = []
    karateclub = karateclub_status()
    rows.append(
        {
            "backend": BACKEND_KARATECLUB,
            "status": BackendStatus.AVAILABLE.value
            if karateclub["available"]
            else BackendStatus.UNAVAILABLE.value,
            "version": karateclub.get("version") or "-",
            "detail": karateclub.get("error") or "官方实现",
        }
    )
    rows.append(
        {
            "backend": BACKEND_INTERNAL,
            "status": BackendStatus.AVAILABLE.value,
            "version": "v0.1.0",
            "detail": "内置经典论文复现（非 SOTA）",
        }
    )
    return rows
