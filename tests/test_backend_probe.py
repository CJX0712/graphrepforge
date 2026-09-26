"""后端探测单测：永不抛异常、降级语义正确。"""

from __future__ import annotations


def test_available_backends_never_raises() -> None:
    """available_backends 永不抛异常，internal 恒为 True。"""
    import graphforge

    status = graphforge.available_backends()
    assert isinstance(status, dict)
    assert status["internal"] is True
    assert isinstance(status["karateclub"], bool)


def test_resolve_backend_fallback() -> None:
    """karateclub 偏好在不可用时降级 internal。"""
    from graphforge.graph.embeddings.backends import available_backends, resolve_backend

    resolved = resolve_backend("auto")
    assert resolved in available_backends()
    if not available_backends()["karateclub"]:
        assert resolve_backend("karateclub") == "internal"


def test_karateclub_status_structure() -> None:
    """状态报告字段齐全。"""
    from graphforge.graph.embeddings.backends import karateclub_status

    status = karateclub_status()
    assert "available" in status
    if not status["available"]:
        assert status.get("error")  # 失败原因可读


def test_backend_report_rows() -> None:
    """backend_report 每行字段完整。"""
    from graphforge.graph.embeddings.backends import backend_report

    rows = backend_report()
    assert rows
    for row in rows:
        assert {"backend", "status", "detail"} <= set(row)


def test_karateclub_build_raises_when_unavailable() -> None:
    """不可用时构造官方嵌入器 → E301。"""
    from graphforge.core.errors import BackendUnavailableError
    from graphforge.graph.embeddings.backends import available_backends, build_karateclub_embedder

    if available_backends()["karateclub"]:
        return  # 已安装时跳过该断言
    try:
        build_karateclub_embedder("deepwalk", dim=8)
    except BackendUnavailableError:
        return
    raise AssertionError("expected BackendUnavailableError")
