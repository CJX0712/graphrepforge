"""core.errors 错误体系单测。

作者：晨星
"""

from __future__ import annotations

import pytest

from graphforge.core.errors import (
    GraphForgeError,
    InvalidEmbeddingParamError,
    error_class,
    error_codes,
    raise_error,
)


def test_error_string_format() -> None:
    """``str(exc)`` 形如 ``"[E303] message (key=value; hint=...)"``。"""
    error = InvalidEmbeddingParamError("参数 dim 为保留参数", hint="请在构造器传入")
    text = str(error)
    assert text.startswith("[E303] ")
    assert "hint=请在构造器传入" in text
    assert error.code == "E303"
    assert error.hint == "请在构造器传入"


def test_error_without_context() -> None:
    """无上下文时不带括号尾部。"""
    error = GraphForgeError("配置错误", code="E501")
    assert str(error) == "[E501] 配置错误"


def test_error_codes_unique_and_complete() -> None:
    """错误码唯一，且覆盖 E101–E505 的关键区间。"""
    codes = error_codes()
    assert len(codes) == len(set(codes))
    for prefix in ("1", "2", "3", "4", "5"):
        assert any(code.startswith(f"E{prefix}") for code in codes)
    assert "E303" in codes and "E505" in codes


def test_error_class_lookup() -> None:
    """error_class 按码取类；未知码回退基类。"""
    assert error_class("E303") is InvalidEmbeddingParamError
    assert error_class("E999") is GraphForgeError


def test_raise_error_factory() -> None:
    """raise_error 产出对应类的实例（不自动抛出）。"""
    error = raise_error("E101", "数据集不存在", hint="检查名字")
    assert isinstance(error, GraphForgeError)
    assert error.code == "E101"
    with pytest.raises(GraphForgeError) as info:
        raise error
    assert info.value.code == "E101"


def test_all_errors_are_catchable_by_base() -> None:
    """全部错误均为 GraphForgeError 子类，可统一捕获。"""
    for code in error_codes():
        with pytest.raises(GraphForgeError):
            raise raise_error(code, f"test {code}")


def test_error_to_dict() -> None:
    """to_dict 可序列化且含 code/message/context。"""
    payload = InvalidEmbeddingParamError("非法参数", dim=128).to_dict()
    assert payload["code"] == "E303"
    assert payload["message"] == "非法参数"
    assert payload["context"]["dim"] == 128
