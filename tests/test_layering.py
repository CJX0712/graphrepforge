"""分层依赖静态断言：core 零反向依赖、禁止循环 import。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: 层级（数字越小越底层）。
LAYER_ORDER = {"core": 0, "data": 1, "preprocess": 1, "training": 1, "eval": 1, "graph": 2, "hpo": 3, "pipeline": 4}

#: 允许的上层依赖（key 依赖 value 集合）。
ALLOWED_DEPS = {
    "core": set(),
    "data": {"core"},
    "preprocess": {"core"},
    "training": {"core", "eval"},  # cv.py 延迟解析 eval 指标（单向）
    "eval": {"core"},
    "graph": {"core", "data", "preprocess", "training", "eval"},
    "hpo": {"core", "data", "graph", "training", "eval"},  # runner 需加载 dataset（单向）
    "pipeline": {"core", "data", "preprocess", "graph", "training", "hpo", "eval"},
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _package_root() -> Path:
    return _project_root() / "graphforge"


def _imports_of(path: Path) -> list:
    """AST 解析模块的 graphforge.* import 目标。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names if alias.name.startswith("graphforge"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("graphforge"):
            modules.append(node.module)
    return modules


def test_core_has_no_upstream_imports() -> None:
    """core 不得 import 任何上层模块。"""
    core_dir = _package_root() / "core"
    for path in core_dir.rglob("*.py"):
        for module in _imports_of(path):
            parts = module.split(".")
            if len(parts) >= 2:
                assert parts[1] == "core", f"{path.name} 非法依赖 {module}"


def test_layering_rules_respected() -> None:
    """全部模块遵守单向无环依赖规则。"""
    violations = []
    for layer in LAYER_ORDER:
        layer_dir = _package_root() / layer
        if not layer_dir.exists():
            continue
        for path in layer_dir.rglob("*.py"):
            for module in _imports_of(path):
                parts = module.split(".")
                if len(parts) >= 2 and parts[1] in LAYER_ORDER:
                    target = parts[1]
                    if target != layer and target not in ALLOWED_DEPS[layer]:
                        violations.append(f"{layer}/{path.name} → {module}")
    assert not violations, "依赖违规：\n" + "\n".join(violations)


def test_no_circular_imports() -> None:
    """逐个导入全部顶层子包，无循环 import 异常。"""
    import importlib

    for layer in LAYER_ORDER:
        module = importlib.import_module(f"graphforge.{layer}")
        assert module is not None
