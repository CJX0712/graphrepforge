"""core.config 配置与环境变量覆盖单测。

作者：晨星
"""

from __future__ import annotations

import pytest

from graphforge.core.config import (
    GraphForgeConfig,
    config_context,
    get_config,
    reset_config,
    set_config,
    update_config,
)
from graphforge.core.errors import ConfigError


def test_defaults() -> None:
    """默认配置取值正确。"""
    config = get_config()
    assert config.random_state == 42
    assert config.log_level == "INFO"
    assert config.dim == 128
    assert config.embed_backend == "auto"
    assert config.hpo_backend == "auto"
    assert config.hpo_trials == 20
    assert config.output_dir == "artifacts"
    assert config.n_jobs == 1


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """GRAPHFORGE_* 环境变量可覆盖默认值。"""
    monkeypatch.setenv("GRAPHFORGE_RANDOM_STATE", "7")
    monkeypatch.setenv("GRAPHFORGE_DIM", "64")
    monkeypatch.setenv("GRAPHFORGE_LOG_LEVEL", "debug")
    monkeypatch.setenv("GRAPHFORGE_OUTPUT_DIR", "outputs")
    monkeypatch.setenv("GRAPHFORGE_N_JOBS", "4")
    reset_config()
    config = get_config()
    assert config.random_state == 7
    assert config.dim == 64
    assert config.log_level_name == "DEBUG"
    assert config.output_dir == "outputs"
    assert config.n_jobs == 4


def test_env_invalid_int_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """非整数环境变量 → E501。"""
    monkeypatch.setenv("GRAPHFORGE_DIM", "not-a-number")
    with pytest.raises(ConfigError) as info:
        reset_config()
        get_config()
    assert info.value.code == "E501"


def test_validate_rejects_bad_values() -> None:
    """非法字段取值 → E501。"""
    with pytest.raises(ConfigError):
        GraphForgeConfig(dim=0).validate()
    with pytest.raises(ConfigError):
        GraphForgeConfig(n_jobs=0).validate()
    with pytest.raises(ConfigError):
        GraphForgeConfig(log_level="TRACE").validate()
    with pytest.raises(ConfigError):
        GraphForgeConfig(embed_backend="torch").validate()
    with pytest.raises(ConfigError):
        GraphForgeConfig(hpo_backend="magic").validate()


def test_output_path_resolution() -> None:
    """output_path 相对路径按项目根解析，绝对路径原样返回。"""
    config = GraphForgeConfig(output_dir="artifacts")
    resolved = config.output_path("benchmark.json")
    assert resolved.is_absolute()
    assert resolved.parts[-1] == "benchmark.json"
    assert resolved.parent.name == "artifacts"


def test_set_update_and_context() -> None:
    """set_config / update_config / config_context 行为正确。"""
    set_config(GraphForgeConfig(dim=32))
    assert get_config().dim == 32
    update_config(dim=16)
    assert get_config().dim == 16
    with config_context(dim=8):
        assert get_config().dim == 8
    assert get_config().dim == 16  # 退出后恢复


def test_reset_returns_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """reset_config 清空环境变量影响后回到默认值。"""
    monkeypatch.setenv("GRAPHFORGE_DIM", "33")
    reset_config()
    assert get_config().dim == 33
    monkeypatch.delenv("GRAPHFORGE_DIM")
    reset_config()
    assert get_config().dim == 128


def test_to_dict_contains_keys() -> None:
    """to_dict 含全部关键字段。"""
    payload = get_config().to_dict()
    for key in ("random_state", "log_level", "dim", "embed_backend", "n_jobs"):
        assert key in payload
