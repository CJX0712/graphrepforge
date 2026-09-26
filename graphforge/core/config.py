"""全局配置：``GraphForgeConfig`` + ``GRAPHFORGE_*`` 环境变量覆盖（12-factor）。

用法::

    from graphforge.core.config import get_config
    cfg = get_config()               # 首次调用时读取环境变量
    cfg = get_config().output_path() # 产物目录（相对项目根解析）

测试中用 :func:`reset_config` 清空单例，避免环境变量互相污染。

作者：晨星
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from graphforge.core.errors import ConfigError

__all__ = [
    "GraphForgeConfig",
    "get_config",
    "set_config",
    "reset_config",
    "update_config",
    "config_context",
    "ENV_PREFIX",
    "ENV_KEYS",
]

#: 环境变量前缀。
ENV_PREFIX: str = "GRAPHFORGE_"

#: 受支持的环境变量名（不含前缀）。
ENV_KEYS: Dict[str, str] = {
    "RANDOM_STATE": "random_state",
    "LOG_LEVEL": "log_level",
    "DIM": "dim",
    "EMBED_BACKEND": "embed_backend",
    "HPO_BACKEND": "hpo_backend",
    "HPO_TRIALS": "hpo_trials",
    "OUTPUT_DIR": "output_dir",
    "N_JOBS": "n_jobs",
}

#: 允许的日志级别。
LOG_LEVELS: tuple = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

#: 允许的嵌入后端。
EMBED_BACKENDS: tuple = ("auto", "internal", "karateclub")

#: 项目根目录：``graphforge/core/config.py`` → parents[2]。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _parse_int(raw: str, field_name: str) -> int:
    """把环境变量字符串解析为 int。

    Args:
        raw: 原始字符串。
        field_name: 目标字段名（用于报错）。

    Returns:
        解析后的整数。

    Raises:
        ConfigError: 无法解析。
    """
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"环境变量 {ENV_PREFIX}{field_name} 需要整数，实际 {raw!r}",
            hint="示例：GRAPHFORGE_DIM=128",
        ) from exc


@dataclass
class GraphForgeConfig:
    """GraphForge 全局配置。

    Attributes:
        random_state: 全局随机种子。
        log_level: 日志级别。
        dim: 默认嵌入维度。
        embed_backend: 嵌入后端偏好（auto / internal / karateclub）。
        hpo_backend: HPO 后端偏好（auto / optuna / random / grid / none）。
        hpo_trials: HPO 试验次数。
        output_dir: 产物目录（相对路径按项目根解析）。
        n_jobs: 并行度。
        extra: 其余自定义键值。
    """

    random_state: int = 42
    log_level: str = "INFO"
    dim: int = 128
    embed_backend: str = "auto"
    hpo_backend: str = "auto"
    hpo_trials: int = 20
    output_dir: str = "artifacts"
    n_jobs: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- 构造
    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "GraphForgeConfig":
        """从环境变量构造配置。

        Args:
            env: 环境变量字典；None 时使用 :data:`os.environ`。

        Returns:
            新的配置对象。

        Raises:
            ConfigError: 环境变量取值非法。
        """
        source = os.environ if env is None else env
        values: Dict[str, Any] = {}
        for suffix, field_name in ENV_KEYS.items():
            raw = source.get(ENV_PREFIX + suffix)
            if raw is None or str(raw).strip() == "":
                continue
            raw = str(raw).strip()
            if field_name in ("random_state", "dim", "hpo_trials", "n_jobs"):
                values[field_name] = _parse_int(raw, suffix)
            else:
                values[field_name] = raw
        config = cls(**values)
        config.validate()
        return config

    def copy_with(self, **changes: Any) -> "GraphForgeConfig":
        """返回带修改的副本。"""
        return replace(self, **changes)

    # ---------------------------------------------------------------- 校验
    def validate(self) -> None:
        """校验配置取值。

        Raises:
            ConfigError: dim / n_jobs / hpo_trials 非正，或枚举取值非法。
        """
        if self.dim <= 0:
            raise ConfigError(f"dim={self.dim} 非法", hint="嵌入维度必须 >= 1")
        if self.n_jobs < 1:
            raise ConfigError(f"n_jobs={self.n_jobs} 非法", hint="并行度必须 >= 1")
        if self.hpo_trials < 1:
            raise ConfigError(f"hpo_trials={self.hpo_trials} 非法", hint="试验次数必须 >= 1")
        level = str(self.log_level).upper()
        if level not in LOG_LEVELS:
            raise ConfigError(
                f"log_level={self.log_level!r} 非法", hint=f"可选值：{list(LOG_LEVELS)}"
            )
        backend = str(self.embed_backend).lower()
        if backend not in EMBED_BACKENDS:
            raise ConfigError(
                f"embed_backend={self.embed_backend!r} 非法", hint=f"可选值：{list(EMBED_BACKENDS)}"
            )
        hpo = str(self.hpo_backend).lower()
        if hpo not in ("auto", "optuna", "random", "grid", "none"):
            raise ConfigError(
                f"hpo_backend={self.hpo_backend!r} 非法",
                hint="可选值：auto / optuna / random / grid / none",
            )

    # ---------------------------------------------------------------- 派生
    @property
    def log_level_name(self) -> str:
        """返回大写日志级别名。"""
        return str(self.log_level).upper()

    def output_path(self, *parts: str) -> Path:
        """解析产物路径。

        Args:
            *parts: 追加的子路径。

        Returns:
            绝对路径。``output_dir`` 为绝对路径时直接拼接，否则相对项目根解析。
        """
        base = Path(self.output_dir)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base.joinpath(*parts) if parts else base

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（含 ``extra`` 展开）。"""
        data: Dict[str, Any] = {
            "random_state": self.random_state,
            "log_level": self.log_level_name,
            "dim": self.dim,
            "embed_backend": self.embed_backend,
            "hpo_backend": self.hpo_backend,
            "hpo_trials": self.hpo_trials,
            "output_dir": self.output_dir,
            "n_jobs": self.n_jobs,
        }
        data.update(self.extra)
        return data


# ---------------------------------------------------------------- 单例管理
_CONFIG: Optional[GraphForgeConfig] = None


def get_config() -> GraphForgeConfig:
    """获取全局配置单例（首次调用时从环境变量构造）。

    Returns:
        全局 :class:`GraphForgeConfig`。
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = GraphForgeConfig.from_env()
    return _CONFIG


def set_config(config: GraphForgeConfig) -> GraphForgeConfig:
    """设置全局配置单例。

    Args:
        config: 新配置（会先 ``validate()``）。

    Returns:
        被设置的配置。
    """
    global _CONFIG
    config.validate()
    _CONFIG = config
    return _CONFIG


def update_config(**changes: Any) -> GraphForgeConfig:
    """在现有配置上做局部修改。

    Args:
        **changes: 待修改字段。

    Returns:
        修改后的配置。
    """
    current = get_config()
    updated = current.copy_with(**changes)
    return set_config(updated)


def reset_config() -> GraphForgeConfig:
    """清空单例，下一次 :func:`get_config` 会重新读取环境变量。

    Returns:
        重置后的默认配置（已写入单例）。
    """
    global _CONFIG
    _CONFIG = GraphForgeConfig.from_env()
    return _CONFIG


@contextmanager
def config_context(**changes: Any) -> Iterator[GraphForgeConfig]:
    """临时修改配置的上下文管理器（退出时恢复原配置）。

    Args:
        **changes: 临时生效的字段。

    Yields:
        修改后的配置。
    """
    previous = get_config()
    updated = previous.copy_with(**changes)
    set_config(updated)
    try:
        yield updated
    finally:
        set_config(previous)
