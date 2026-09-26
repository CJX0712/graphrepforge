"""日志约定（L0）。

* 库内**只**调用 :func:`get_logger`，**不安装任何 handler**（避免污染宿主应用）；
* 只有 ``cli.py`` 调用 :func:`setup_logging` 安装 handler；
* 格式固定为 ``"%(asctime)s %(levelname)-7s [%(name)s] %(message)s"``。

作者：晨星
"""

from __future__ import annotations

import logging
import os
from typing import Optional

__all__ = [
    "LOG_FORMAT",
    "LOGGER_ROOT",
    "get_logger",
    "setup_logging",
    "log_level_from_env",
]

#: 统一日志格式。
LOG_FORMAT: str = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"

#: 库统一使用的 logger 前缀。
LOGGER_ROOT: str = "graphforge"


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger（不安装 handler）。

    Args:
        name: 通常是 ``__name__``。

    Returns:
        ``logging.Logger`` 实例。
    """
    return logging.getLogger(name if name.startswith(LOGGER_ROOT) else f"{LOGGER_ROOT}.{name}")


def log_level_from_env(default: str = "INFO") -> str:
    """从 ``GRAPHFORGE_LOG_LEVEL`` 读取日志级别。

    Args:
        default: 缺省级别。

    Returns:
        大写级别名；环境值非法时回退到 ``default``。
    """
    raw = os.environ.get("GRAPHFORGE_LOG_LEVEL", default)
    level = str(raw).strip().upper()
    valid = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    return level if level in valid else default


def setup_logging(level: Optional[str] = None, force: bool = True) -> logging.Logger:
    """安装 stream handler（**仅 CLI / demo 调用**）。

    Args:
        level: 日志级别；None 时取环境变量 ``GRAPHFORGE_LOG_LEVEL``。
        force: 是否先移除已有 handler（避免重复输出）。

    Returns:
        ``graphforge`` 根 logger。
    """
    logger = logging.getLogger(LOGGER_ROOT)
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, str(level or log_level_from_env()).upper(), logging.INFO))
    logger.propagate = False
    return logger
