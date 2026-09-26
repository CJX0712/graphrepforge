"""通用工具（L0）：种子派生、随机数发生器、计时、目录与序列化辅助。

关键约定（见 docs/architecture.md §9.2）：
    * 种子派生用 ``hashlib.sha256``，**禁用内置 ``hash()``**（字符串 hash 带随机盐）；
    * **禁用** ``np.random.*`` 全局调用，一律 ``np.random.default_rng(seed)``。

作者：晨星
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Union

import numpy as np
from scipy import sparse as sp

__all__ = [
    "SEED_TAGS",
    "to_csr",
    "symmetrize_adjacency",
    "remove_self_loops",
    "degree_vector",
    "derive_seed",
    "make_rng",
    "make_py_rng",
    "Timer",
    "timer",
    "ensure_dir",
    "stable_json",
    "save_json",
    "load_json",
    "format_duration",
    "clamp",
    "safe_div",
    "normalize_rows",
]

#: 各阶段种子标签（与架构文档 §9.2 保持一致）。
SEED_TAGS: Dict[str, str] = {
    "DATA": "data",
    "SPLIT": "split",
    "WALK": "walk",
    "SKIPGRAM": "skipgram",
    "SVD": "svd",
    "NEG": "neg",
    "ESTIMATOR": "estimator",
    "HPO": "hpo",
}

#: 派生种子的取值上限（与 numpy 的 SeedSequence 兼容）。
_SEED_UPPER: int = 2 ** 31 - 1


def derive_seed(base: Union[int, str], *tags: str) -> int:
    """基于 sha256 派生确定性子种子。

    Args:
        base: 基础种子（int 或 str）。
        *tags: 阶段标签，例如 ``"data"`` / ``"walk"`` / ``"svd"``。

    Returns:
        ``[0, 2**31 - 1]`` 范围内的整数。

    Examples:
        >>> derive_seed(42, "data") == derive_seed(42, "data")
        True
        >>> derive_seed(42, "data") == derive_seed(42, "walk")
        False
    """
    payload = "|".join([str(base)] + [str(tag) for tag in tags])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % _SEED_UPPER


def make_rng(seed: Union[int, str], *tags: str) -> np.random.Generator:
    """构造 ``numpy`` 随机数发生器（**不使用全局随机状态**）。

    Args:
        seed: 基础种子。
        *tags: 阶段标签；为空时直接用 ``seed``。

    Returns:
        ``np.random.Generator``。
    """
    value = derive_seed(seed, *tags) if tags else (int(seed) % _SEED_UPPER)
    return np.random.default_rng(value)


def make_py_rng(seed: Union[int, str], *tags: str) -> random.Random:
    """构造 ``random.Random``（纯 Python 侧随机，用于游走等热路径）。

    Args:
        seed: 基础种子。
        *tags: 阶段标签。

    Returns:
        ``random.Random`` 实例。
    """
    value = derive_seed(seed, *tags) if tags else (int(seed) % _SEED_UPPER)
    return random.Random(value)


class Timer:
    """计时器（可作上下文管理器）。

    Attributes:
        elapsed: 已耗时（秒）；进入时清零，退出时写入。
    """

    def __init__(self) -> None:
        """构造计时器，初始耗时为 0。"""
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        """开始计时。"""
        self.elapsed = 0.0
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        """结束计时；不吞异常。"""
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
            self._start = None
        return False


def timer() -> Timer:
    """创建 :class:`Timer`（供 ``with timer() as t:`` 使用）。

    Returns:
        新的计时器。
    """
    return Timer()


@contextmanager
def time_block() -> Iterator[Dict[str, float]]:
    """计时上下文，出口写入 ``{"elapsed": 秒}``。

    Yields:
        可变字典，键 ``elapsed`` 在退出时被赋值。
    """
    holder: Dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder["elapsed"] = time.perf_counter() - start


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在（不存在则递归创建）。

    Args:
        path: 目标目录。

    Returns:
        已存在的目录路径。
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def stable_json(obj: Any, indent: int = 2, ensure_ascii: bool = False) -> str:
    """确定性 JSON 序列化（``sort_keys=True``，保证同输入同输出）。

    Args:
        obj: 待序列化对象。
        indent: 缩进空格数。
        ensure_ascii: 是否转义非 ASCII（默认否，保留中文）。

    Returns:
        JSON 字符串。
    """
    return json.dumps(
        obj, indent=indent, ensure_ascii=ensure_ascii, sort_keys=True, default=str
    )


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> Path:
    """确定性落盘 JSON。

    Args:
        obj: 待序列化对象。
        path: 目标文件路径。
        indent: 缩进空格数。

    Returns:
        写入的文件路径。
    """
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(stable_json(obj, indent=indent))
        handle.write("\n")
    return target


def load_json(path: Union[str, Path]) -> Any:
    """读取 JSON 文件。

    Args:
        path: 文件路径。

    Returns:
        反序列化后的对象。
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_duration(seconds: float) -> str:
    """把秒格式化为 ``"1.23s"`` / ``"1m02.3s"``。

    Args:
        seconds: 耗时秒数。

    Returns:
        人类可读字符串。
    """
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:.2f}s"
    minutes = int(value // 60)
    return f"{minutes}m{value - minutes * 60:04.1f}s"


def clamp(value: float, low: float, high: float) -> float:
    """把数值裁剪到 ``[low, high]``。"""
    return float(min(max(value, low), high))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法：分母为 0 时返回 ``default``。"""
    if denominator == 0 or not np.isfinite(denominator):
        return float(default)
    return float(numerator) / float(denominator)


# ---------------------------------------------------------------- 稀疏邻接
def to_csr(adjacency: Any) -> "sp.csr_matrix":
    """把任意输入（scipy 稀疏 / 稠密 ndarray / 类数组）转为 ``csr_matrix``。

    Args:
        adjacency: 输入矩阵。

    Returns:
        ``scipy.sparse.csr_matrix``（float64）。
    """
    if sp.issparse(adjacency):
        return adjacency.tocsr().astype(np.float64)
    return sp.csr_matrix(np.asarray(adjacency, dtype=np.float64))


def symmetrize_adjacency(adjacency: Any, method: str = "max") -> "sp.csr_matrix":
    """把邻接矩阵对称化（无向图）。

    Args:
        adjacency: 输入邻接矩阵。
        method: ``"max"`` 取 ``max(A, A.T)``（默认，避免重复边权重翻倍）；
            ``"sum"`` 取 ``A + A.T``；``"any"`` 取布尔或（置 1）。

    Returns:
        对称的 ``csr_matrix``。

    Raises:
        ValueError: ``method`` 非法。
    """
    matrix = to_csr(adjacency)
    transposed = matrix.T.tocsr()
    if method == "max":
        return matrix.maximum(transposed).tocsr()
    if method == "sum":
        return (matrix + transposed).tocsr()
    if method == "any":
        return ((matrix + transposed) > 0).astype(np.float64).tocsr()
    raise ValueError(f"未知的对称化方法：{method!r}")


def remove_self_loops(adjacency: Any) -> "sp.csr_matrix":
    """去除邻接矩阵的对角线（自环）。

    Args:
        adjacency: 输入邻接矩阵。

    Returns:
        无自环的 ``csr_matrix``。
    """
    matrix = to_csr(adjacency)
    matrix = matrix - sp.diags(matrix.diagonal())
    matrix.eliminate_zeros()
    return matrix.tocsr()


def degree_vector(adjacency: Any) -> np.ndarray:
    """返回邻接矩阵的度向量（有向图为出度）。

    Args:
        adjacency: 输入邻接矩阵。

    Returns:
        ``(n,)`` 的 float 数组。
    """
    return np.asarray(to_csr(adjacency).sum(axis=1)).ravel().astype(np.float64)


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行做 L2 归一化（零行保持为零）。

    Args:
        matrix: ``(n, d)`` 输入。
        eps: 数值稳定项。

    Returns:
        行归一化后的新矩阵。
    """
    array = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return array / norms
