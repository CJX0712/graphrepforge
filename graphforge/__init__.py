"""GraphForge —— 图嵌入与图机器学习评测基准。

本模块为包顶层，只做两件事：

1. 暴露版本号与常用类型；
2. 暴露 :func:`available_backends`（后端探测，**永不抛异常**）。

刻意不在顶层 import 重模块（``graph`` / ``pipeline`` 等），避免 import 期的
可选依赖失败拖垮整个包。

作者：晨星
"""

from __future__ import annotations

from typing import Dict, List

__version__: str = "0.1.0"
__author__: str = "晨星"
__all__: List[str] = [
    "__version__",
    "__author__",
    "available_backends",
    "TaskType",
    "GraphData",
    "Embedding",
    "TaskSpec",
    "Split",
    "EvalResult",
    "BenchmarkResult",
    "GraphForgeError",
    "get_config",
]

from graphforge.core.errors import GraphForgeError  # noqa: E402
from graphforge.core.types import (  # noqa: E402
    BenchmarkResult,
    Embedding,
    EvalResult,
    GraphData,
    Split,
    TaskSpec,
    TaskType,
)
from graphforge.core.config import get_config  # noqa: E402


def available_backends() -> Dict[str, bool]:
    """探测图嵌入后端可用性。

    该函数**保证不抛异常**：任何探测失败都退化为 ``{"karateclub": False, "internal": True}``。

    Returns:
        ``{"karateclub": bool, "internal": bool}`` 形式的字典。``internal`` 恒为 True。
    """
    try:
        from graphforge.graph.embeddings.backends import (  # 延迟导入，避免顶层依赖
            available_backends as _probe_backends,
        )
    except Exception:  # pragma: no cover - 仅在 graph 层不可用时触发
        return {"karateclub": False, "internal": True}
    try:
        return _probe_backends()
    except Exception:  # pragma: no cover - 探测函数内部已自保，此处为双保险
        return {"karateclub": False, "internal": True}
