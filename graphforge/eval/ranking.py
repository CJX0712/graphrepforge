"""结果排名：统一按 ``direction`` 归一后降序排列（越大越好）。

作者：晨星
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from graphforge.core.errors import MissingMetricError
from graphforge.core.types import EvalResult

__all__ = ["rank_results", "sort_by_metric"]


def sort_by_metric(
    rows: Sequence[EvalResult], metric: Optional[str] = None
) -> List[EvalResult]:
    """按指定指标降序排序（全部指标方向为 +1，无需再归一方向）。

    Args:
        rows: 结果行。
        metric: 指标名；None 时取各行主指标（多行主指标必须一致）。

    Returns:
        排序后的新列表。

    Raises:
        MissingMetricError: 行中缺少该指标 / 主指标不一致。
    """
    if not rows:
        return []
    name = metric or rows[0].primary_metric
    if metric is None:
        names = {row.primary_metric for row in rows}
        if len(names) != 1:
            raise MissingMetricError(
                f"主指标不一致：{sorted(names)}", hint="显式传入 metric"
            )
    missing = [row for row in rows if name not in row.metrics]
    if missing:
        raise MissingMetricError(
            f"{len(missing)} 行缺少指标 {name!r}", hint="检查 scoring 配置"
        )
    return sorted(rows, key=lambda row: float(row.metrics[name]), reverse=True)


def rank_results(
    rows: Sequence[EvalResult], metric: Optional[str] = None
) -> List[dict]:
    """排序并附带 ``rank`` 字段（1 为最优）。

    Args:
        rows: 结果行。
        metric: 指标名；None 时取主指标。

    Returns:
        形如 ``[{"rank": 1, "row": EvalResult}]`` 的列表。
    """
    ordered = sort_by_metric(rows, metric)
    return [{"rank": index + 1, "row": row} for index, row in enumerate(ordered)]
