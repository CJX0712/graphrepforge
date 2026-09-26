"""报告渲染：固定宽度表格（列宽 12）+ Markdown + JSON。

列宽固定的原因：中文/数字混排时用可变宽度会错位（架构 §10 坑 4）。

作者：晨星
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from graphforge.core.errors import SerializationError
from graphforge.core.types import BenchmarkResult, EvalResult
from graphforge.core.utils import stable_json

__all__ = ["COLUMN_WIDTH", "render_table", "render_markdown", "BenchmarkReport"]

#: 固定列宽。
COLUMN_WIDTH: int = 12

#: 默认表头。
DEFAULT_HEADERS: tuple = (
    "dataset",
    "method",
    "backend",
    "task",
    "metric",
    "value",
    "elapsed",
)


def _cell(value: Any) -> str:
    """把任意值渲染为定宽单元格（超长截断，保证各行等宽）。"""
    text = str(value)
    return f"{text[:COLUMN_WIDTH]:<{COLUMN_WIDTH}}"


def render_table(
    rows: Sequence[EvalResult],
    headers: Sequence[str] = DEFAULT_HEADERS,
    metric: Optional[str] = None,
) -> str:
    """渲染定宽文本表格。

    Args:
        rows: 结果行。
        headers: 表头。
        metric: 展示的指标列；None 时取各行主指标。

    Returns:
        多行字符串（首行为表头，次行为分隔线）。
    """
    lines: List[str] = ["".join(_cell(header) for header in headers)]
    lines.append("-" * COLUMN_WIDTH * len(headers))
    for row in rows:
        name = metric or row.primary_metric
        value = row.metrics.get(name, row.primary_value)
        cells = (
            row.dataset,
            row.method,
            row.backend,
            row.task.value if hasattr(row.task, "value") else str(row.task),
            name,
            f"{float(value):.4f}",
            f"{float(row.elapsed_sec):.3f}",
        )
        lines.append("".join(_cell(cell) for cell in cells))
    return "\n".join(lines)


def render_markdown(result: BenchmarkResult, metric: Optional[str] = None) -> str:
    """渲染 Markdown 表格。

    Args:
        result: 基准结果。
        metric: 展示的指标列；None 时取主指标。

    Returns:
        Markdown 文本。
    """
    headers = ("dataset", "method", "backend", "task", "metric", "value", "elapsed")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in result.rows:
        name = metric or row.primary_metric
        value = row.metrics.get(name, row.primary_value)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.dataset),
                    str(row.method),
                    str(row.backend),
                    row.task.value if hasattr(row.task, "value") else str(row.task),
                    str(name),
                    f"{float(value):.4f}",
                    f"{float(row.elapsed_sec):.3f}",
                ]
            )
            + " |"
        )
    if result.skipped:
        lines.append("")
        lines.append("被跳过的方法：")
        for item in result.skipped:
            lines.append(f"- {item.get('method', 'unknown')}: {item.get('reason', '')}")
    return "\n".join(lines)


class BenchmarkReport:
    """基准报告：文本 / Markdown / JSON 三种渲染 + 落盘。

    Attributes:
        result: 被渲染的 :class:`BenchmarkResult`。
    """

    def __init__(self, result: BenchmarkResult) -> None:
        """构造报告。

        Args:
            result: 基准结果。
        """
        self.result = result

    def to_table(self, metric: Optional[str] = None) -> str:
        """定宽文本表格。"""
        return render_table(self.result.rows, metric=metric)

    def to_markdown(self, metric: Optional[str] = None) -> str:
        """Markdown 表格。"""
        return render_markdown(self.result, metric=metric)

    def to_json(self) -> str:
        """确定性 JSON 字符串。"""
        return stable_json(self.result.to_dict())

    def save_json(self, path: Union[str, Path]) -> Path:
        """落盘 JSON。

        Args:
            path: 目标文件路径。

        Returns:
            写入路径。

        Raises:
            SerializationError: 写入失败（E505）。
        """
        target = Path(path)
        try:
            if target.parent and not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.to_json() + "\n", encoding="utf-8")
        except OSError as exc:
            raise SerializationError(
                f"写入 {target} 失败：{exc}", hint="检查目录权限"
            ) from exc
        return target

    def save_markdown(self, path: Union[str, Path], metric: Optional[str] = None) -> Path:
        """落盘 Markdown。

        Args:
            path: 目标文件路径。
            metric: 展示的指标列。

        Returns:
            写入路径。

        Raises:
            SerializationError: 写入失败（E505）。
        """
        target = Path(path)
        try:
            if target.parent and not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.to_markdown(metric=metric) + "\n", encoding="utf-8")
        except OSError as exc:
            raise SerializationError(
                f"写入 {target} 失败：{exc}", hint="检查目录权限"
            ) from exc
        return target

    def render(self, result: BenchmarkResult) -> str:
        """满足 :class:`~graphforge.core.interfaces.Reporter` 协议。"""
        self.result = result
        return self.to_table()

    def summary(self) -> Dict[str, Any]:
        """返回汇总信息（行数、最优行、跳过数）。"""
        best: Optional[EvalResult] = None
        if self.result.rows:
            best = self.result.best()
        return {
            "name": self.result.name,
            "created_at": self.result.created_at,
            "n_rows": len(self.result.rows),
            "n_skipped": len(self.result.skipped),
            "best": None if best is None else best.as_row(),
        }
