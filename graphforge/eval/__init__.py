"""GraphForge 评测层：指标注册表（方向恒 +1）/ 排名 / 报告。

作者：晨星
"""

from __future__ import annotations

from graphforge.eval import metrics as metrics
from graphforge.eval import ranking as ranking
from graphforge.eval import report as report

from graphforge.eval.metrics import (
    METRIC_REGISTRY,
    Scorer,
    accuracy_score,
    average_precision_score,
    check_directions,
    compute_metrics,
    default_scoring,
    get_metric,
    list_metrics,
    macro_f1_score,
    micro_f1_score,
    roc_auc_score,
    weighted_f1_score,
)
from graphforge.eval.ranking import rank_results
from graphforge.eval.report import BenchmarkReport, render_table

__all__ = [
    "metrics",
    "ranking",
    "report",
    "Scorer",
    "METRIC_REGISTRY",
    "accuracy_score",
    "macro_f1_score",
    "micro_f1_score",
    "weighted_f1_score",
    "roc_auc_score",
    "average_precision_score",
    "list_metrics",
    "get_metric",
    "compute_metrics",
    "default_scoring",
    "check_directions",
    "rank_results",
    "render_table",
    "BenchmarkReport",
]
