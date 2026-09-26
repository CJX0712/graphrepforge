"""GraphForge 端到端演示（一键复现，固定 seed=42）。

执行：
    python examples/run_demo.py

产出（相对项目根）：
    artifacts/benchmark.json   网格基准结果（同 seed 逐字节可复现）
    artifacts/report.md        Markdown 报告

作者：晨星
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 让脚本从仓库根直接运行（examples/ 的上一级就是项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphforge.core.config import reset_config  # noqa: E402
from graphforge.core.types import TaskType  # noqa: E402
from graphforge.eval.report import BenchmarkReport  # noqa: E402
from graphforge.pipeline.benchmark import Benchmark  # noqa: E402

SEED = 42
OUTPUT_DIR = PROJECT_ROOT / "artifacts"


def main() -> int:
    """执行端到端演示。"""
    from graphforge.data.synthetic import generate_sbm

    reset_config()
    started = time.perf_counter()

    # 预生成小规模 SBM（160 节点 / 4 社区），karate 直接按名字取
    sbm_demo = generate_sbm(n_blocks=4, block_size=40, p_in=0.15, p_out=0.02, random_state=SEED, name="sbm_demo")
    bench = Benchmark(
        datasets=(sbm_demo, "karate"),
        methods=("spectral", "deepwalk", "node2vec", "grarep"),
        tasks=(TaskType.NODE_CLASSIFICATION, TaskType.LINK_PREDICTION),
        dim=32,
        name="graphforge-demo",
        feature_method="degree",
    )

    result = bench.run()

    elapsed = time.perf_counter() - started
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "benchmark.json"
    md_path = OUTPUT_DIR / "report.md"
    result.to_json(json_path)

    report = BenchmarkReport(result)
    lines = [
        "# GraphForge Demo 报告",
        "",
        f"- 生成时间：{result.created_at}",
        f"- 随机种子：{SEED}（同 seed 逐字节可复现）",
        f"- 总耗时：{elapsed:.1f}s",
        f"- 成功组合：{len(result.rows)}；跳过：{len(result.skipped)}",
        "",
        "## 基准结果",
        "",
        "```",
        result.to_table(),
        "```",
        "",
        "## Markdown 表",
        "",
        report.to_markdown(),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(result.to_table())
    print(f"\n成功 {len(result.rows)} 组合 / 跳过 {len(result.skipped)}；耗时 {elapsed:.1f}s")
    print(f"产物：{json_path}")
    print(f"产物：{md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
