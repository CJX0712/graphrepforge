"""CLI 单测：子命令退出码与输出。"""

from __future__ import annotations

from pathlib import Path

from graphforge.cli import main


def test_doctor_exits_zero(capsys: object) -> None:
    """doctor 退出码 0 且包含后端状态。"""
    code = main(["--log-level", "ERROR", "doctor"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert code == 0
    assert "internal" in captured.out


def test_list_datasets_exits_zero(capsys: object) -> None:
    """list-datasets 覆盖全部注册名。"""
    from graphforge.data.registry import list_datasets

    code = main(["--log-level", "ERROR", "list-datasets"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert code == 0
    for name in list_datasets():
        assert name in captured.out


def test_list_methods_kinds(capsys: object) -> None:
    """list-methods 四种 kind 均退出 0。"""
    for kind in ("embedders", "tasks", "metrics", "estimators"):
        code = main(["--log-level", "ERROR", "list-methods", "--kind", kind])
        assert code == 0


def test_embed_and_run(tmp_path: Path, capsys: object) -> None:
    """embed 与 run 输出摘要并落盘 JSON。"""
    summary_path = tmp_path / "embed.json"
    code = main(
        [
            "--log-level", "ERROR", "embed",
            "--dataset", "karate", "--method", "spectral", "--dim", "8", "--seed", "42",
            "--output", str(summary_path),
        ]
    )
    assert code == 0 and summary_path.exists()

    result_path = tmp_path / "run.json"
    code = main(
        [
            "--log-level", "ERROR", "run",
            "--dataset", "karate", "--task", "node_classification",
            "--method", "spectral", "--dim", "8", "--seed", "42",
            "--output", str(result_path),
        ]
    )
    assert code == 0 and result_path.exists()
    assert '"primary_metric"' in result_path.read_text(encoding="utf-8")


def test_benchmark_writes_artifacts(tmp_path: Path) -> None:
    """benchmark 落盘 JSON + Markdown。"""
    json_path = tmp_path / "bench.json"
    md_path = tmp_path / "bench.md"
    code = main(
        [
            "--log-level", "ERROR", "benchmark",
            "--datasets", "karate", "--methods", "spectral",
            "--tasks", "node_classification", "--dim", "8", "--seed", "42",
            "--output", str(json_path), "--report", str(md_path),
        ]
    )
    assert code == 0
    assert json_path.exists() and md_path.exists()
    assert '"rows"' in json_path.read_text(encoding="utf-8")
