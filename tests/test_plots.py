from __future__ import annotations

from pathlib import Path

from sg_autotune.plots import render_optimization_plot
from sg_autotune.runners.mock import MockRunner
from sg_autotune.study import run_study


def test_render_optimization_plot(tmp_path: Path) -> None:
    records = tmp_path / "study.jsonl"
    image = tmp_path / "plot.png"
    run_study(
        runner_kind="mock",
        runner=MockRunner(seed=7),
        budget_s=10,
        out_path=records,
        max_iterations=6,
    )

    render_optimization_plot([records], image, labels=["mock"])

    assert image.exists()
    assert image.stat().st_size > 1000
