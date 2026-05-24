from __future__ import annotations

from pathlib import Path

from sg_autotune.report import render_markdown
from sg_autotune.runners.mock import MockRunner
from sg_autotune.study import run_study


def test_report_contains_recommendation(tmp_path: Path) -> None:
    out = tmp_path / "study.jsonl"
    run_study(
        runner_kind="mock",
        runner=MockRunner(seed=2),
        budget_s=10,
        out_path=out,
        max_iterations=8,
    )

    markdown = render_markdown(out, model_path="supergemma.gguf")

    assert "SuperGemma AutoTune Report" in markdown
    assert "Best Eligible Recommendation" in markdown
    assert "llama-server" in markdown
    assert "supergemma.gguf" in markdown
