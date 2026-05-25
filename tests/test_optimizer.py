from __future__ import annotations

from pathlib import Path

from sg_autotune.runners.mock import MockRunner
from sg_autotune.study import load_results, run_study


def test_bayesian_study_produces_records(tmp_path: Path) -> None:
    out = tmp_path / "study.jsonl"
    history = run_study(
        runner_kind="mock",
        runner=MockRunner(seed=1),
        budget_s=10,
        out_path=out,
        max_iterations=12,
    )

    records = load_results(out)
    assert len(history) == 12
    assert len(records) == 12
    assert records[0].result.config.model_dump() == records[0].result.config.__class__().model_dump()
    assert max(record.result.score for record in records) > 0.25
