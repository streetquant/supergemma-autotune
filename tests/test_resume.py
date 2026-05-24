from __future__ import annotations

from pathlib import Path

from sg_autotune.runners.mock import MockRunner
from sg_autotune.study import load_results, run_study


def test_study_resumes_existing_ledger(tmp_path: Path) -> None:
    out = tmp_path / "study.jsonl"
    run_study(
        runner_kind="mock",
        runner=MockRunner(seed=1),
        budget_s=10,
        out_path=out,
        max_iterations=3,
        resume=False,
    )
    run_study(
        runner_kind="mock",
        runner=MockRunner(seed=1),
        budget_s=10,
        out_path=out,
        max_iterations=2,
        resume=True,
    )

    records = load_results(out)

    assert [record.iteration for record in records] == [0, 1, 2, 3, 4]


def test_study_fresh_overwrites_existing_ledger(tmp_path: Path) -> None:
    out = tmp_path / "study.jsonl"
    run_study(
        runner_kind="mock",
        runner=MockRunner(seed=1),
        budget_s=10,
        out_path=out,
        max_iterations=4,
        resume=False,
    )
    run_study(
        runner_kind="mock",
        runner=MockRunner(seed=1),
        budget_s=10,
        out_path=out,
        max_iterations=2,
        resume=False,
    )

    records = load_results(out)

    assert [record.iteration for record in records] == [0, 1]
