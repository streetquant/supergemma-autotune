from __future__ import annotations

from pathlib import Path

import pytest

from sg_autotune.runners.mock import MockRunner
from sg_autotune.study import run_study


def test_resume_metadata_mismatch_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "study.jsonl"
    run_study(
        runner_kind="mock",
        runner=MockRunner(seed=1),
        budget_s=10,
        out_path=out,
        profile="coding-agent",
        seed=1,
        max_iterations=1,
    )

    with pytest.raises(ValueError, match="Resume metadata mismatch"):
        run_study(
            runner_kind="mock",
            runner=MockRunner(seed=1),
            budget_s=10,
            out_path=out,
            profile="latency",
            seed=1,
            max_iterations=1,
        )
