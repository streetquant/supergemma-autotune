from __future__ import annotations

import json
from pathlib import Path

from sg_autotune.metadata import metadata_path
from sg_autotune.runners.mock import MockRunner
from sg_autotune.study import run_study


def test_run_writes_study_metadata(tmp_path: Path) -> None:
    out = tmp_path / "study.jsonl"
    run_study(
        runner_kind="mock",
        runner=MockRunner(seed=1),
        budget_s=10,
        out_path=out,
        max_iterations=1,
        resume=False,
        seed=123,
    )

    meta = json.loads(metadata_path(out).read_text(encoding="utf-8"))

    assert meta["schema"] == "sg-autotune-study-metadata-v1"
    assert meta["runner"] == "mock"
    assert meta["seed"] == 123
    assert "hardware" in meta
