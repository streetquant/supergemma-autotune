from __future__ import annotations

from pathlib import Path

import pytest

from sg_autotune.exporters import export_from_records
from sg_autotune.models import BenchmarkResult, ProbeResult, StudyRecord, TuneConfig
from sg_autotune.study import is_recommendation_eligible


def test_failed_result_is_not_recommendation_eligible() -> None:
    result = BenchmarkResult(
        config=TuneConfig(),
        score=1.0,
        quality_score=1.0,
        tokens_per_second=100,
        ttft_s=0.1,
        peak_memory_mb=1,
        memory_pressure=0.1,
        failed=True,
        probes=[ProbeResult(name="x", passed=False, score=0, latency_s=0, tokens_per_second=0)],
    )

    assert not is_recommendation_eligible(result)


def test_export_refuses_ineligible_only_ledger(tmp_path: Path) -> None:
    result = BenchmarkResult(
        config=TuneConfig(),
        score=1.0,
        quality_score=0.1,
        tokens_per_second=100,
        ttft_s=0.1,
        peak_memory_mb=1,
        memory_pressure=0.1,
        failed=False,
        probes=[ProbeResult(name="x", passed=False, score=0.1, latency_s=0, tokens_per_second=0)],
    )
    path = tmp_path / "study.jsonl"
    path.write_text(
        StudyRecord(iteration=0, runner="mock", profile="coding-agent", result=result).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        export_from_records(path, target="llamacpp")
