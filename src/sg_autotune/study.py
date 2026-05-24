from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TextIO

from sg_autotune.constraints import ConstraintPolicy, constraint_failure_result
from sg_autotune.models import BenchmarkResult, RunnerKind, StudyRecord
from sg_autotune.optimizer import BayesianOptimizer
from sg_autotune.runners import LlamaCppManagedRunner, MockRunner, OpenAICompatibleRunner, Runner


def build_runner(
    runner: RunnerKind,
    *,
    base_url: str | None = None,
    model: str | None = None,
    model_path: str | None = None,
    llama_server: str = "llama-server",
    port: int = 0,
) -> Runner:
    if runner == "mock":
        return MockRunner()
    if runner == "openai":
        if not base_url or not model:
            raise ValueError("--base-url and --model are required for --runner openai")
        return OpenAICompatibleRunner(base_url=base_url, model=model)
    if runner == "llamacpp":
        if not model_path:
            raise ValueError("--model-path is required for --runner llamacpp")
        return LlamaCppManagedRunner(model_path=model_path, binary=llama_server, port=port)
    raise ValueError(f"Unsupported runner: {runner}")


def run_study(
    *,
    runner_kind: RunnerKind,
    runner: Runner,
    budget_s: float,
    out_path: Path,
    profile: str = "coding-agent",
    seed: int = 7,
    max_iterations: int | None = None,
    constraint_policy: ConstraintPolicy | None = None,
    resume: bool = True,
) -> list[BenchmarkResult]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = BayesianOptimizer(seed=seed)
    records = load_results(out_path) if resume and out_path.exists() else []
    history: list[BenchmarkResult] = [record.result for record in records]
    optimizer.observe_existing(history)
    deadline = time.monotonic() + budget_s
    iteration = (max((record.iteration for record in records), default=-1) + 1) if resume else 0
    mode = "a" if resume else "w"
    with out_path.open(mode, encoding="utf-8") as fh:
        while time.monotonic() < deadline:
            new_iterations = iteration - (records[-1].iteration + 1 if records else 0)
            if max_iterations is not None and new_iterations >= max_iterations:
                break
            config = optimizer.suggest(history)
            rejection = constraint_policy.explain_rejection(config) if constraint_policy else None
            if rejection:
                result = constraint_failure_result(config, profile, rejection)
            else:
                result = runner.benchmark(config, profile=profile)
            history.append(result)
            record = StudyRecord(
                iteration=iteration,
                runner=runner_kind,
                profile=profile,
                result=result,
            )
            _write_record(fh, record)
            iteration += 1
    return history


def load_results(path: Path) -> list[StudyRecord]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(StudyRecord.model_validate_json(line))
    return records


def _write_record(fh: TextIO, record: StudyRecord) -> None:
    fh.write(record.model_dump_json() + "\n")
    fh.flush()


def best_result(records: list[StudyRecord]) -> BenchmarkResult:
    if not records:
        raise ValueError("No study records found")
    return max((record.result for record in records), key=lambda item: item.score)


def json_summary(path: Path) -> str:
    records = load_results(path)
    best = best_result(records)
    return json.dumps(
        {
            "iterations": len(records),
            "best_score": best.score,
            "best_config": best.config.model_dump(),
            "quality_score": best.quality_score,
            "tokens_per_second": best.tokens_per_second,
        },
        indent=2,
        sort_keys=True,
    )
