from __future__ import annotations

from pathlib import Path
import shlex

from sg_autotune.constraints import ConstraintPolicy, auto_constraint_policy
from sg_autotune.models import BenchmarkResult, Recommendation
from sg_autotune.study import best_eligible_result, best_result, load_results


def build_recommendation(
    records_path: Path,
    *,
    model_path: str = "MODEL.gguf",
    hardware_policy: ConstraintPolicy | None = None,
) -> Recommendation:
    records = load_results(records_path)
    best = best_eligible_result(records)
    used_ineligible = False
    if best is None:
        best = best_result(records)
        used_ineligible = True
    runner = records[0].runner if records else "mock"
    if runner == "llamacpp":
        policy = hardware_policy or auto_constraint_policy(model_path)
        best = best.model_copy(update={"config": policy.apply_hardware_defaults(best.config)})
    warnings = _warnings(best)
    if used_ineligible:
        warnings.insert(0, "No eligible result met the quality/failure gate; showing highest scalar score only.")
    command = _recommendation_command(best, runner=runner, model_path=model_path)
    summary = (
        f"Best score {best.score:.3f}; quality {best.quality_score:.2f}; "
        f"{best.tokens_per_second:.1f} tok/s; fastest request latency {best.ttft_s:.2f}s."
    )
    return Recommendation(best=best, command=command, summary=summary, warnings=warnings)


def _recommendation_command(best: BenchmarkResult, *, runner: str, model_path: str) -> str:
    if runner == "openai":
        return (
            "OpenAI-compatible request settings: "
            f"temperature={best.config.temperature:.3f}, "
            f"top_p={best.config.top_p:.3f}, "
            f"top_k={best.config.top_k}. "
            "Runtime/server flags were not tuned by this runner."
        )
    return shlex.join(best.config.llama_cpp_args(model_path))


def render_markdown(records_path: Path, *, model_path: str = "MODEL.gguf") -> str:
    records = load_results(records_path)
    recommendation = build_recommendation(records_path, model_path=model_path)
    best = recommendation.best
    recommendation_heading = (
        "## Highest Scalar Score"
        if recommendation.warnings and recommendation.warnings[0].startswith("No eligible result")
        else "## Best Eligible Recommendation"
    )
    lines = [
        "# SuperGemma AutoTune Report",
        "",
        f"Iterations: {len(records)}",
        "",
        recommendation_heading,
        "",
        recommendation.summary,
        "",
        "```bash",
        recommendation.command,
        "```",
        "",
        "## Best Config",
        "",
        "```json",
        best.config.model_dump_json(indent=2),
        "```",
        "",
    ]
    hardware_lines = _hardware_lines(best)
    if hardware_lines:
        lines += ["## Hardware-Aware Runner Flags", ""]
        lines += hardware_lines
        lines += [""]
    lines += [
        "## Probe Results",
        "",
        "| Probe | Pass | Score | Latency | tok/s | Error |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for probe in best.probes:
        lines.append(
            f"| {probe.name} | {probe.passed} | {probe.score:.2f} | "
            f"{probe.latency_s:.2f}s | {probe.tokens_per_second:.1f} | {probe.error or ''} |"
        )
    if recommendation.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in recommendation.warnings]
    return "\n".join(lines) + "\n"


def write_report(records_path: Path, out_path: Path, *, model_path: str = "MODEL.gguf") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(records_path, model_path=model_path), encoding="utf-8")


def _warnings(best: BenchmarkResult) -> list[str]:
    warnings = []
    if best.config.kv_cache == "q4_0":
        warnings.append("q4_0 KV cache is fast and memory-efficient but can degrade reliability.")
    if best.memory_pressure > 0.85:
        warnings.append("Best config is close to memory limits; keep a safer fallback config.")
    if best.config.mtp_enabled and best.config.mtp_draft_n > 3:
        warnings.append("High MTP draft depth can regress quality on some MoE models.")
    if best.failed:
        warnings.append("Best-scoring run was marked failed; inspect the raw JSONL before using it.")
    return warnings


def _hardware_lines(best: BenchmarkResult) -> list[str]:
    config = best.config
    rows = []
    if config.device:
        rows.append(f"- Devices: `{config.device}`")
    if config.split_mode:
        rows.append(f"- Multi-GPU split mode: `{config.split_mode}`")
    if config.tensor_split:
        rows.append(f"- Tensor split weights: `{config.tensor_split}`")
    if config.main_gpu is not None:
        rows.append(f"- Main GPU: `{config.main_gpu}`")
    if config.threads is not None:
        rows.append(f"- Generation CPU threads: `{config.threads}`")
    if config.threads_batch is not None:
        rows.append(f"- Batch CPU threads: `{config.threads_batch}`")
    if config.numa:
        rows.append(f"- NUMA policy: `{config.numa}`")
    if config.fit_target:
        rows.append(f"- llama.cpp fit target: `{config.fit_target}` MiB per active GPU")
    return rows
