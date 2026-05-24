from __future__ import annotations

from pathlib import Path
import shlex

from sg_autotune.models import BenchmarkResult, Recommendation
from sg_autotune.study import best_eligible_result, best_result, load_results


def build_recommendation(
    records_path: Path,
    *,
    model_path: str = "MODEL.gguf",
) -> Recommendation:
    records = load_results(records_path)
    best = best_eligible_result(records)
    used_ineligible = False
    if best is None:
        best = best_result(records)
        used_ineligible = True
    runner = records[0].runner if records else "mock"
    warnings = _warnings(best)
    if used_ineligible:
        warnings.insert(0, "No eligible result met the quality/failure gate; showing highest scalar score only.")
    command = _recommendation_command(best, runner=runner, model_path=model_path)
    summary = (
        f"Best score {best.score:.3f}; quality {best.quality_score:.2f}; "
        f"{best.tokens_per_second:.1f} tok/s; TTFT {best.ttft_s:.2f}s."
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
    lines = [
        "# SuperGemma AutoTune Report",
        "",
        f"Iterations: {len(records)}",
        "",
        "## Best Eligible Recommendation",
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
