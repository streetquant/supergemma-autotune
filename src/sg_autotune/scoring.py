from __future__ import annotations

from dataclasses import dataclass

from sg_autotune.models import BenchmarkResult, ProbeResult


@dataclass(frozen=True)
class ScoreWeights:
    quality: float = 0.58
    speed: float = 0.24
    latency: float = 0.10
    memory: float = 0.08
    failure_penalty: float = 0.55


PROFILE_WEIGHTS: dict[str, ScoreWeights] = {
    "coding-agent": ScoreWeights(),
    "throughput": ScoreWeights(quality=0.35, speed=0.45, latency=0.08, memory=0.12),
    "json-tools": ScoreWeights(quality=0.72, speed=0.14, latency=0.08, memory=0.06),
}


def quality_from_probes(probes: list[ProbeResult]) -> float:
    if not probes:
        return 0.0
    return sum(probe.score for probe in probes) / len(probes)


def combined_score(
    *,
    quality_score: float,
    tokens_per_second: float,
    ttft_s: float,
    memory_pressure: float,
    failed: bool,
    profile: str,
) -> float:
    weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["coding-agent"])
    speed_norm = min(tokens_per_second / 80.0, 1.0)
    latency_norm = min(ttft_s / 6.0, 1.0)
    score = (
        weights.quality * quality_score
        + weights.speed * speed_norm
        - weights.latency * latency_norm
        - weights.memory * memory_pressure
    )
    if failed:
        score -= weights.failure_penalty
    return round(max(score, -1.0), 6)


def recompute_result_score(result: BenchmarkResult, profile: str) -> BenchmarkResult:
    result.quality_score = quality_from_probes(result.probes)
    result.score = combined_score(
        quality_score=result.quality_score,
        tokens_per_second=result.tokens_per_second,
        ttft_s=result.ttft_s,
        memory_pressure=result.memory_pressure,
        failed=result.failed,
        profile=profile,
    )
    return result

