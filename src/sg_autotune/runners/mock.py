from __future__ import annotations

import math
import random
import time

from sg_autotune.models import BenchmarkResult, ProbeResult, TuneConfig
from sg_autotune.runners.base import Runner, RunnerCapabilities
from sg_autotune.search_space import DEFAULT_SPACE
from sg_autotune.scoring import recompute_result_score


class MockRunner(Runner):
    """Deterministic local surface for fast end-to-end optimizer tests."""

    def __init__(self, seed: int = 11):
        self.rng = random.Random(seed)

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            name="mock",
            applied_params=tuple(spec.name for spec in DEFAULT_SPACE),
            notes="Deterministic simulated backend for CI and optimizer development.",
        )

    def benchmark(self, config: TuneConfig, *, profile: str) -> BenchmarkResult:
        started = time.perf_counter()
        time.sleep(0.01)

        ctx_quality = _closeness(math.log2(config.ctx_size), math.log2(65536), 1.7)
        batch_quality = _closeness(config.batch_size, 2048, 1024)
        ubatch_quality = _closeness(config.ubatch_size, 512, 256)
        mtp_bonus = 0.12 if config.mtp_enabled and config.mtp_draft_n in {2, 3} else -0.03
        kv_quality = {"f16": 0.98, "q8_0": 0.94, "q4_0": 0.74}[config.kv_cache]
        temp_quality = _closeness(config.temperature, 0.55, 0.35)

        reliability = max(
            0.0,
            min(
                1.0,
                0.18
                + 0.22 * ctx_quality
                + 0.18 * kv_quality
                + 0.15 * temp_quality
                + 0.12 * ubatch_quality
                + mtp_bonus,
            ),
        )
        if config.parallel > 2 and config.ctx_size >= 98304:
            reliability -= 0.18
        reliability = max(0.0, min(1.0, reliability))

        tokens_per_second = (
            18
            + 26 * batch_quality
            + 18 * ubatch_quality
            + (24 if config.mtp_enabled else 0)
            + (8 if config.flash_attn else -5)
            - 0.00008 * max(config.ctx_size - 32768, 0)
        )
        if config.mtp_enabled:
            tokens_per_second -= abs(config.mtp_draft_n - 2) * 5
        tokens_per_second = max(1.0, tokens_per_second)

        ttft_s = 0.28 + (config.ctx_size / 131072) * 1.9 + (0.3 if config.parallel > 1 else 0.0)
        memory_pressure = min(
            1.0,
            0.18
            + config.ctx_size / 180000
            + config.parallel * 0.06
            + {"f16": 0.12, "q8_0": 0.06, "q4_0": 0.02}[config.kv_cache],
        )
        failed = memory_pressure > 0.93 or (config.kv_cache == "q4_0" and config.ctx_size > 98304)

        probe_names = ["json", "tool-call", "code-edit", "long-context", "stability"]
        probes = []
        for index, name in enumerate(probe_names):
            probe_score = max(0.0, min(1.0, reliability + self.rng.uniform(-0.04, 0.04)))
            if name == "long-context":
                probe_score *= ctx_quality
            if name == "json" and config.temperature > 0.8:
                probe_score -= 0.1
            probes.append(
                ProbeResult(
                    name=name,
                    passed=probe_score >= 0.72 and not failed,
                    score=max(0.0, min(1.0, probe_score if not failed else probe_score * 0.4)),
                    latency_s=ttft_s + index * 0.03,
                    tokens_per_second=tokens_per_second,
                )
            )

        result = BenchmarkResult(
            config=config,
            score=0,
            quality_score=0,
            tokens_per_second=round(tokens_per_second, 3),
            ttft_s=round(ttft_s, 3),
            peak_memory_mb=round(24576 * memory_pressure, 1),
            memory_pressure=round(memory_pressure, 4),
            failed=failed,
            error="simulated memory pressure failure" if failed else None,
            probes=probes,
            duration_s=round(time.perf_counter() - started, 4),
        )
        return recompute_result_score(result, profile)


def _closeness(value: float, target: float, scale: float) -> float:
    return math.exp(-((value - target) ** 2) / (2 * scale * scale))
