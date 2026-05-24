from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sg_autotune.hardware import scan_hardware
from sg_autotune.models import BenchmarkResult, ProbeResult, TuneConfig
from sg_autotune.scoring import recompute_result_score


@dataclass(frozen=True)
class ConstraintPolicy:
    model_size_mb: float = 0.0
    vram_limit_mb: float = 0.0
    system_memory_limit_mb: float = 0.0
    reserve_vram_mb: float = 1536.0
    enabled: bool = True

    def explain_rejection(self, config: TuneConfig) -> str | None:
        if not self.enabled:
            return None
        estimate = estimate_memory_mb(config, model_size_mb=self.model_size_mb)
        if self.vram_limit_mb and estimate.vram_mb > max(0.0, self.vram_limit_mb - self.reserve_vram_mb):
            return (
                f"estimated VRAM {estimate.vram_mb:.0f} MB exceeds safe limit "
                f"{self.vram_limit_mb - self.reserve_vram_mb:.0f} MB"
            )
        if self.system_memory_limit_mb and estimate.host_mb > self.system_memory_limit_mb:
            return (
                f"estimated host memory {estimate.host_mb:.0f} MB exceeds limit "
                f"{self.system_memory_limit_mb:.0f} MB"
            )
        return None


@dataclass(frozen=True)
class MemoryEstimate:
    vram_mb: float
    host_mb: float


def estimate_memory_mb(config: TuneConfig, *, model_size_mb: float) -> MemoryEstimate:
    # Deliberately approximate: exact KV cache depends on architecture. This is
    # a safety filter, not a final allocator.
    kv_bytes = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.55}[config.kv_cache]
    kv_mb = (config.ctx_size / 1024.0) * config.parallel * kv_bytes * 42.0
    batch_mb = (config.batch_size + config.ubatch_size) * 0.12
    flash_mb = 384.0 if config.flash_attn else 128.0
    mtp_mb = 512.0 if config.mtp_enabled else 0.0
    model_on_gpu = model_size_mb * min(config.gpu_layers / 99.0, 1.0)
    model_on_host = max(0.0, model_size_mb - model_on_gpu)
    return MemoryEstimate(
        vram_mb=model_on_gpu + kv_mb + batch_mb + flash_mb + mtp_mb,
        host_mb=model_on_host + kv_mb * 0.2 + 1024.0,
    )


def auto_constraint_policy(model_path: str | None = None) -> ConstraintPolicy:
    hardware = scan_hardware()
    gpus = hardware.get("gpus") or []
    vram_limit_mb = float(gpus[0]["memory_free_mb"]) if gpus else 0.0
    system_memory_limit_mb = float(hardware.get("memory_available_gb") or 0.0) * 1024.0
    model_size_mb = 0.0
    if model_path and Path(model_path).exists():
        model_size_mb = Path(model_path).stat().st_size / (1024.0 * 1024.0)
    return ConstraintPolicy(
        model_size_mb=model_size_mb,
        vram_limit_mb=vram_limit_mb,
        system_memory_limit_mb=system_memory_limit_mb,
    )


def constraint_failure_result(config: TuneConfig, profile: str, reason: str) -> BenchmarkResult:
    result = BenchmarkResult(
        config=config,
        score=0,
        quality_score=0,
        tokens_per_second=0,
        ttft_s=0,
        peak_memory_mb=0,
        memory_pressure=1,
        failed=True,
        error=reason,
        probes=[
            ProbeResult(
                name="constraints",
                passed=False,
                score=0,
                latency_s=0,
                tokens_per_second=0,
                error=reason,
            )
        ],
    )
    return recompute_result_score(result, profile)

