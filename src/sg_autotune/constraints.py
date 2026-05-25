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
    gpu_free_mb: tuple[float, ...] = ()
    active_gpu_indices: tuple[int, ...] = ()
    system_memory_limit_mb: float = 0.0
    reserve_vram_mb: float = 1536.0
    reserve_system_memory_mb: float = 2048.0
    device: str | None = None
    split_mode: str | None = None
    tensor_split: str | None = None
    main_gpu: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    numa: str | None = None
    fit_target: str | None = None
    enabled: bool = True

    def apply_hardware_defaults(self, config: TuneConfig) -> TuneConfig:
        updates = {}
        for field in (
            "device",
            "split_mode",
            "tensor_split",
            "main_gpu",
            "threads",
            "threads_batch",
            "numa",
            "fit_target",
        ):
            if getattr(config, field) is None and getattr(self, field) is not None:
                updates[field] = getattr(self, field)
        if not updates:
            return config
        return config.model_copy(update=updates)

    def explain_rejection(self, config: TuneConfig) -> str | None:
        if not self.enabled:
            return None
        estimate = estimate_memory_mb(config, model_size_mb=self.model_size_mb)
        if self.gpu_free_mb:
            per_gpu = estimate_per_gpu_vram_mb(
                config,
                estimate,
                gpu_free_mb=self.gpu_free_mb,
                active_gpu_indices=self.active_gpu_indices,
            )
            for position, estimated_mb in enumerate(per_gpu):
                free_mb = self.gpu_free_mb[position]
                safe_mb = max(0.0, free_mb - self.reserve_vram_mb)
                gpu_index = self.active_gpu_indices[position] if self.active_gpu_indices else position
                if estimated_mb > safe_mb:
                    return (
                        f"estimated GPU {gpu_index} VRAM {estimated_mb:.0f} MB exceeds safe limit "
                        f"{safe_mb:.0f} MB"
                    )
        if self.vram_limit_mb and estimate.vram_mb > max(0.0, self.vram_limit_mb - self.reserve_vram_mb):
            return (
                f"estimated VRAM {estimate.vram_mb:.0f} MB exceeds safe limit "
                f"{self.vram_limit_mb - self.reserve_vram_mb:.0f} MB"
            )
        safe_system_mb = max(0.0, self.system_memory_limit_mb - self.reserve_system_memory_mb)
        if self.system_memory_limit_mb and estimate.host_mb > safe_system_mb:
            return (
                f"estimated host memory {estimate.host_mb:.0f} MB exceeds limit "
                f"{safe_system_mb:.0f} MB"
            )
        return None


@dataclass(frozen=True)
class MemoryEstimate:
    vram_mb: float
    host_mb: float
    model_gpu_mb: float = 0.0
    kv_mb: float = 0.0
    runtime_mb: float = 0.0


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
    runtime_mb = kv_mb + batch_mb + flash_mb + mtp_mb
    return MemoryEstimate(
        vram_mb=model_on_gpu + runtime_mb,
        host_mb=model_on_host + kv_mb * 0.2 + 1024.0,
        model_gpu_mb=model_on_gpu,
        kv_mb=kv_mb,
        runtime_mb=runtime_mb,
    )


def estimate_per_gpu_vram_mb(
    config: TuneConfig,
    estimate: MemoryEstimate,
    *,
    gpu_free_mb: tuple[float, ...],
    active_gpu_indices: tuple[int, ...] = (),
) -> tuple[float, ...]:
    if not gpu_free_mb:
        return ()
    if len(gpu_free_mb) == 1:
        return (estimate.vram_mb,)
    main_position = _main_gpu_position(config.main_gpu, active_gpu_indices, len(gpu_free_mb))
    if config.split_mode == "none":
        allocations = [0.0 for _ in gpu_free_mb]
        allocations[main_position] = estimate.vram_mb
        return tuple(allocations)
    weights = _tensor_split_weights(config.tensor_split, len(gpu_free_mb)) or tuple(gpu_free_mb)
    weight_sum = sum(weights) or 1.0
    if config.split_mode == "row":
        allocations = [estimate.model_gpu_mb * weight / weight_sum for weight in weights]
        allocations[main_position] += estimate.runtime_mb
        return tuple(allocations)
    return tuple(estimate.vram_mb * weight / weight_sum for weight in weights)


def auto_constraint_policy(model_path: str | None = None) -> ConstraintPolicy:
    hardware = scan_hardware()
    gpus = hardware.get("gpus") or []
    active_gpus = [gpu for gpu in gpus if float(gpu["memory_free_mb"]) > 0]
    gpu_free_mb = tuple(float(gpu["memory_free_mb"]) for gpu in active_gpus)
    active_gpu_indices = tuple(int(gpu.get("index", idx)) for idx, gpu in enumerate(active_gpus))
    vram_limit_mb = sum(gpu_free_mb) if gpu_free_mb else 0.0
    system_memory_limit_mb = float(hardware.get("memory_available_gb") or 0.0) * 1024.0
    model_size_mb = 0.0
    if model_path and Path(model_path).exists():
        model_size_mb = Path(model_path).stat().st_size / (1024.0 * 1024.0)
    main_gpu = None
    if active_gpus:
        main_gpu = int(max(active_gpus, key=lambda gpu: float(gpu["memory_free_mb"])).get("index", 0))
    safe_splits = tuple(max(1.0, free_mb - 1536.0) for free_mb in gpu_free_mb)
    return ConstraintPolicy(
        model_size_mb=model_size_mb,
        vram_limit_mb=vram_limit_mb,
        gpu_free_mb=gpu_free_mb,
        active_gpu_indices=active_gpu_indices,
        system_memory_limit_mb=system_memory_limit_mb,
        device=",".join(str(index) for index in active_gpu_indices) if active_gpu_indices else None,
        split_mode="layer" if len(active_gpu_indices) > 1 else None,
        tensor_split=",".join(str(int(value)) for value in safe_splits) if len(safe_splits) > 1 else None,
        main_gpu=main_gpu,
        threads=int(hardware.get("recommended_threads") or 0) or None,
        threads_batch=int(hardware.get("recommended_threads_batch") or 0) or None,
        numa="distribute" if int(hardware.get("numa_nodes") or 1) > 1 else None,
        fit_target=",".join(str(int(1536)) for _ in safe_splits) if len(safe_splits) > 1 else None,
    )


def _tensor_split_weights(raw: str | None, expected_len: int) -> tuple[float, ...] | None:
    if not raw:
        return None
    values = []
    for part in raw.split(","):
        try:
            values.append(max(0.0, float(part.strip())))
        except ValueError:
            return None
    if len(values) != expected_len:
        return None
    return tuple(values)


def _main_gpu_position(main_gpu: int | None, active_gpu_indices: tuple[int, ...], gpu_count: int) -> int:
    if main_gpu is None:
        return 0
    if active_gpu_indices and main_gpu in active_gpu_indices:
        return active_gpu_indices.index(main_gpu)
    if 0 <= main_gpu < gpu_count:
        return main_gpu
    return 0


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
