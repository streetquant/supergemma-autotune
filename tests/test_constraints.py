from __future__ import annotations

from sg_autotune.constraints import (
    ConstraintPolicy,
    auto_constraint_policy,
    estimate_memory_mb,
    estimate_per_gpu_vram_mb,
)
from sg_autotune.models import TuneConfig


def test_constraint_rejects_obvious_vram_overflow() -> None:
    config = TuneConfig(ctx_size=131072, parallel=4, kv_cache="f16", batch_size=2048)
    policy = ConstraintPolicy(model_size_mb=18000, vram_limit_mb=24000, reserve_vram_mb=1500)

    reason = policy.explain_rejection(config)

    assert reason is not None
    assert "VRAM" in reason


def test_memory_estimate_increases_with_context() -> None:
    small = estimate_memory_mb(TuneConfig(ctx_size=8192), model_size_mb=10000)
    large = estimate_memory_mb(TuneConfig(ctx_size=131072), model_size_mb=10000)

    assert large.vram_mb > small.vram_mb


def test_multi_gpu_policy_applies_llamacpp_hardware_defaults() -> None:
    policy = ConstraintPolicy(
        model_size_mb=20000,
        gpu_free_mb=(24000, 12000),
        active_gpu_indices=(0, 1),
        device="0,1",
        split_mode="layer",
        tensor_split="22464,10464",
        main_gpu=0,
        threads=24,
        threads_batch=48,
        numa="distribute",
        fit_target="1536,1536",
    )

    config = policy.apply_hardware_defaults(TuneConfig())

    assert config.device == "0,1"
    assert config.split_mode == "layer"
    assert config.tensor_split == "22464,10464"
    assert config.main_gpu == 0
    assert config.threads == 24
    assert config.threads_batch == 48
    assert config.numa == "distribute"
    assert config.fit_target == "1536,1536"


def test_multi_gpu_constraint_checks_each_device_limit() -> None:
    policy = ConstraintPolicy(
        model_size_mb=20000,
        gpu_free_mb=(12000, 12000),
        active_gpu_indices=(0, 1),
        reserve_vram_mb=1024,
    )
    config = TuneConfig(split_mode="none", main_gpu=0)

    reason = policy.explain_rejection(config)

    assert reason is not None
    assert "GPU 0 VRAM" in reason


def test_per_gpu_estimate_uses_tensor_split_weights() -> None:
    config = TuneConfig(split_mode="layer", tensor_split="3,1", main_gpu=0)
    estimate = estimate_memory_mb(config, model_size_mb=16000)

    per_gpu = estimate_per_gpu_vram_mb(
        config,
        estimate,
        gpu_free_mb=(24000, 12000),
        active_gpu_indices=(0, 1),
    )

    assert per_gpu[0] > per_gpu[1]
    assert round(sum(per_gpu), 6) == round(estimate.vram_mb, 6)


def test_auto_constraint_policy_builds_multi_gpu_plan(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake_scan_hardware() -> dict:
        return {
            "memory_available_gb": 64,
            "recommended_threads": 16,
            "recommended_threads_batch": 32,
            "numa_nodes": 2,
            "gpus": [
                {"index": 0, "memory_free_mb": 24000},
                {"index": 1, "memory_free_mb": 16000},
            ],
        }

    monkeypatch.setattr("sg_autotune.constraints.scan_hardware", fake_scan_hardware)

    policy = auto_constraint_policy()

    assert policy.gpu_free_mb == (24000.0, 16000.0)
    assert policy.active_gpu_indices == (0, 1)
    assert policy.device == "0,1"
    assert policy.split_mode == "layer"
    assert policy.tensor_split == "22464,14464"
    assert policy.main_gpu == 0
    assert policy.threads == 16
    assert policy.threads_batch == 32
    assert policy.numa == "distribute"
    assert policy.fit_target == "1536,1536"
