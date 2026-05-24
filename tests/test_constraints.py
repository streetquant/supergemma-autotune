from __future__ import annotations

from sg_autotune.constraints import ConstraintPolicy, estimate_memory_mb
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

