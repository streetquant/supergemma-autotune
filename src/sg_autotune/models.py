from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


RunnerKind = Literal["mock", "openai", "llamacpp"]


class TuneConfig(BaseModel):
    ctx_size: int = Field(32768, ge=2048)
    batch_size: int = Field(1024, ge=128)
    ubatch_size: int = Field(256, ge=64)
    parallel: int = Field(1, ge=1)
    kv_cache: Literal["f16", "q8_0", "q4_0"] = "q8_0"
    gpu_layers: int = Field(99, ge=0)
    flash_attn: bool = True
    mtp_enabled: bool = False
    mtp_draft_n: int = Field(0, ge=0, le=8)
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.05, le=1.0)
    top_k: int = Field(20, ge=1, le=200)
    min_p: float = Field(0.0, ge=0.0, le=0.5)
    device: str | None = None
    split_mode: Literal["none", "layer", "row", "tensor"] | None = None
    tensor_split: str | None = None
    main_gpu: int | None = Field(default=None, ge=0)
    threads: int | None = Field(default=None, ge=1)
    threads_batch: int | None = Field(default=None, ge=1)
    numa: Literal["distribute", "isolate", "numactl"] | None = None
    fit_target: str | None = None

    def llama_cpp_args(self, model_ref: str, *, hf_model: bool = False) -> list[str]:
        args = [
            "llama-server",
            "-hf" if hf_model else "-m",
            model_ref,
            "-c",
            str(self.ctx_size),
            "-b",
            str(self.batch_size),
            "-ub",
            str(self.ubatch_size),
            "-np",
            str(self.parallel),
            "-ngl",
            str(self.gpu_layers),
            "--temp",
            f"{self.temperature:.3f}",
            "--top-p",
            f"{self.top_p:.3f}",
            "--top-k",
            str(self.top_k),
            "--min-p",
            f"{self.min_p:.3f}",
            "--reasoning",
            "off",
            "-ctk",
            self.kv_cache,
            "-ctv",
            self.kv_cache,
        ]
        if self.flash_attn:
            args += ["-fa", "on"]
        if self.device:
            args += ["--device", self.device]
        if self.split_mode:
            args += ["--split-mode", self.split_mode]
        if self.tensor_split:
            args += ["--tensor-split", self.tensor_split]
        if self.main_gpu is not None:
            args += ["--main-gpu", str(self.main_gpu)]
        if self.threads is not None:
            args += ["--threads", str(self.threads)]
        if self.threads_batch is not None:
            args += ["--threads-batch", str(self.threads_batch)]
        if self.numa:
            args += ["--numa", self.numa]
        if self.fit_target:
            args += ["--fit", "on", "--fit-target", self.fit_target]
        if self.mtp_enabled and self.mtp_draft_n > 0:
            args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(self.mtp_draft_n)]
        return args


class ProbeResult(BaseModel):
    name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    latency_s: float = Field(ge=0.0)
    tokens_per_second: float = Field(ge=0.0)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    config: TuneConfig
    score: float
    quality_score: float = Field(ge=0.0, le=1.0)
    tokens_per_second: float = Field(ge=0.0)
    ttft_s: float = Field(ge=0.0)
    peak_memory_mb: float = Field(ge=0.0)
    memory_pressure: float = Field(ge=0.0, le=1.0)
    failed: bool = False
    error: str | None = None
    probes: list[ProbeResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_s: float = Field(ge=0.0, default=0.0)


class StudyRecord(BaseModel):
    iteration: int
    runner: RunnerKind
    profile: str
    result: BenchmarkResult


class Recommendation(BaseModel):
    best: BenchmarkResult
    command: str
    summary: str
    warnings: list[str] = Field(default_factory=list)
