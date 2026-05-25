from __future__ import annotations

import json
import time
from collections.abc import Callable

import httpx

from sg_autotune.models import BenchmarkResult, ProbeResult, TuneConfig
from sg_autotune.runners.base import Runner, RunnerCapabilities
from sg_autotune.scoring import recompute_result_score


class OpenAICompatibleRunner(Runner):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout_s: float = 120.0,
        send_top_k: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.send_top_k = send_top_k

    def capabilities(self) -> RunnerCapabilities:
        applied_params = ("temperature", "top_p", "top_k") if self.send_top_k else ("temperature", "top_p")
        return RunnerCapabilities(
            name="openai-compatible",
            applied_params=applied_params,
            notes=(
                "Only request-level sampling parameters are applied. Server/runtime flags "
                "such as context, KV cache, batching, flash attention, GPU layers, and MTP "
                "must be configured outside this runner."
            ),
        )

    def benchmark(self, config: TuneConfig, *, profile: str) -> BenchmarkResult:
        started = time.perf_counter()
        probes = [self._run_probe(config, probe) for probe in PROBES]
        failed = any(probe.error for probe in probes)
        tps_values = [probe.tokens_per_second for probe in probes if probe.tokens_per_second > 0]
        latency_values = [probe.latency_s for probe in probes]
        result = BenchmarkResult(
            config=config,
            score=0,
            quality_score=0,
            tokens_per_second=round(sum(tps_values) / max(len(tps_values), 1), 3),
            ttft_s=round(min(latency_values) if latency_values else self.timeout_s, 3),
            peak_memory_mb=0.0,
            memory_pressure=_estimate_memory_pressure(config),
            failed=failed,
            error="one or more probes failed at transport level" if failed else None,
            probes=probes,
            duration_s=round(time.perf_counter() - started, 3),
        )
        return recompute_result_score(result, profile)

    def _run_probe(self, config: TuneConfig, probe: "Probe") -> ProbeResult:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are being evaluated by a local LLM runner autotuner. "
                        "Follow the requested output format exactly."
                    ),
                },
                {"role": "user", "content": probe.prompt},
            ],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": probe.max_tokens,
            "stream": False,
        }
        if self.send_top_k and config.top_k:
            payload["top_k"] = config.top_k

        t0 = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout_s,
            )
            latency = time.perf_counter() - t0
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"].get("content") or ""
            completion_tokens = _completion_tokens(data, text)
            tps = completion_tokens / max(latency, 1e-6)
            score = probe.score(text)
            return ProbeResult(
                name=probe.name,
                passed=score >= 0.75,
                score=score,
                latency_s=round(latency, 3),
                tokens_per_second=round(tps, 3),
                metadata={
                    "chars": len(text),
                    "completion_tokens": completion_tokens,
                    "text_preview": text[:500],
                },
            )
        except Exception as exc:  # noqa: BLE001 - probe errors should become data.
            return ProbeResult(
                name=probe.name,
                passed=False,
                score=0.0,
                latency_s=round(time.perf_counter() - t0, 3),
                tokens_per_second=0.0,
                error=str(exc),
            )


class Probe:
    def __init__(
        self,
        name: str,
        prompt: str,
        max_tokens: int,
        validator: Callable[[str], float],
    ):
        self.name = name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.validator = validator

    def score(self, text: str) -> float:
        return max(0.0, min(1.0, self.validator(text)))


PROBES = (
    Probe(
        "json",
        'Return exactly one JSON object with keys "ok", "runner", and "risk".',
        80,
        lambda text: 1.0
        if _json_object(text) is not None
        and set(_json_object(text) or {}) == {"ok", "runner", "risk"}
        else 0.0,
    ),
    Probe(
        "tool-call",
        'Return a tool call as JSON: {"tool":"read_file","args":{"path":"README.md"}}',
        80,
        lambda text: _tool_call_score(text),
    ),
    Probe(
        "code-edit",
        "Fix this Python bug and return only the corrected line: for i in range(len(items)+1):",
        80,
        lambda text: _code_edit_score(text),
    ),
    Probe(
        "long-context",
        "Remember token ALPHA-729. Explain in one sentence why local runner settings matter. End with ALPHA-729.",
        120,
        lambda text: _long_context_score(text),
    ),
    Probe(
        "stability",
        "Return five comma-separated words describing reliable local inference.",
        60,
        lambda text: _comma_words_score(text),
    ),
)


def _completion_tokens(data: dict, text: str) -> int:
    usage = data.get("usage") or {}
    if usage.get("completion_tokens"):
        return int(usage["completion_tokens"])
    return max(1, len(text.split()))


def _json_object(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    candidates = [stripped]
    extracted = _extract_first_json_object(stripped)
    if extracted and extracted != stripped:
        candidates.append(extracted)
    parsed = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for offset, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    return None


def _tool_call_score(text: str) -> float:
    parsed = _json_object(text)
    if not parsed:
        return 0.0
    if parsed.get("tool") != "read_file":
        return 0.0
    args = parsed.get("args")
    if not isinstance(args, dict):
        return 0.0
    return 1.0 if args.get("path") == "README.md" else 0.0


def _code_edit_score(text: str) -> float:
    stripped = text.strip().strip("`").strip()
    corrected = "for i in range(len(items)):"
    if stripped == corrected or stripped == corrected.rstrip(":"):
        return 1.0
    if corrected in stripped and "len(items)+1" not in stripped:
        return 1.0
    return 0.0


def _long_context_score(text: str) -> float:
    stripped = text.strip()
    if "settings" not in stripped.lower() or "ALPHA-729" not in stripped:
        return 0.0
    normalized = stripped.rstrip(" .!`")
    return 1.0 if normalized.endswith("ALPHA-729") else 0.75


def _comma_words_score(text: str) -> float:
    if "\n" in text.strip():
        return 0.0
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 5 or any(not part or " " in part for part in parts):
        return 0.0
    return 1.0 if any(part.lower() == "reliable" for part in parts) else 0.8


def _estimate_memory_pressure(config: TuneConfig) -> float:
    kv = {"f16": 0.12, "q8_0": 0.07, "q4_0": 0.04}[config.kv_cache]
    return round(min(1.0, 0.2 + config.ctx_size / 220000 + config.parallel * 0.05 + kv), 4)
