from __future__ import annotations

import json
import shlex
from pathlib import Path

from sg_autotune.models import TuneConfig
from sg_autotune.study import best_eligible_result, best_result, load_results


def export_from_records(
    records_path: Path,
    *,
    target: str,
    model_path: str = "MODEL.gguf",
    allow_ineligible: bool = False,
) -> str:
    records = load_results(records_path)
    best = best_eligible_result(records)
    if best is None:
        if not allow_ineligible:
            raise ValueError("No eligible recommendation found; rerun longer or pass allow_ineligible.")
        best = best_result(records)
    return export_config(best.config, target=target, model_path=model_path)


def export_config(config: TuneConfig, *, target: str, model_path: str = "MODEL.gguf") -> str:
    if target == "llamacpp":
        return shlex.join(config.llama_cpp_args(model_path)) + "\n"
    if target == "ollama":
        return render_ollama_modelfile(config, model_path=model_path)
    if target == "lmstudio":
        return render_lmstudio_json(config)
    if target == "codex":
        return render_codex_env(config)
    raise ValueError(f"Unsupported export target: {target}")


def render_ollama_modelfile(config: TuneConfig, *, model_path: str) -> str:
    lines = [
        f"FROM {model_path}",
        f"PARAMETER temperature {config.temperature:.3f}",
        f"PARAMETER top_p {config.top_p:.3f}",
        f"PARAMETER top_k {config.top_k}",
        f"PARAMETER num_ctx {config.ctx_size}",
    ]
    # Ollama does not expose every llama.cpp server flag in a Modelfile. Keep
    # runner-only settings as comments so users do not lose the tuning context.
    lines += [
        "",
        "# Runner hints from SuperGemma AutoTune",
        f"# kv_cache={config.kv_cache}",
        f"# batch_size={config.batch_size}",
        f"# ubatch_size={config.ubatch_size}",
        f"# parallel={config.parallel}",
        f"# flash_attn={config.flash_attn}",
        f"# mtp_enabled={config.mtp_enabled}",
        f"# mtp_draft_n={config.mtp_draft_n}",
    ]
    return "\n".join(lines) + "\n"


def render_lmstudio_json(config: TuneConfig) -> str:
    return json.dumps(
        {
            "contextLength": config.ctx_size,
            "gpuLayers": config.gpu_layers,
            "flashAttention": config.flash_attn,
            "llamaCpp": {
                "batchSize": config.batch_size,
                "ubatchSize": config.ubatch_size,
                "cacheTypeK": config.kv_cache,
                "cacheTypeV": config.kv_cache,
                "parallelSlots": config.parallel,
                "mtp": {
                    "enabled": config.mtp_enabled,
                    "draftNMax": config.mtp_draft_n,
                },
            },
            "sampling": {
                "temperature": config.temperature,
                "topP": config.top_p,
                "topK": config.top_k,
                "minP": config.min_p,
            },
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


def render_codex_env(config: TuneConfig) -> str:
    return "\n".join(
        [
            "# Use this with a local OpenAI-compatible endpoint started from the recommended command.",
            "OPENAI_BASE_URL=http://127.0.0.1:8080/v1",
            "OPENAI_API_KEY=not-needed",
            f"# tuned_ctx_size={config.ctx_size}",
            f"# tuned_temperature={config.temperature:.3f}",
            f"# tuned_top_p={config.top_p:.3f}",
        ]
    ) + "\n"
