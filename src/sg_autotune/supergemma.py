from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SuperGemmaModel:
    repo_id: str
    filename: str | None
    family: str
    runner_hint: str
    notes: str

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"

    def download_command(self, *, local_dir: str = "models") -> str:
        if self.filename:
            return (
                "huggingface-cli download "
                f"{self.repo_id} {self.filename} "
                f"--local-dir {local_dir}/{self.repo_id.split('/')[-1]}"
            )
        return f"huggingface-cli download {self.repo_id} --local-dir {local_dir}/{self.repo_id.split('/')[-1]}"


SUPERGEMMA_MODELS: tuple[SuperGemmaModel, ...] = (
    SuperGemmaModel(
        repo_id="Jiunsong/supergemma4-26b-uncensored-gguf-v2",
        filename="supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf",
        family="26B GGUF",
        runner_hint="llama.cpp / LM Studio",
        notes="Best first target for AutoTune: single GGUF, consumer-GPU friendly quant.",
    ),
    SuperGemmaModel(
        repo_id="Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2",
        filename=None,
        family="26B MLX 4-bit",
        runner_hint="MLX on Apple Silicon",
        notes="Apple Silicon path; useful for future MLX runner support.",
    ),
    SuperGemmaModel(
        repo_id="Jiunsong/SuperGemma4-31b-abliterated-GGUF",
        filename=None,
        family="31B GGUF",
        runner_hint="llama.cpp / LM Studio",
        notes="Larger GGUF target; use conservative safety constraints first.",
    ),
    SuperGemmaModel(
        repo_id="Jiunsong/supergemma4-e4b-abliterated",
        filename=None,
        family="E4B",
        runner_hint="Transformers / compatible runtimes",
        notes="Smaller experimentation target for lower-memory machines.",
    ),
)


def model_catalog_json() -> str:
    return json.dumps(
        [
            {
                "repo_id": model.repo_id,
                "filename": model.filename,
                "family": model.family,
                "runner_hint": model.runner_hint,
                "notes": model.notes,
                "hf_url": model.hf_url,
            }
            for model in SUPERGEMMA_MODELS
        ],
        indent=2,
        sort_keys=True,
    )

