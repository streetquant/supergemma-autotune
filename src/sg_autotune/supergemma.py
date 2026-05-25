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
    quant_ref: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"

    def download_command(self, *, local_dir: str = "models") -> str:
        if self.filename:
            return (
                "hf download "
                f"{self.repo_id} {self.filename} "
                f"--local-dir {local_dir}/{self.repo_id.split('/')[-1]}"
            )
        return f"hf download {self.repo_id} --local-dir {local_dir}/{self.repo_id.split('/')[-1]}"

    def llama_server_command(self, *, llama_server: str = "llama-server") -> str:
        if not self.quant_ref:
            return f"{llama_server} -hf {self.repo_id}"
        return f"{llama_server} -hf {self.repo_id}:{self.quant_ref}"

    def local_path(self, *, local_dir: str = "models") -> str | None:
        if not self.filename:
            return None
        return f"{local_dir}/{self.repo_id.split('/')[-1]}/{self.filename}"

    def checksum_command(self, *, local_dir: str = "models") -> str | None:
        local_path = self.local_path(local_dir=local_dir)
        if not local_path or not self.sha256:
            return None
        return f"echo '{self.sha256}  {local_path}' | sha256sum -c -"


SUPERGEMMA_MODELS: tuple[SuperGemmaModel, ...] = (
    SuperGemmaModel(
        repo_id="Abiray/supergemma4-e4b-abliterated-GGUF",
        filename="supergemma4-Q4_K_M.gguf",
        family="E4B GGUF",
        runner_hint="llama.cpp / LM Studio",
        notes=(
            "Fast validation target: GGUF quant of Jiunsong/supergemma4-e4b-abliterated; "
            "8B total parameters with 4-bit Q4_K_M local runner footprint."
        ),
        quant_ref="Q4_K_M",
        sha256="55cd785e8386557eb8722ef618fa8468d9587c78e48d23cfdd828018647f4a77",
        size_bytes=5335290144,
    ),
    SuperGemmaModel(
        repo_id="Jiunsong/supergemma4-26b-uncensored-gguf-v2",
        filename="supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf",
        family="26B GGUF",
        runner_hint="llama.cpp / LM Studio",
        notes="Best first target for AutoTune: single GGUF, consumer-GPU friendly quant.",
        quant_ref="Q4_K_M",
        sha256="e773b0a209d48524f9d485bca0818247f75d7ddde7cce951367a7e441fb59137",
        size_bytes=16796015232,
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
                "quant_ref": model.quant_ref,
                "sha256": model.sha256,
                "size_bytes": model.size_bytes,
                "hf_url": model.hf_url,
            }
            for model in SUPERGEMMA_MODELS
        ],
        indent=2,
        sort_keys=True,
    )
