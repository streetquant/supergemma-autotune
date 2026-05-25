from __future__ import annotations

from sg_autotune.supergemma import SUPERGEMMA_MODELS, model_catalog_json


def test_supergemma_catalog_has_primary_gguf() -> None:
    primary = SUPERGEMMA_MODELS[0]

    assert "supergemma4-e4b" in primary.repo_id.lower()
    assert primary.filename is not None
    assert primary.filename.endswith(".gguf")
    assert primary.sha256 == "55cd785e8386557eb8722ef618fa8468d9587c78e48d23cfdd828018647f4a77"
    assert primary.checksum_command() is not None
    assert "hf download" in primary.download_command()
    assert primary.llama_server_command() == (
        "llama-server -hf Abiray/supergemma4-e4b-abliterated-GGUF:Q4_K_M"
    )


def test_supergemma_catalog_json() -> None:
    payload = model_catalog_json()

    assert "Abiray/supergemma4-e4b-abliterated-GGUF" in payload
    assert "Jiunsong/supergemma4-26b-uncensored-gguf-v2" in payload
    assert "hf_url" in payload
