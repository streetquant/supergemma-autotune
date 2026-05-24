from __future__ import annotations

from sg_autotune.supergemma import SUPERGEMMA_MODELS, model_catalog_json


def test_supergemma_catalog_has_primary_gguf() -> None:
    primary = SUPERGEMMA_MODELS[0]

    assert "supergemma4-26b" in primary.repo_id.lower()
    assert primary.filename is not None
    assert primary.filename.endswith(".gguf")
    assert "huggingface-cli download" in primary.download_command()


def test_supergemma_catalog_json() -> None:
    payload = model_catalog_json()

    assert "Jiunsong/supergemma4-26b-uncensored-gguf-v2" in payload
    assert "hf_url" in payload
