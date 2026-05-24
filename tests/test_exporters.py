from __future__ import annotations

from sg_autotune.exporters import export_config
from sg_autotune.models import TuneConfig


def test_export_targets_include_tuned_values() -> None:
    config = TuneConfig(ctx_size=65536, mtp_enabled=True, mtp_draft_n=2)

    llamacpp = export_config(config, target="llamacpp", model_path="model.gguf")
    ollama = export_config(config, target="ollama", model_path="model.gguf")
    lmstudio = export_config(config, target="lmstudio", model_path="model.gguf")

    assert "--spec-type draft-mtp" in llamacpp
    assert "PARAMETER num_ctx 65536" in ollama
    assert '"contextLength": 65536' in lmstudio

