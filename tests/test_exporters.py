from __future__ import annotations

from sg_autotune.exporters import export_config
from sg_autotune.models import TuneConfig


def test_export_targets_include_tuned_values() -> None:
    config = TuneConfig(ctx_size=65536, mtp_enabled=True, mtp_draft_n=2)

    llamacpp = export_config(config, target="llamacpp", model_path="model.gguf")
    ollama = export_config(config, target="ollama", model_path="model.gguf")
    lmstudio = export_config(config, target="lmstudio", model_path="model.gguf")

    assert "--spec-type draft-mtp" in llamacpp
    assert "--reasoning off" in llamacpp
    assert "PARAMETER num_ctx 65536" in ollama
    assert '"contextLength": 65536' in lmstudio


def test_llamacpp_export_quotes_model_paths() -> None:
    config = TuneConfig()

    rendered = export_config(config, target="llamacpp", model_path="models/my model.gguf")

    assert "'models/my model.gguf'" in rendered


def test_llamacpp_args_support_hf_refs() -> None:
    config = TuneConfig()

    args = config.llama_cpp_args("Abiray/supergemma4-e4b-abliterated-GGUF:Q4_K_M", hf_model=True)

    assert args[:3] == ["llama-server", "-hf", "Abiray/supergemma4-e4b-abliterated-GGUF:Q4_K_M"]


def test_llamacpp_args_include_multi_gpu_and_cpu_hardware_flags() -> None:
    config = TuneConfig(
        device="0,1",
        split_mode="layer",
        tensor_split="22000,18000",
        main_gpu=0,
        threads=24,
        threads_batch=48,
        numa="distribute",
        fit_target="1536,1536",
    )

    rendered = " ".join(config.llama_cpp_args("model.gguf"))

    assert "--device 0,1" in rendered
    assert "--split-mode layer" in rendered
    assert "--tensor-split 22000,18000" in rendered
    assert "--main-gpu 0" in rendered
    assert "--threads 24" in rendered
    assert "--threads-batch 48" in rendered
    assert "--numa distribute" in rendered
    assert "--fit on --fit-target 1536,1536" in rendered
