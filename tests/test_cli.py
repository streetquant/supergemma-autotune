from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from sg_autotune.cli import app, parse_duration


runner = CliRunner()


def test_parse_duration() -> None:
    assert parse_duration("30s") == 30
    assert parse_duration("2m") == 120
    assert parse_duration("1h") == 3600


@pytest.mark.parametrize("value", ["", "0s", "-1m", "soon"])
def test_parse_duration_rejects_bad_values(value: str) -> None:
    with pytest.raises(typer.BadParameter):
        parse_duration(value)


def test_supergemma_models_json_is_machine_readable() -> None:
    result = runner.invoke(app, ["supergemma", "models", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["repo_id"] == "Abiray/supergemma4-e4b-abliterated-GGUF"


def test_verify_model_rejects_incomplete_aria2_download(tmp_path: Path) -> None:
    model = tmp_path / "supergemma4-Q4_K_M.gguf"
    model.write_bytes(b"partial")
    model.with_name(model.name + ".aria2").write_text("metadata", encoding="utf-8")

    result = runner.invoke(app, ["supergemma", "verify-model", "--model-path", str(model)])

    assert result.exit_code != 0
    assert "Download appears incomplete" in result.output
