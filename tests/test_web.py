from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from sg_autotune.web import app


def test_report_rejects_paths_outside_runs(tmp_path: Path) -> None:
    secret = tmp_path / "secret.jsonl"
    secret.write_text("secret", encoding="utf-8")

    client = TestClient(app)
    response = client.get("/report", params={"path": str(secret)})

    assert response.status_code == 200
    assert response.text == "Invalid run path."
