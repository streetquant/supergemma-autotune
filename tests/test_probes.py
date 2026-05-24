from __future__ import annotations

from sg_autotune.runners.openai import PROBES


def _probe(name: str):
    return next(probe for probe in PROBES if probe.name == name)


def test_json_probe_rejects_keyword_junk() -> None:
    probe = _probe("json")

    assert probe.score("ok runner risk but not json") == 0.0
    assert probe.score('{"ok": true, "runner": "llama.cpp", "risk": "low"}') == 1.0


def test_tool_probe_requires_exact_json_shape() -> None:
    probe = _probe("tool-call")

    assert probe.score("tool read_file args README.md") == 0.0
    assert probe.score('{"tool":"read_file","args":{"path":"README.md"}}') == 1.0


def test_code_probe_requires_corrected_line() -> None:
    probe = _probe("code-edit")

    assert probe.score("for i in range(len(items)+1):") == 0.0
    assert probe.score("for i in range(len(items)):") == 1.0
