from __future__ import annotations

from sg_autotune.runners.openai import PROBES


def _probe(name: str):
    return next(probe for probe in PROBES if probe.name == name)


def test_json_probe_rejects_keyword_junk() -> None:
    probe = _probe("json")

    assert probe.score("ok runner risk but not json") == 0.0
    assert probe.score('{"ok": true, "runner": "llama.cpp", "risk": "low"}') == 1.0
    assert probe.score('Sure:\n{"ok": true, "runner": "llama.cpp", "risk": "low"}') == 1.0


def test_tool_probe_requires_exact_json_shape() -> None:
    probe = _probe("tool-call")

    assert probe.score("tool read_file args README.md") == 0.0
    assert probe.score('{"tool":"read_file","args":{"path":"README.md"}}') == 1.0
    assert probe.score('```json\n{"tool":"read_file","args":{"path":"README.md"}}\n```') == 1.0


def test_code_probe_requires_corrected_line() -> None:
    probe = _probe("code-edit")

    assert probe.score("for i in range(len(items)+1):") == 0.0
    assert probe.score("for i in range(len(items)):") == 1.0
    assert probe.score("Corrected line:\nfor i in range(len(items)):") == 1.0


def test_stability_probe_accepts_valid_comma_words() -> None:
    probe = _probe("stability")

    assert probe.score("fast, stable, local, deterministic, measured") >= 0.75
    assert probe.score("fast, stable local, deterministic, measured, safe") == 0.0
