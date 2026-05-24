from __future__ import annotations

from sg_autotune.cli import parse_duration


def test_parse_duration() -> None:
    assert parse_duration("30s") == 30
    assert parse_duration("2m") == 120
    assert parse_duration("1h") == 3600

