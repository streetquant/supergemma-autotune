from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sg_autotune import __version__
from sg_autotune.constraints import ConstraintPolicy
from sg_autotune.hardware import scan_hardware
from sg_autotune.models import RunnerKind
from sg_autotune.runners.base import RunnerCapabilities


def metadata_path(records_path: Path) -> Path:
    return records_path.with_suffix(records_path.suffix + ".meta.json")


def write_study_metadata(
    records_path: Path,
    *,
    runner: RunnerKind,
    profile: str,
    seed: int,
    capabilities: RunnerCapabilities,
    constraint_policy: ConstraintPolicy | None,
) -> None:
    path = metadata_path(records_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "sg-autotune-study-metadata-v1",
        "run_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "runner": runner,
        "profile": profile,
        "seed": seed,
        "capabilities": {
            "name": capabilities.name,
            "applied_params": list(capabilities.applied_params),
            "ignored_params": list(capabilities.ignored_params),
            "notes": capabilities.notes,
        },
        "constraint_policy": asdict(constraint_policy) if constraint_policy else None,
        "hardware": scan_hardware(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None

