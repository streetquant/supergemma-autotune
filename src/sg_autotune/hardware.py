from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import psutil


def scan_hardware() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage(".").free / (1024**3), 2),
        "tools": {},
        "gpus": [],
    }
    for tool in ["ollama", "llama-server", "llama-cli", "nvidia-smi"]:
        payload["tools"][tool] = shutil.which(tool)
    if payload["tools"]["nvidia-smi"]:
        payload["gpus"] = _scan_nvidia()
    return payload


def _scan_nvidia() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "name": parts[0],
                    "memory_total_mb": int(parts[1]),
                    "memory_free_mb": int(parts[2]),
                    "driver_version": parts[3],
                }
            )
    return gpus


def scan_json() -> str:
    return json.dumps(scan_hardware(), indent=2, sort_keys=True)

