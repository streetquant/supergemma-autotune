from __future__ import annotations

import json
import platform
import shutil
import subprocess
from typing import Any

import psutil


def scan_hardware() -> dict[str, Any]:
    virtual_memory = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    logical_cpus = psutil.cpu_count(logical=True) or 1
    physical_cpus = psutil.cpu_count(logical=False) or logical_cpus
    numa_nodes = _scan_numa_nodes()
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_count": logical_cpus,
        "cpu_physical_count": physical_cpus,
        "cpu_freq_mhz": round(cpu_freq.current, 1) if cpu_freq else None,
        "numa_nodes": numa_nodes,
        "recommended_threads": min(logical_cpus, max(4, physical_cpus)),
        "recommended_threads_batch": min(logical_cpus, max(4, logical_cpus)),
        "memory_total_gb": round(virtual_memory.total / (1024**3), 2),
        "memory_available_gb": round(virtual_memory.available / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage(".").free / (1024**3), 2),
        "tools": {},
        "gpus": [],
    }
    for tool in ["ollama", "llama-server", "llama-cli", "nvidia-smi"]:
        payload["tools"][tool] = shutil.which(tool)
    if payload["tools"]["nvidia-smi"]:
        payload["gpus"] = _scan_nvidia()
    gpu_free = [gpu["memory_free_mb"] for gpu in payload["gpus"]]
    gpu_total = [gpu["memory_total_mb"] for gpu in payload["gpus"]]
    payload["gpu_count"] = len(payload["gpus"])
    payload["gpu_memory_free_total_mb"] = sum(gpu_free)
    payload["gpu_memory_total_mb"] = sum(gpu_total)
    payload["largest_gpu_free_mb"] = max(gpu_free, default=0)
    return payload


def _scan_nvidia() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version,"
        "compute_cap,pci.bus_id,power.limit,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 11:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "driver_version": parts[5],
                    "compute_capability": parts[6],
                    "pci_bus_id": parts[7],
                    "power_limit_w": _float_or_none(parts[8]),
                    "temperature_c": _int_or_none(parts[9]),
                    "utilization_gpu_percent": _int_or_none(parts[10]),
                }
            )
    return gpus


def _scan_numa_nodes() -> int:
    try:
        output = subprocess.check_output(
            ["lscpu"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return 1
    for line in output.splitlines():
        if line.startswith("NUMA node(s):"):
            try:
                return max(1, int(line.split(":", 1)[1].strip()))
            except ValueError:
                return 1
    return 1


def _int_or_none(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def scan_json() -> str:
    return json.dumps(scan_hardware(), indent=2, sort_keys=True)
