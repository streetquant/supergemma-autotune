from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx

from sg_autotune.models import BenchmarkResult, ProbeResult, TuneConfig
from sg_autotune.runners.base import Runner
from sg_autotune.runners.openai import OpenAICompatibleRunner
from sg_autotune.scoring import recompute_result_score


class LlamaCppManagedRunner(Runner):
    """Start llama-server for each candidate, run probes, then stop it."""

    def __init__(
        self,
        *,
        model_path: str,
        binary: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_s: float = 90.0,
    ):
        self.model_path = model_path
        self.binary = binary
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s

    def benchmark(self, config: TuneConfig, *, profile: str) -> BenchmarkResult:
        if not Path(self.model_path).exists():
            return self._failed(config, profile, f"model file not found: {self.model_path}")
        binary = shutil.which(self.binary) or self.binary
        if not shutil.which(binary) and not Path(binary).exists():
            return self._failed(config, profile, f"llama-server binary not found: {self.binary}")

        port = self.port or _pick_free_port()
        cmd = config.llama_cpp_args(self.model_path)
        cmd[0] = binary
        cmd += ["--host", self.host, "--port", str(port)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            self._wait_until_ready(port)
            runner = OpenAICompatibleRunner(
                base_url=f"http://{self.host}:{port}/v1",
                model="local-model",
                timeout_s=180.0,
            )
            return runner.benchmark(config, profile=profile)
        except Exception as exc:  # noqa: BLE001 - convert startup issues into study data.
            return self._failed(config, profile, str(exc))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def _wait_until_ready(self, port: int) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_error = "not ready"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"http://{self.host}:{port}/health", timeout=2)
                if response.status_code < 500:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            time.sleep(1)
        raise TimeoutError(f"llama-server did not become ready: {last_error}")

    @staticmethod
    def _failed(config: TuneConfig, profile: str, error: str) -> BenchmarkResult:
        result = BenchmarkResult(
            config=config,
            score=0,
            quality_score=0,
            tokens_per_second=0,
            ttft_s=0,
            peak_memory_mb=0,
            memory_pressure=1,
            failed=True,
            error=error,
            probes=[
                ProbeResult(
                    name="startup",
                    passed=False,
                    score=0,
                    latency_s=0,
                    tokens_per_second=0,
                    error=error,
                )
            ],
        )
        return recompute_result_score(result, profile)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

