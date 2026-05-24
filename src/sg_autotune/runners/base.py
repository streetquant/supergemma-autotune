from __future__ import annotations

from abc import ABC, abstractmethod

from sg_autotune.models import BenchmarkResult, TuneConfig


class Runner(ABC):
    @abstractmethod
    def benchmark(self, config: TuneConfig, *, profile: str) -> BenchmarkResult:
        raise NotImplementedError

