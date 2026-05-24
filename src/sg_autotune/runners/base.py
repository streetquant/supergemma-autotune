from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sg_autotune.models import BenchmarkResult, TuneConfig
from sg_autotune.search_space import DEFAULT_SPACE, SearchSpace


@dataclass(frozen=True)
class RunnerCapabilities:
    name: str
    applied_params: tuple[str, ...]
    notes: str = ""

    @property
    def ignored_params(self) -> tuple[str, ...]:
        applied = set(self.applied_params)
        return tuple(spec.name for spec in DEFAULT_SPACE if spec.name not in applied)

    def search_space(self) -> SearchSpace:
        return SearchSpace.for_params(self.applied_params)


class Runner(ABC):
    @abstractmethod
    def capabilities(self) -> RunnerCapabilities:
        raise NotImplementedError

    @abstractmethod
    def benchmark(self, config: TuneConfig, *, profile: str) -> BenchmarkResult:
        raise NotImplementedError
