from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sg_autotune.models import TuneConfig


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: Literal["int", "float", "bool", "categorical"]
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = ()
    log: bool = False


DEFAULT_SPACE: tuple[ParamSpec, ...] = (
    ParamSpec("ctx_size", "categorical", choices=(8192, 16384, 32768, 65536, 98304, 131072)),
    ParamSpec("batch_size", "categorical", choices=(512, 1024, 2048)),
    ParamSpec("ubatch_size", "categorical", choices=(128, 256, 512)),
    ParamSpec("parallel", "categorical", choices=(1, 2, 4)),
    ParamSpec("kv_cache", "categorical", choices=("f16", "q8_0", "q4_0")),
    ParamSpec("gpu_layers", "categorical", choices=(32, 64, 99)),
    ParamSpec("flash_attn", "bool"),
    ParamSpec("mtp_enabled", "bool"),
    ParamSpec("mtp_draft_n", "categorical", choices=(0, 1, 2, 3, 4, 6)),
    ParamSpec("temperature", "float", low=0.1, high=0.9),
    ParamSpec("top_p", "float", low=0.75, high=1.0),
    ParamSpec("top_k", "categorical", choices=(10, 20, 40, 80)),
    ParamSpec("min_p", "float", low=0.0, high=0.12),
)


class SearchSpace:
    def __init__(self, specs: tuple[ParamSpec, ...] = DEFAULT_SPACE):
        self.specs = specs
        self._names = [spec.name for spec in specs]

    @classmethod
    def for_params(cls, names: tuple[str, ...]) -> "SearchSpace":
        wanted = set(names)
        specs = tuple(spec for spec in DEFAULT_SPACE if spec.name in wanted)
        return cls(specs)

    def sample(self, rng: random.Random) -> TuneConfig:
        values: dict[str, Any] = {}
        for spec in self.specs:
            if spec.kind == "bool":
                values[spec.name] = rng.choice([False, True])
            elif spec.kind == "categorical":
                values[spec.name] = rng.choice(list(spec.choices))
            elif spec.kind == "int":
                assert spec.low is not None and spec.high is not None
                if spec.log:
                    value = math.exp(rng.uniform(math.log(spec.low), math.log(spec.high)))
                else:
                    value = rng.uniform(spec.low, spec.high)
                values[spec.name] = int(round(value))
            elif spec.kind == "float":
                assert spec.low is not None and spec.high is not None
                values[spec.name] = rng.uniform(spec.low, spec.high)
        self._repair(values)
        return TuneConfig(**values)

    def encode(self, config: TuneConfig) -> np.ndarray:
        encoded: list[float] = []
        raw = config.model_dump()
        for spec in self.specs:
            value = raw[spec.name]
            if spec.kind == "bool":
                encoded.append(1.0 if value else 0.0)
            elif spec.kind == "categorical":
                choices = list(spec.choices)
                encoded.append(float(choices.index(value)) / max(len(choices) - 1, 1))
            elif spec.kind in {"int", "float"}:
                assert spec.low is not None and spec.high is not None
                if spec.log:
                    low = math.log(spec.low)
                    high = math.log(spec.high)
                    encoded.append((math.log(float(value)) - low) / (high - low))
                else:
                    encoded.append((float(value) - spec.low) / (spec.high - spec.low))
        return np.array(encoded, dtype=float)

    def decode(self, vector: np.ndarray) -> TuneConfig:
        values: dict[str, Any] = {}
        for spec, raw_value in zip(self.specs, vector, strict=True):
            value = float(np.clip(raw_value, 0.0, 1.0))
            if spec.kind == "bool":
                values[spec.name] = value >= 0.5
            elif spec.kind == "categorical":
                idx = int(round(value * (len(spec.choices) - 1)))
                values[spec.name] = spec.choices[idx]
            elif spec.kind == "int":
                assert spec.low is not None and spec.high is not None
                if spec.log:
                    decoded = math.exp(math.log(spec.low) + value * (math.log(spec.high) - math.log(spec.low)))
                else:
                    decoded = spec.low + value * (spec.high - spec.low)
                values[spec.name] = int(round(decoded))
            elif spec.kind == "float":
                assert spec.low is not None and spec.high is not None
                values[spec.name] = spec.low + value * (spec.high - spec.low)
        self._repair(values)
        return TuneConfig(**values)

    def candidate_pool(self, rng: random.Random, size: int) -> list[TuneConfig]:
        return [self.sample(rng) for _ in range(size)]

    @staticmethod
    def _repair(values: dict[str, Any]) -> None:
        if not values.get("mtp_enabled", False):
            values["mtp_draft_n"] = 0
        elif values.get("mtp_draft_n", 0) == 0:
            values["mtp_draft_n"] = 2
        if values.get("ubatch_size", 0) > values.get("batch_size", 0):
            values["ubatch_size"] = values["batch_size"]
