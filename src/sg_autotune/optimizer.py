from __future__ import annotations

import json
import random

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from sg_autotune.models import BenchmarkResult, TuneConfig
from sg_autotune.search_space import SearchSpace


class BayesianOptimizer:
    def __init__(
        self,
        search_space: SearchSpace | None = None,
        *,
        seed: int = 7,
        initial_random: int = 6,
        candidate_pool_size: int = 256,
    ):
        self.search_space = search_space or SearchSpace()
        self.rng = random.Random(seed)
        self.initial_random = initial_random
        self.candidate_pool_size = candidate_pool_size
        self._seen: set[str] = set()

    def suggest(self, history: list[BenchmarkResult]) -> TuneConfig:
        if len(history) < self.initial_random:
            return self._unique_random()

        x = np.vstack([self.search_space.encode(item.config) for item in history])
        y = np.array([item.score for item in history], dtype=float)

        kernel = ConstantKernel(1.0, (0.01, 10.0)) * Matern(nu=2.5) + WhiteKernel(
            noise_level=0.02, noise_level_bounds=(1e-5, 0.2)
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=self.rng.randrange(1, 2**31 - 1),
            n_restarts_optimizer=2,
        )
        model.fit(x, y)

        candidates = self.search_space.candidate_pool(self.rng, self.candidate_pool_size)
        candidates = [candidate for candidate in candidates if self._fingerprint(candidate) not in self._seen]
        if not candidates:
            return self._unique_random()

        x_candidates = np.vstack([self.search_space.encode(candidate) for candidate in candidates])
        mean, std = model.predict(x_candidates, return_std=True)
        acquisition = expected_improvement(mean, std, best=float(np.max(y)))
        best_idx = int(np.argmax(acquisition))
        chosen = candidates[best_idx]
        self._seen.add(self._fingerprint(chosen))
        return chosen

    def observe_existing(self, history: list[BenchmarkResult]) -> None:
        for item in history:
            self._seen.add(self._fingerprint(item.config))

    def _unique_random(self) -> TuneConfig:
        for _ in range(1000):
            config = self.search_space.sample(self.rng)
            key = self._fingerprint(config)
            if key not in self._seen:
                self._seen.add(key)
                return config
        config = self.search_space.sample(self.rng)
        self._seen.add(self._fingerprint(config))
        return config

    @staticmethod
    def _fingerprint(config: TuneConfig) -> str:
        return json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))


def expected_improvement(mean: np.ndarray, std: np.ndarray, *, best: float) -> np.ndarray:
    std = np.maximum(std, 1e-9)
    improvement = mean - best
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)
