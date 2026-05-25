from __future__ import annotations

from sg_autotune.models import TuneConfig
from sg_autotune.optimizer import BayesianOptimizer
from sg_autotune.runners.openai import OpenAICompatibleRunner


def test_openai_runner_search_space_only_samples_applied_params() -> None:
    runner = OpenAICompatibleRunner(base_url="http://127.0.0.1:9/v1", model="x")
    optimizer = BayesianOptimizer(search_space=runner.capabilities().search_space(), seed=1)

    suggestion = optimizer.suggest([])
    default = TuneConfig()

    assert runner.capabilities().applied_params == ("temperature", "top_p")
    assert suggestion.ctx_size == default.ctx_size
    assert suggestion.batch_size == default.batch_size
    assert suggestion.kv_cache == default.kv_cache
    assert suggestion.mtp_enabled == default.mtp_enabled
