# SuperGemma AutoTune

Bayesian optimization for local LLM runner settings.

Local LLM users often copy `llama.cpp`, Ollama, or LM Studio settings from a random
post and then wonder whether the model, runner, quant, KV cache, context size, or
chat template is responsible for slow or flaky behavior. SuperGemma AutoTune treats
the user's own machine as the benchmark target: run it for a chosen time budget and
it searches runner parameters, scores real reliability probes, and emits a
copy-paste configuration.

## Quick Start

Run the deterministic mock optimizer:

```bash
uv run sg-autotune run --runner mock --budget 30s --out runs/mock.jsonl
uv run sg-autotune report runs/mock.jsonl --out runs/mock-report.md
```

Run against an OpenAI-compatible local endpoint:

```bash
uv run sg-autotune run \
  --runner openai \
  --base-url http://127.0.0.1:8080/v1 \
  --model supergemma \
  --budget 30m \
  --profile coding-agent
```

Start the minimal web UI:

```bash
uv run sg-autotune web --host 127.0.0.1 --port 7860
```

## What It Tunes

- `ctx_size`
- `batch_size`
- `ubatch_size`
- `parallel`
- `kv_cache`
- `gpu_layers`
- `flash_attn`
- `mtp_enabled`
- `mtp_draft_n`
- `temperature`
- `top_p`
- `top_k`
- `min_p`

## What It Scores

The optimizer balances speed and reliability:

```text
score = quality
      + speed_weight * normalized_tokens_per_second
      - latency_weight * normalized_time_to_first_token
      - memory_weight * memory_pressure
      - failure_penalty
```

Reliability probes check JSON output, tool-call-shaped output, a tiny coding edit,
long-context recall, repeated-call stability, and runner crashes/timeouts.

## Project Status

This first iteration ships the production shape:

- Bayesian optimizer using `GaussianProcessRegressor` + expected improvement
- deterministic mock runner for end-to-end tests
- OpenAI-compatible runner for Ollama, llama.cpp server, LM Studio, and vLLM-style APIs
- CLI UI with scan/run/report/web commands
- minimal FastAPI web UI
- JSONL run ledger and Markdown reports

The next iteration should add native `llama-server` process orchestration and richer
hardware-aware constraints.

