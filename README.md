# SuperGemma AutoTune

Bayesian optimization for SuperGemma local runner settings.

Local LLM users often copy `llama.cpp`, Ollama, or LM Studio settings from a random
post and then wonder whether the model, runner, quant, KV cache, context size, or
chat template is responsible for slow or flaky behavior. SuperGemma AutoTune treats
the user's own machine as the benchmark target: run it for a chosen time budget and
it searches runner parameters, scores real reliability probes, and emits a
copy-paste configuration.

This is an open-source contribution to the SuperGemma ecosystem. It is **runtime
autotuning, not weight fine-tuning**: it does not modify model weights. It helps
SuperGemma users find the best runner configuration for their own hardware.

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

Show SuperGemma-first helpers:

```bash
uv run sg-autotune supergemma models
uv run sg-autotune supergemma quickstart
```

Run managed `llama-server` trials, one process per candidate:

```bash
uv run sg-autotune run \
  --runner llamacpp \
  --model-path ./supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf \
  --budget 2h \
  --profile coding-agent
```

By default, AutoTune applies a conservative hardware-aware safety filter using
available VRAM/RAM and the model file size. Add `--unsafe` only if you explicitly
want to let the runner try every candidate.

Runs are resumable by default. Re-run with the same `--out` path to continue the
same Bayesian study from its JSONL ledger, or pass `--fresh` to overwrite it.

Export the best result:

```bash
uv run sg-autotune export runs/latest.jsonl --target llamacpp --model-path ./model.gguf
uv run sg-autotune export runs/latest.jsonl --target ollama --model-path ./model.gguf
uv run sg-autotune export runs/latest.jsonl --target lmstudio
uv run sg-autotune export runs/latest.jsonl --target codex
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
- managed `llama-server` runner that starts, probes, and stops a server per candidate
- exporters for llama.cpp commands, Ollama Modelfiles, LM Studio JSON, and Codex env hints
- conservative hardware-aware constraints to avoid obvious OOM-prone candidates
- SuperGemma model catalog and quickstart helper commands
- resumable JSONL studies for long-running optimization sessions
- CLI UI with scan/run/report/web commands
- minimal FastAPI web UI
- JSONL run ledger and Markdown reports

The next iteration should add safer long-running resume/stop behavior and richer
model-specific memory estimates.
