# SuperGemma Contribution Positioning

SuperGemma AutoTune is intended as an open-source contribution to the SuperGemma
ecosystem.

It is **not** a model fine-tuning tool. It does not train, merge, quantize, or
modify SuperGemma weights. It is a runtime autotuner: it searches local runner
hyperparameters for a specific SuperGemma model on a specific user's hardware.

## Why This Helps SuperGemma Users

Local LLM quality is often blamed on the model when the real issue is the runner
configuration:

- context size too high for available VRAM
- KV cache type degrading structured output
- MTP draft depth helping speed but harming acceptance or stability
- copied `llama-server` flags that fit another person's GPU, not yours
- harness expectations around JSON and tool calls that are not tested directly

AutoTune makes those tradeoffs measurable. It runs a time-budgeted search, scores
speed and reliability probes, and gives the user a concrete command/config to run.

## First-Class SuperGemma Targets

The fast validation target is:

```text
Abiray/supergemma4-e4b-abliterated-GGUF
supergemma4-Q4_K_M.gguf
```

This is the primary validation model because it is the SuperGemma E4B line in a
single llama.cpp-ready GGUF, with Q4_K_M called out by the quantizer as the
recommended everyday local-use quant. The catalog also includes the larger
`Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M` target for users with more
time and bandwidth.

The model-card llama.cpp entrypoint is:

```bash
llama-server -hf Abiray/supergemma4-e4b-abliterated-GGUF:Q4_K_M
```

For benchmark candidates, AutoTune also appends `--reasoning off` to the
generated llama.cpp command. The validation probes are designed to score final
JSON/tool/code output, so hidden reasoning tokens should not consume the probe
budget or produce empty OpenAI-compatible `content` fields.

## Desired Upstream Feedback

Useful feedback from SuperGemma maintainers and users:

- recommended sampling defaults per model family
- runner-specific flags that should be excluded from the search space
- reliable structured-output probes that reflect real SuperGemma use
- known-bad context/KV/MTP combinations
- model card snippets for "known good local runner configs"

## Three-Bullet Competition Pitch

```text
SuperGemma AutoTune: Bayesian optimization for SuperGemma local runner settings.

- Runs SuperGemma on your actual hardware for a chosen time budget and searches llama.cpp runtime settings directly.
- Scores speed plus reliability: JSON, tool calls, coding edits, context pressure, crashes, and memory safety.
- Outputs copy-paste configs so SuperGemma users stop guessing whether the model or their runner flags are the problem.
```

For LM Studio, vLLM, Ollama, and other OpenAI-compatible endpoints, AutoTune only
tunes request-level parameters that the endpoint confirms. Full runtime tuning is
the managed llama.cpp path used by the validation workflow.
