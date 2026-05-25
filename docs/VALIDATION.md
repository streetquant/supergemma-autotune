# Validation Notes

This page captures reproducible validation for SuperGemma AutoTune.

The release validation target is the SuperGemma E4B GGUF path, not a generic
Ollama proxy model:

- Model: `Abiray/supergemma4-e4b-abliterated-GGUF`
- Quant/ref: `Q4_K_M`
- File: `supergemma4-Q4_K_M.gguf`
- Runner: `llama.cpp` / `llama-server`

The Hugging Face model card recommends running this build with:

```bash
llama-server -hf Abiray/supergemma4-e4b-abliterated-GGUF:Q4_K_M
```

Before benchmarking, verify the local GGUF. A stale/corrupt file can still load
but generate control-token garbage, so validation treats the SHA-256 as a gate:

```bash
uv run sg-autotune supergemma verify-model
```

## Validation Tiers

1. **CI and mock validation**: deterministic tests for the optimizer, CLI,
   reports, exports, plots, metadata, web path safety, and recommendation gates.
2. **SuperGemma llama.cpp validation**: managed `llama-server` trials against the
   recommended `Abiray/supergemma4-e4b-abliterated-GGUF:Q4_K_M` GGUF target.
3. **Long-run evidence**: raw JSONL ledger, metadata sidecar, Markdown report,
   optimization dashboard, hardware scan, and exported command.

## Commands

```bash
# Optional explicit download. The managed llama.cpp runner can also fetch via -hf.
hf download \
  Abiray/supergemma4-e4b-abliterated-GGUF \
  supergemma4-Q4_K_M.gguf \
  --local-dir models/supergemma4-e4b-abliterated-GGUF

uv run sg-autotune run \
  --runner llamacpp \
  --llama-server /8tb/projects/deps/llama.cpp/build/bin/llama-server \
  --model-path models/supergemma4-e4b-abliterated-GGUF/supergemma4-Q4_K_M.gguf \
  --budget 20m \
  --max-iterations 10 \
  --out runs/validation-supergemma-e4b-q4km.jsonl \
  --profile coding-agent \
  --startup-timeout 900 \
  --fresh

uv run sg-autotune report \
  runs/validation-supergemma-e4b-q4km.jsonl \
  --out runs/validation-supergemma-e4b-q4km.md \
  --model-path models/supergemma4-e4b-abliterated-GGUF/supergemma4-Q4_K_M.gguf

uv run sg-autotune plot \
  runs/validation-supergemma-e4b-q4km.jsonl \
  --label "SuperGemma E4B Q4_K_M" \
  --out docs/assets/optimization-supergemma.png \
  --title "SuperGemma AutoTune Validation"
```

## Reading The Plot

The upper-left panel is a search trajectory: it shows the best score observed so
far during the run. It demonstrates what the optimizer found in this ledger, but
it is not a standalone proof against every default or random-search baseline.
The first candidate is the default config, so the footer reports deltas against
that baseline for score, throughput, and request latency.

## 2026-05-25 Validation Run

Validation was run with the recommended SuperGemma E4B GGUF on the managed
`llama.cpp` path, not through Ollama:

- Model: `Abiray/supergemma4-e4b-abliterated-GGUF`
- File: `models/supergemma4-e4b-abliterated-GGUF/supergemma4-Q4_K_M.gguf`
- SHA-256: `55cd785e8386557eb8722ef618fa8468d9587c78e48d23cfdd828018647f4a77`
- Size: `5,335,290,144` bytes
- Runner: `/8tb/projects/deps/llama.cpp/build/bin/llama-server`
- llama.cpp version: `version: 1 (549b9d8)`
- Hardware: NVIDIA GeForce RTX 3090, 24,576 MiB VRAM; 48 CPU threads;
  251.56 GiB RAM

AutoTune explicitly applies `--reasoning off` for llama.cpp benchmark candidates.
That keeps reasoning-template models from spending the probe budget in hidden
reasoning content and makes the JSON/tool/code validators measure final output.

Results:

- Iterations: `10`
- Best eligible score: `0.765`
- Baseline/default score: `0.757`
- Best eligible quality: `0.96`
- Best eligible throughput: `92.6 tok/s`
- Best eligible fastest request latency: `0.13s`
- Best observed throughput during the search: `102.4 tok/s`
- Best observed fastest request latency during the search: `0.114s`
- Probe gate: best eligible candidate passed JSON, tool-call, code-edit,
  long-context, and stability probes.

Best eligible command:

```bash
llama-server -m models/supergemma4-e4b-abliterated-GGUF/supergemma4-Q4_K_M.gguf -c 8192 -b 1024 -ub 512 -np 2 -ngl 99 --temp 0.828 --top-p 0.925 --top-k 80 --min-p 0.025 --reasoning off -ctk q4_0 -ctv q4_0
```

Artifacts:

- Ledger: `runs/validation-supergemma-e4b-q4km.jsonl`
- Metadata: `runs/validation-supergemma-e4b-q4km.jsonl.meta.json`
- Report: `runs/validation-supergemma-e4b-q4km.md`
- Dashboard: `docs/assets/optimization-supergemma.png`
