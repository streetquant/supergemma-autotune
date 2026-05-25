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
   optimization dashboard, hardware scan, and exported hardware-aware command.

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
- Hardware: NVIDIA GeForce RTX 3090, 24,576 MiB VRAM, compute capability 8.6;
  48 logical / 24 physical CPU threads; 251.56 GiB RAM; 1 NUMA node

AutoTune explicitly applies `--reasoning off` for llama.cpp benchmark candidates.
That keeps reasoning-template models from spending the probe budget in hidden
reasoning content and makes the JSON/tool/code validators measure final output.

Before suggesting a final command, the current build also scans hardware and
applies a runner plan: active GPU devices, per-GPU VRAM safety limits, tensor
split weights for multi-GPU hosts, main-GPU selection, CPU generation/batch
threads, NUMA hints, and llama.cpp fit targets where relevant. On this single
RTX 3090 host, the final command pins device `0`, main GPU `0`, 24 generation
threads, and 48 batch threads. On multi-GPU machines, the same planner emits
`--device`, `--split-mode layer`, `--tensor-split`, `--main-gpu`, and
`--fit-target` based on each GPU's free VRAM.

Results:

- Iterations: `100`
- Eligible candidates: `90`
- Best eligible score: `0.769` (`+1.5%` vs default)
- Baseline/default score: `0.757`
- Best eligible quality: `0.96`
- Best eligible throughput: `90.7 tok/s`
- Best eligible fastest request latency: `0.12s`
- Best observed throughput during the search: `102.4 tok/s` (`+11.4%` vs default)
- Best observed fastest request latency during the search: `0.114s` (`8.1%` lower than default)
- Probe gate: best eligible candidate passed JSON, tool-call, code-edit,
  long-context, and stability probes.

Best eligible command:

```bash
llama-server -m models/supergemma4-e4b-abliterated-GGUF/supergemma4-Q4_K_M.gguf -c 8192 -b 2048 -ub 512 -np 1 -ngl 99 --temp 0.242 --top-p 0.968 --top-k 20 --min-p 0.085 --reasoning off -ctk q4_0 -ctv q4_0 -fa on --device 0 --main-gpu 0 --threads 24 --threads-batch 48
```

Artifacts:

- Ledger: `runs/validation-supergemma-e4b-q4km.jsonl`
- Metadata: `runs/validation-supergemma-e4b-q4km.jsonl.meta.json`
- Report: `runs/validation-supergemma-e4b-q4km.md`
- Dashboard: `docs/assets/optimization-supergemma.png`
