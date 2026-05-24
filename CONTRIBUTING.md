# Contributing

SuperGemma AutoTune welcomes small, evidence-backed contributions.

## Good Contributions

- Add a runner backend with a deterministic mock path and tests.
- Add a probe that catches a real local-runner failure mode.
- Add a search-space constraint for a known unsafe flag combination.
- Improve SuperGemma model-specific defaults with links to model cards or measured runs.
- Add report/export formats for local coding harnesses.

## Ground Rules

- Do not claim this fine-tunes model weights; it tunes runtime parameters.
- Keep every runner side effect explicit and reversible.
- Prefer JSONL artifacts for reproducibility.
- Include mock-runner tests so CI does not require a GPU or model download.
- Document hardware, runner version, model file, and command flags for real benchmark claims.

## Local Development

```bash
pytest -q
PYTHONPATH=src python -m sg_autotune.cli run --runner mock --budget 10s --max-iterations 8
PYTHONPATH=src python -m sg_autotune.cli web
```

