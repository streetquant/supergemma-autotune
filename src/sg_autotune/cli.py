from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from sg_autotune.constraints import auto_constraint_policy
from sg_autotune.exporters import export_from_records
from sg_autotune.hardware import scan_hardware, scan_json
from sg_autotune.models import RunnerKind
from sg_autotune.report import build_recommendation, write_report
from sg_autotune.study import build_runner, json_summary, run_study
from sg_autotune.supergemma import SUPERGEMMA_MODELS, model_catalog_json

app = typer.Typer(help="Bayesian autotuning for local LLM runner settings.")
supergemma_app = typer.Typer(help="SuperGemma-first helper commands.")
app.add_typer(supergemma_app, name="supergemma")
console = Console()


@app.command()
def scan(json_output: bool = typer.Option(False, "--json", help="Print raw JSON.")) -> None:
    """Inspect local hardware and runner tools."""
    if json_output:
        console.print(scan_json())
        return
    data = scan_hardware()
    console.print("[bold]Hardware[/bold]")
    console.print(f"CPU threads: {data['cpu_count']}")
    console.print(f"RAM: {data['memory_available_gb']} GB available / {data['memory_total_gb']} GB total")
    console.print(f"Disk free: {data['disk_free_gb']} GB")
    if data["gpus"]:
        table = Table("GPU", "VRAM free", "VRAM total", "Driver")
        for gpu in data["gpus"]:
            table.add_row(
                gpu["name"],
                f"{gpu['memory_free_mb']} MB",
                f"{gpu['memory_total_mb']} MB",
                gpu["driver_version"],
            )
        console.print(table)
    console.print("[bold]Tools[/bold]")
    for tool, path in data["tools"].items():
        console.print(f"{tool}: {path or '[red]not found[/red]'}")


@supergemma_app.command("models")
def supergemma_models(json_output: bool = typer.Option(False, "--json", help="Print raw JSON.")) -> None:
    """List known SuperGemma model targets for autotuning."""
    if json_output:
        console.print(model_catalog_json())
        return
    table = Table("Model", "Family", "Runner", "Notes")
    for model in SUPERGEMMA_MODELS:
        table.add_row(model.repo_id, model.family, model.runner_hint, model.notes)
    console.print(table)


@supergemma_app.command("download-command")
def supergemma_download_command(
    index: int = typer.Argument(0, help="Model index from `sg-autotune supergemma models`."),
    local_dir: str = typer.Option("models", help="Download directory."),
) -> None:
    """Print a Hugging Face download command for a SuperGemma target."""
    try:
        model = SUPERGEMMA_MODELS[index]
    except IndexError as exc:
        raise typer.BadParameter(f"index must be between 0 and {len(SUPERGEMMA_MODELS) - 1}") from exc
    console.print(model.download_command(local_dir=local_dir), soft_wrap=True)


@supergemma_app.command("quickstart")
def supergemma_quickstart(
    model_path: str = typer.Option(
        "./models/supergemma4-26b-uncensored-gguf-v2/supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf",
        help="Expected GGUF model path.",
    )
) -> None:
    """Print the recommended SuperGemma AutoTune flow."""
    console.print(
        "\n".join(
            [
                "SuperGemma AutoTune is runtime autotuning, not weight fine-tuning.",
                "",
                "1. Download a SuperGemma GGUF:",
                SUPERGEMMA_MODELS[0].download_command(),
                "",
                "2. Run a short smoke study:",
                (
                    "sg-autotune run --runner llamacpp "
                    f"--model-path {model_path} --budget 10m --profile coding-agent"
                ),
                "",
                "3. Run the real study when the smoke run looks healthy:",
                (
                    "sg-autotune run --runner llamacpp "
                    f"--model-path {model_path} --budget 2h --profile coding-agent"
                ),
                "",
                "4. Export the best config:",
                "sg-autotune export runs/latest.jsonl --target llamacpp --model-path " + model_path,
            ]
        ),
        soft_wrap=True,
    )


@app.command()
def run(
    runner: RunnerKind = typer.Option("mock", help="Runner backend."),
    budget: str = typer.Option("60s", help="Time budget, e.g. 90s, 30m, 2h."),
    out: Path | None = typer.Option(None, help="JSONL run ledger path."),
    profile: str = typer.Option("coding-agent", help="Scoring profile."),
    base_url: str | None = typer.Option(None, help="OpenAI-compatible base URL."),
    model: str | None = typer.Option(None, help="Model name for OpenAI-compatible runner."),
    model_path: str | None = typer.Option(None, help="GGUF model path for managed llama.cpp."),
    llama_server: str = typer.Option("llama-server", help="llama-server binary path/name."),
    port: int = typer.Option(0, help="Managed llama.cpp port; 0 chooses a free port."),
    unsafe: bool = typer.Option(False, help="Disable hardware-aware safety constraints."),
    fresh: bool = typer.Option(False, help="Do not resume an existing JSONL ledger."),
    max_iterations: int | None = typer.Option(None, help="Optional iteration cap."),
    seed: int = typer.Option(7, help="Optimizer seed."),
) -> None:
    """Run an autotuning study."""
    budget_s = parse_duration(budget)
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = Path("runs") / f"{stamp}-{runner}.jsonl"
    runner_impl = build_runner(
        runner,
        base_url=base_url,
        model=model,
        model_path=model_path,
        llama_server=llama_server,
        port=port,
    )
    constraint_policy = None if unsafe else auto_constraint_policy(model_path)
    console.print(f"[bold]Starting {runner} study[/bold] for {budget_s:.0f}s -> {out}")
    history = run_study(
        runner_kind=runner,
        runner=runner_impl,
        budget_s=budget_s,
        out_path=out,
        profile=profile,
        seed=seed,
        max_iterations=max_iterations,
        constraint_policy=constraint_policy,
        resume=not fresh,
    )
    if not history:
        raise typer.BadParameter("Budget ended before any iteration completed.")
    recommendation = build_recommendation(out)
    console.print(f"[green]Done[/green] {len(history)} iterations")
    console.print(recommendation.summary)
    console.print(recommendation.command, soft_wrap=True)


@app.command()
def export(
    records: Path = typer.Argument(..., help="JSONL run ledger."),
    target: str = typer.Option("llamacpp", help="llamacpp, ollama, lmstudio, or codex."),
    model_path: str = typer.Option("MODEL.gguf", help="Path to use in generated config."),
    out: Path | None = typer.Option(None, help="Write export to this path."),
    allow_ineligible: bool = typer.Option(False, help="Allow exporting failed/low-quality results."),
) -> None:
    """Export the best config for another runner or harness."""
    rendered = export_from_records(
        records,
        target=target,
        model_path=model_path,
        allow_ineligible=allow_ineligible,
    )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {out}")
    else:
        console.print(rendered, soft_wrap=True)


@app.command()
def report(
    records: Path = typer.Argument(..., help="JSONL run ledger."),
    out: Path | None = typer.Option(None, help="Markdown report path."),
    model_path: str = typer.Option("MODEL.gguf", help="Path to use in generated llama.cpp command."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON summary."),
) -> None:
    """Generate a report from a run ledger."""
    if json_output:
        console.print(json_summary(records))
        return
    if out is None:
        out = records.with_suffix(".md")
    write_report(records, out, model_path=model_path)
    console.print(f"[green]Wrote[/green] {out}")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(7860, help="Bind port."),
) -> None:
    """Start the minimal web UI."""
    uvicorn.run("sg_autotune.web:app", host=host, port=port, reload=False)


def parse_duration(value: str) -> float:
    value = value.strip().lower()
    multipliers = {"s": 1, "m": 60, "h": 3600}
    if value[-1] in multipliers:
        return float(value[:-1]) * multipliers[value[-1]]
    return float(value)


if __name__ == "__main__":
    app()
