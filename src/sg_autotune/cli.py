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

app = typer.Typer(help="Bayesian autotuning for local LLM runner settings.")
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
    )
    if not history:
        raise typer.BadParameter("Budget ended before any iteration completed.")
    recommendation = build_recommendation(out)
    console.print(f"[green]Done[/green] {len(history)} iterations")
    console.print(recommendation.summary)
    console.print(recommendation.command)


@app.command()
def export(
    records: Path = typer.Argument(..., help="JSONL run ledger."),
    target: str = typer.Option("llamacpp", help="llamacpp, ollama, lmstudio, or codex."),
    model_path: str = typer.Option("MODEL.gguf", help="Path to use in generated config."),
    out: Path | None = typer.Option(None, help="Write export to this path."),
) -> None:
    """Export the best config for another runner or harness."""
    rendered = export_from_records(records, target=target, model_path=model_path)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {out}")
    else:
        console.print(rendered)


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
