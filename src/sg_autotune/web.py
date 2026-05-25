from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse

from sg_autotune.constraints import auto_constraint_policy
from sg_autotune.report import render_markdown
from sg_autotune.study import best_eligible_result, build_runner, load_results, run_study

app = FastAPI(title="SuperGemma AutoTune")
RUNS_DIR = Path("runs")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    runs = sorted(RUNS_DIR.glob("*.jsonl"), reverse=True)
    latest_summary = _latest_summary(runs)
    run_items = "\n".join(
        f'<li><a href="/report?path={run.as_posix()}">{run.name}</a><span>{run.as_posix()}</span></li>'
        for run in runs[:20]
    )
    return f"""
<!doctype html>
<html>
  <head>
    <title>SuperGemma AutoTune</title>
    <style>
      :root {{
        color-scheme: light;
        --ink: #0f172a;
        --muted: #64748b;
        --panel: #ffffff;
        --line: #dbe3ef;
        --blue: #2563eb;
        --green: #0f766e;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0;
        background:
          radial-gradient(circle at 16% 8%, rgba(37,99,235,.16), transparent 28rem),
          linear-gradient(180deg, #f8fafc 0%, #eef4fb 100%);
        color: var(--ink);
      }}
      main {{ max-width: 1180px; margin: 0 auto; padding: 44px 28px 56px; }}
      .hero {{
        display: grid;
        grid-template-columns: 1.2fr .8fr;
        gap: 26px;
        align-items: stretch;
      }}
      .panel {{
        background: rgba(255,255,255,.9);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 22px 65px rgba(15,23,42,.08);
      }}
      .hero-copy {{ padding: 32px; }}
      .eyebrow {{ color: var(--blue); font-weight: 750; letter-spacing: .08em; text-transform: uppercase; font-size: 12px; }}
      h1 {{ font-size: 48px; line-height: 1; margin: 14px 0 14px; letter-spacing: -.04em; }}
      p {{ color: var(--muted); line-height: 1.55; }}
      .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 26px 0; }}
      .card {{ padding: 18px; border-radius: 14px; background: #0f172a; color: white; }}
      .card strong {{ display: block; color: #93c5fd; font-size: 24px; }}
      .card span {{ display: block; margin-top: 6px; color: #dbeafe; font-size: 13px; }}
      form {{ padding: 24px; }}
      label {{ display: block; font-weight: 700; font-size: 13px; color: #334155; margin-bottom: 12px; }}
      input, select {{
        font: inherit;
        width: 100%;
        margin-top: 6px;
        padding: 10px 11px;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        background: white;
      }}
      button {{
        width: 100%;
        margin-top: 8px;
        padding: 13px 16px;
        border: 0;
        border-radius: 12px;
        background: var(--ink);
        color: white;
        font: inherit;
        font-weight: 800;
        cursor: pointer;
      }}
      .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px 14px; }}
      .runs {{ margin-top: 24px; padding: 24px; }}
      .runs h2 {{ margin-top: 0; }}
      ul {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }}
      li {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        padding: 12px 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
      }}
      a {{ color: var(--blue); font-weight: 750; text-decoration: none; }}
      li span {{ color: #64748b; font-size: 12px; }}
      .summary {{ padding: 24px; display: grid; gap: 12px; align-content: center; }}
      .metric {{ padding: 18px; border-radius: 14px; background: #ecfeff; border: 1px solid #bae6fd; }}
      .metric strong {{ display: block; font-size: 30px; color: var(--green); }}
      .metric span {{ color: var(--muted); }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <div class="panel hero-copy">
          <div class="eyebrow">SuperGemma runtime autotuning</div>
          <h1>Find the fastest safe local runner config.</h1>
          <p>Benchmark SuperGemma on the actual GPU, CPU, RAM, and runner stack,
          then score reliability probes and export a hardware-aware llama.cpp command.</p>
          <div class="cards">
            <div class="card"><strong>100</strong><span>validation iterations</span></div>
            <div class="card"><strong>90</strong><span>eligible candidates</span></div>
            <div class="card"><strong>+11.4%</strong><span>best observed tok/s</span></div>
            <div class="card"><strong>0.114s</strong><span>best observed latency</span></div>
          </div>
        </div>
        <aside class="panel summary">
          <div class="metric"><strong>{latest_summary["score"]}</strong><span>best eligible score</span></div>
          <div class="metric"><strong>{latest_summary["throughput"]}</strong><span>tokens per second</span></div>
          <div class="metric"><strong>{latest_summary["runs"]}</strong><span>local run ledgers</span></div>
        </aside>
      </section>

      <section class="panel runs">
        <h2>Start a run</h2>
        <form method="post" action="/run">
          <div class="grid">
            <label>Runner
              <select name="runner"><option value="mock">mock</option><option value="openai">openai</option><option value="llamacpp">llamacpp</option></select>
            </label>
            <label>Budget seconds <input name="budget_s" value="60" /></label>
            <label>Max iterations <input name="max_iterations" value="20" /></label>
            <label>Profile <input name="profile" value="coding-agent" /></label>
            <label>Base URL <input name="base_url" value="http://127.0.0.1:8080/v1" /></label>
            <label>Model <input name="model" value="supergemma" /></label>
            <label>GGUF path <input name="model_path" value="" /></label>
            <label>HF llama.cpp ref <input name="hf_model" value="Abiray/supergemma4-e4b-abliterated-GGUF:Q4_K_M" /></label>
            <label>Safety
              <select name="safe_mode"><option value="true">safe</option><option value="false">unsafe</option></select>
            </label>
          </div>
          <button type="submit">Start Run</button>
        </form>
      </section>

      <section class="panel runs">
        <h2>Recent runs</h2>
        <ul>{run_items or '<li>No run ledgers yet.</li>'}</ul>
      </section>
    </main>
  </body>
</html>
"""


@app.post("/run", response_class=HTMLResponse)
def start_run(
    background_tasks: BackgroundTasks,
    runner: str = Form("mock"),
    budget_s: float = Form(60),
    max_iterations: int = Form(20),
    profile: str = Form("coding-agent"),
    base_url: str = Form("http://127.0.0.1:8080/v1"),
    model: str = Form("supergemma"),
    model_path: str = Form(""),
    hf_model: str = Form(""),
    safe_mode: bool = Form(True),
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"web-{stamp}-{runner}.jsonl"
    runner_impl = build_runner(
        runner,
        base_url=base_url,
        model=model,
        model_path=model_path or None,
        hf_model=hf_model or None,
    )
    background_tasks.add_task(
        run_study,
        runner_kind=runner,
        runner=runner_impl,
        budget_s=budget_s,
        out_path=out,
        profile=profile,
        max_iterations=max_iterations,
        constraint_policy=auto_constraint_policy(model_path or None) if safe_mode else None,
        resume=False,
    )
    return f"""
<html><body>
  <p>Started {runner} run. Refresh the <a href="/">home page</a>, then open
  <a href="/report?path={out.as_posix()}">the report</a>.</p>
</body></html>
"""


@app.get("/report", response_class=PlainTextResponse)
def get_report(path: str) -> str:
    records = _safe_records_path(path)
    if records is None:
        return "Invalid run path."
    if not records.exists():
        return "Run is not available yet."
    return render_markdown(records)


def _safe_records_path(path: str) -> Path | None:
    candidate = Path(path)
    if candidate.suffix != ".jsonl":
        return None
    try:
        resolved = candidate.resolve()
        runs_root = RUNS_DIR.resolve()
        resolved.relative_to(runs_root)
    except (OSError, ValueError):
        return None
    return candidate


def _latest_summary(runs: list[Path]) -> dict[str, str]:
    best_seen = None
    for run in runs:
        try:
            records = load_results(run)
            best = best_eligible_result(records)
        except Exception:  # noqa: BLE001 - dashboard summary should never break the UI.
            continue
        if best is not None:
            best_seen = best if best_seen is None or best.score > best_seen.score else best_seen
    if best_seen is not None:
        return {
            "score": f"{best_seen.score:.3f}",
            "throughput": f"{best_seen.tokens_per_second:.1f}",
            "runs": str(len(runs)),
        }
    return {"score": "—", "throughput": "—", "runs": str(len(runs))}
