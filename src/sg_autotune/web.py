from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse

from sg_autotune.constraints import auto_constraint_policy
from sg_autotune.report import render_markdown
from sg_autotune.study import build_runner, run_study

app = FastAPI(title="SuperGemma AutoTune")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    runs = sorted(Path("runs").glob("*.jsonl"), reverse=True)
    run_items = "\n".join(
        f'<li><a href="/report?path={run.as_posix()}">{run.as_posix()}</a></li>' for run in runs[:20]
    )
    return f"""
<!doctype html>
<html>
  <head>
    <title>SuperGemma AutoTune</title>
    <style>
      body {{ font-family: system-ui, sans-serif; max-width: 920px; margin: 40px auto; padding: 0 20px; }}
      input, select, button {{ font: inherit; padding: 8px; margin: 4px 0; width: 100%; }}
      button {{ background: #111827; color: white; border: 0; cursor: pointer; }}
      code, pre {{ background: #f3f4f6; padding: 2px 4px; }}
      .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    </style>
  </head>
  <body>
    <h1>SuperGemma AutoTune</h1>
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
        <label>Safety
          <select name="safe_mode"><option value="true">safe</option><option value="false">unsafe</option></select>
        </label>
      </div>
      <button type="submit">Start Run</button>
    </form>
    <h2>Recent Runs</h2>
    <ul>{run_items}</ul>
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
    safe_mode: bool = Form(True),
) -> str:
    out = Path("runs") / f"web-{runner}.jsonl"
    runner_impl = build_runner(
        runner,
        base_url=base_url,
        model=model,
        model_path=model_path or None,
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
    )
    return f"""
<html><body>
  <p>Started {runner} run. Refresh the <a href="/">home page</a>, then open
  <a href="/report?path={out.as_posix()}">the report</a>.</p>
</body></html>
"""


@app.get("/report", response_class=PlainTextResponse)
def get_report(path: str) -> str:
    records = Path(path)
    if not records.exists():
        return "Run is not available yet."
    return render_markdown(records)
