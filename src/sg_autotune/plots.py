from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from sg_autotune.study import load_results


def render_optimization_plot(
    record_paths: list[Path],
    out_path: Path,
    *,
    title: str = "SuperGemma AutoTune Optimization",
    labels: list[str] | None = None,
) -> None:
    """Render a compact PNG dashboard from one or more JSONL study ledgers."""
    if not record_paths:
        raise ValueError("At least one JSONL study path is required")
    labels = labels or [path.stem for path in record_paths]
    if len(labels) != len(record_paths):
        raise ValueError("--label count must match records count")

    series = []
    for path, label in zip(record_paths, labels, strict=True):
        records = load_results(path)
        if not records:
            raise ValueError(f"No records found in {path}")
        iterations = [record.iteration for record in records]
        scores = [record.result.score for record in records]
        best_scores = _best_so_far(scores)
        quality = [record.result.quality_score for record in records]
        throughput = [record.result.tokens_per_second for record in records]
        ttft = [record.result.ttft_s for record in records]
        best_latency = _best_so_far_lower(ttft)
        uplift = _uplift_summary(label, scores, throughput, ttft)
        series.append((label, iterations, best_scores, quality, throughput, best_latency, uplift))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fig.suptitle(title, fontsize=17, fontweight="bold")

    uplifts = []
    for label, iterations, best_scores, quality, throughput, ttft, uplift in series:
        axes[0, 0].plot(iterations, best_scores, marker="o", linewidth=2, label=label)
        axes[0, 1].plot(iterations, quality, marker="o", linewidth=2, label=label)
        axes[1, 0].plot(iterations, throughput, marker="o", linewidth=2, label=label)
        axes[1, 1].plot(iterations, ttft, marker="o", linewidth=2, label=label)
        uplifts.append(uplift)

    axes[0, 0].set_title("Search Trajectory: Best Score So Far")
    axes[0, 0].set_ylabel("score")
    axes[0, 1].set_title("Reliability Quality")
    axes[0, 1].set_ylabel("quality")
    axes[1, 0].set_title("Throughput")
    axes[1, 0].set_ylabel("tokens / second")
    axes[1, 1].set_title("Fastest Request Latency So Far")
    axes[1, 1].set_ylabel("seconds")

    for ax in axes.ravel():
        ax.set_xlabel("iteration")
        ax.legend(loc="best")
        ax.margins(x=0.02)

    fig.text(0.5, 0.015, " | ".join(uplifts), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _best_so_far(values: list[float]) -> list[float]:
    best = float("-inf")
    output = []
    for value in values:
        best = max(best, value)
        output.append(best)
    return output


def _best_so_far_lower(values: list[float]) -> list[float]:
    best = float("inf")
    output = []
    for value in values:
        best = min(best, value)
        output.append(best)
    return output


def _uplift_summary(label: str, scores: list[float], throughput: list[float], ttft: list[float]) -> str:
    first_score = scores[0]
    best_score = max(scores)
    score_delta = best_score - first_score
    first_tps = throughput[0]
    best_tps = max(throughput)
    tps_delta = _percent_change(first_tps, best_tps)
    first_ttft = ttft[0]
    best_ttft = min(ttft)
    latency_reduction = -_percent_change(first_ttft, best_ttft)
    return (
        f"{label}: score {score_delta:+.3f}, "
        f"tok/s {tps_delta:+.1f}%, latency reduction {latency_reduction:+.1f}% vs first"
    )


def _percent_change(first: float, later: float) -> float:
    if first == 0:
        return 0.0
    return ((later - first) / abs(first)) * 100
