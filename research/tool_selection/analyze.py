"""
research/tool_selection/analyze.py
====================================
Read raw_results_*.jsonl from results/ and produce:

  1. Console summary table (per condition × registry size)
  2. Plot A: Accuracy vs registry size (naive vs retrieval k=3/5/10)
  3. Plot B: Accuracy vs token cost tradeoff (retrieval k values)
  4. Plot C: Prompt token consumption (naive vs retrieval)

Usage
-----
    python research/tool_selection/analyze.py [--results-dir ./results]
                                               [--out-dir ./results]
                                               [--results-file path/to/raw_results.jsonl]

If --results-file is given, uses that file. Otherwise uses the most
recent raw_results_*.jsonl in --results-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── optional matplotlib ────────────────────────────────────────────────── #
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Tables only, no plots.", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────── #
# Data loading                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #


def load_results(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_latest_results(results_dir: Path) -> Path:
    files = sorted(results_dir.glob("raw_results_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No raw_results_*.jsonl in {results_dir}")
    return files[-1]


# ─────────────────────────────────────────────────────────────────────────── #
# Aggregation                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #


def aggregate(rows: List[Dict]) -> Dict:
    """Group rows and compute per-group metrics.

    Returns a dict keyed by (condition, k, registry_size) with metrics.
    """
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r["condition"], r.get("k"), r["registry_size"])
        groups[key].append(r)

    summary = {}
    for key, grp in groups.items():
        n = len(grp)
        summary[key] = {
            "n": n,
            "tool_accuracy": sum(r["tool_correct"] for r in grp) / n,
            "args_accuracy": sum(r["args_correct"] for r in grp if r["correct_tool"] is not None)
                             / max(1, sum(1 for r in grp if r["correct_tool"] is not None)),
            "full_accuracy": sum(r["full_success"] for r in grp) / n,
            "mean_prompt_tokens": np.mean([r["prompt_tokens"] for r in grp]),
            "mean_total_tokens": np.mean([r["total_tokens"] for r in grp]),
            "mean_latency_s": np.mean([r["latency_s"] for r in grp]),
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────── #
# Console table                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #


def print_summary(summary: Dict, model: str) -> None:
    print(f"\n{'='*80}")
    print(f"  Tool Selection Evaluation Results  |  Model: {model}")
    print(f"{'='*80}")
    header = f"{'Condition':<20} {'k':>4} {'Size':>6} {'n':>5} {'Tool%':>7} {'Args%':>7} {'Full%':>7} {'PromptTok':>10} {'Lat(s)':>8}"
    print(header)
    print("-" * len(header))

    # Sort: naive first, then retrieval by k, within each size ascending
    def sort_key(item):
        (cond, k, size), _ = item
        return (size, 0 if cond == "naive" else 1, k or 0)

    for (cond, k, size), m in sorted(summary.items(), key=sort_key):
        k_str = str(k) if k is not None else "all"
        print(
            f"{cond:<20} {k_str:>4} {size:>6} {m['n']:>5}"
            f" {m['tool_accuracy']*100:>7.1f}"
            f" {m['args_accuracy']*100:>7.1f}"
            f" {m['full_accuracy']*100:>7.1f}"
            f" {m['mean_prompt_tokens']:>10.0f}"
            f" {m['mean_latency_s']:>8.2f}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────── #
# Plots                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3f52",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#2d3343",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": "#1e2235",
    "legend.edgecolor": "#3a3f52",
    "font.family":      "DejaVu Sans",
}

_COLORS = {
    "naive":   "#e05c5c",
    "k3":      "#58a6ff",
    "k5":      "#3fb950",
    "k10":     "#d2a8ff",
}


def _apply_style():
    plt.rcParams.update(_STYLE)


def plot_accuracy_vs_registry_size(summary: Dict, out_path: Path, k_values: List[int]) -> None:
    """Plot A: full accuracy vs registry size for naive and each retrieval k."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sizes = sorted({key[2] for key in summary})

    # Naive
    naive_acc = []
    for size in sizes:
        key = ("naive", None, size)
        naive_acc.append(summary.get(key, {}).get("full_accuracy", float("nan")) * 100)
    ax.plot(sizes, naive_acc, "o-", color=_COLORS["naive"], lw=2.5, ms=8,
            label="Naive (all tools)", zorder=5)

    # Retrieval per k
    for k in k_values:
        c = _COLORS.get(f"k{k}", "#ffb347")
        retr_acc = []
        for size in sizes:
            key = ("retrieval", k, size)
            retr_acc.append(summary.get(key, {}).get("full_accuracy", float("nan")) * 100)
        ax.plot(sizes, retr_acc, "s--", color=c, lw=2, ms=7, label=f"Retrieval k={k}")

    ax.set_xlabel("Registry size (# tools)", fontsize=12)
    ax.set_ylabel("Full-task success rate (%)", fontsize=12)
    ax.set_title("Accuracy vs Tool Registry Size", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(sizes)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(framealpha=0.9)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_accuracy_vs_token_cost(summary: Dict, out_path: Path, k_values: List[int]) -> None:
    """Plot B: full accuracy vs mean prompt tokens (retrieval k sweep)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sizes = sorted({key[2] for key in summary})
    markers = ["o", "s", "^", "D", "v"]

    for i, size in enumerate(sizes):
        xs, ys, labels = [], [], []

        # Naive point
        key = ("naive", None, size)
        if key in summary:
            m = summary[key]
            xs.append(m["mean_prompt_tokens"])
            ys.append(m["full_accuracy"] * 100)
            labels.append(f"Naive (size={size})")

        # Retrieval points
        for k in k_values:
            key = ("retrieval", k, size)
            if key in summary:
                m = summary[key]
                xs.append(m["mean_prompt_tokens"])
                ys.append(m["full_accuracy"] * 100)
                labels.append(f"k={k} (size={size})")

        color_cycle = [_COLORS["naive"]] + [_COLORS.get(f"k{k}", "#ffb347") for k in k_values]
        for x, y, lbl, c, mk in zip(xs, ys, labels, color_cycle, markers):
            ax.scatter(x, y, color=c, marker=mk, s=120, label=lbl, zorder=5)

    ax.set_xlabel("Mean prompt tokens per task", fontsize=12)
    ax.set_ylabel("Full-task success rate (%)", fontsize=12)
    ax.set_title("Accuracy vs Token Cost Trade-off", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True)

    # Annotate: higher-right = better
    ax.text(
        0.98, 0.04,
        "← fewer tokens    more tokens →\n↑ higher accuracy is better",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8, alpha=0.6,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_token_consumption(summary: Dict, out_path: Path, k_values: List[int]) -> None:
    """Plot C: mean prompt tokens per task (bar chart, naive vs retrieval k)."""
    _apply_style()

    sizes = sorted({key[2] for key in summary})
    conditions = ["naive"] + [f"k{k}" for k in k_values]
    n_cond = len(conditions)
    x = np.arange(len(sizes))
    bar_w = 0.8 / n_cond

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, cond in enumerate(conditions):
        if cond == "naive":
            tokens = [summary.get(("naive", None, s), {}).get("mean_prompt_tokens", 0) for s in sizes]
            label = "Naive"
            color = _COLORS["naive"]
        else:
            k = int(cond[1:])
            tokens = [summary.get(("retrieval", k, s), {}).get("mean_prompt_tokens", 0) for s in sizes]
            label = f"Retrieval k={k}"
            color = _COLORS.get(cond, "#ffb347")

        offset = (i - n_cond / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, tokens, bar_w * 0.9, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Registry size (# tools)", fontsize=12)
    ax.set_ylabel("Mean prompt tokens per task", fontsize=12)
    ax.set_title("Prompt Token Consumption by Condition", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #


def main() -> None:
    here = Path(__file__).parent
    default_results_dir = here / "results"

    parser = argparse.ArgumentParser(description="Analyze tool-selection eval results")
    parser.add_argument("--results-dir", default=str(default_results_dir))
    parser.add_argument("--results-file", default=None,
                        help="Path to a specific JSONL results file (overrides --results-dir)")
    parser.add_argument("--out-dir", default=str(default_results_dir))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = find_latest_results(Path(args.results_dir))

    print(f"Reading results from: {results_path}")
    rows = load_results(results_path)
    print(f"Loaded {len(rows)} rows.")

    if not rows:
        print("No rows found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Detect model from first row
    model = rows[0].get("model", "unknown")

    summary = aggregate(rows)
    k_values = sorted({key[1] for key in summary if key[1] is not None})

    print_summary(summary, model)

    if HAS_MPL:
        print("Generating plots...")
        plot_accuracy_vs_registry_size(
            summary,
            out_dir / "plot_accuracy_vs_registry_size.png",
            k_values,
        )
        plot_accuracy_vs_token_cost(
            summary,
            out_dir / "plot_accuracy_vs_token_cost.png",
            k_values,
        )
        plot_token_consumption(
            summary,
            out_dir / "plot_token_consumption.png",
            k_values,
        )
        print("\nAll plots saved.")
    else:
        print("\nInstall matplotlib for plots: pip install matplotlib")


if __name__ == "__main__":
    main()
