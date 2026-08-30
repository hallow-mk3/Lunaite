"""
research/tool_selection/run_eval.py
====================================
Evaluation runner for the tool-selection study.

Runs both conditions (naive and retrieval) across all registry sizes
and k values, logs per-task results as JSONL.

Usage
-----
    python research/tool_selection/run_eval.py [OPTIONS]

Options
-------
    --model         Ollama model name          (default: qwen3:8b)
    --registry-sizes  Space-separated ints     (default: 10 25 50)
    --k-values        Space-separated ints     (default: 3 5 10)
    --tasks-file      Path to eval_tasks.jsonl (default: ./eval_tasks.jsonl)
    --out-dir         Output directory         (default: ./results)
    --dry-run         Run only first 5 tasks per condition (for testing)
    --base-url        Ollama base URL          (default: http://localhost:11434/v1)

Output
------
    results/raw_results_<timestamp>.jsonl   — one JSON line per task×condition×size

Each line fields:
    task_id, case_type, query, correct_tool, correct_args,
    registry_size, condition, k,
    tools_shown, tool_called, arguments_generated, no_tool_called,
    tool_correct, args_correct, full_success,
    prompt_tokens, completion_tokens, total_tokens, latency_s,
    execution_result, execution_error, execution_success,
    selector_name, model, error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── repo root on sys.path ──────────────────────────────────────────────── #
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from lunaite.harness import Harness, HarnessResult
from lunaite.selection import NaiveSelector, RetrievalSelector
from lunaite.tools import ToolRegistry

from research.tool_selection.tools_library import build_registry

# ─────────────────────────────────────────────────────────────────────────── #
# Evaluation helpers                                                            #
# ─────────────────────────────────────────────────────────────────────────── #


def _normalise_args(args: Optional[Dict]) -> Optional[Dict]:
    """Coerce argument values to float for numeric comparison."""
    if args is None:
        return None
    out = {}
    for k, v in args.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, list):
            out[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
        else:
            out[k] = v
    return out


def _args_correct(generated: Optional[Dict], expected: Optional[Dict]) -> bool:
    """Return True if generated args match expected (numeric-tolerant)."""
    if expected is None:
        return generated is None or generated == {}
    if generated is None:
        return False
    gen_n = _normalise_args(generated)
    exp_n = _normalise_args(expected)
    if set(gen_n.keys()) != set(exp_n.keys()):
        return False
    for k in exp_n:
        g, e = gen_n[k], exp_n[k]
        if isinstance(e, float) and isinstance(g, float):
            if abs(g - e) > max(1e-3 * abs(e), 1e-6):
                return False
        elif isinstance(e, list) and isinstance(g, list):
            if len(g) != len(e):
                return False
            for gi, ei in zip(g, e):
                if isinstance(ei, float) and isinstance(gi, float):
                    if abs(gi - ei) > max(1e-3 * abs(ei), 1e-6):
                        return False
                elif gi != ei:
                    return False
        elif g != e:
            return False
    return True


def evaluate_result(result: HarnessResult, task: Dict) -> Dict:
    """Compute correctness metrics for one task result."""
    correct_tool = task["correct_tool"]
    correct_args = task["correct_args"]

    # Tool selection correct?
    if correct_tool is None:
        # Correct behaviour = no tool called
        tool_correct = result.no_tool_called
    else:
        tool_correct = result.tool_called == correct_tool

    # Argument correctness (only meaningful if tool was correct)
    if tool_correct and correct_tool is not None:
        args_correct = _args_correct(result.arguments_generated, correct_args)
    else:
        args_correct = False

    # Full success = tool correct AND args correct (or no-tool correct)
    if correct_tool is None:
        full_success = tool_correct
    else:
        full_success = tool_correct and args_correct

    return {
        "tool_correct": tool_correct,
        "args_correct": args_correct,
        "full_success": full_success,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Main runner                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #


def run_eval(
    model: str,
    registry_sizes: List[int],
    k_values: List[int],
    tasks: List[Dict],
    out_path: Path,
    base_url: str = "http://localhost:11434/v1",
    dry_run: bool = False,
) -> None:
    if dry_run:
        tasks = tasks[:5]
        print(f"[DRY RUN] Using {len(tasks)} tasks only.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_selector = RetrievalSelector(k=max(k_values))  # model loaded once

    total_conditions = len(registry_sizes) * (1 + len(k_values))
    total_calls = len(tasks) * total_conditions
    print(
        f"Running {len(tasks)} tasks × {total_conditions} conditions"
        f" = {total_calls} LLM calls. Model: {model}"
    )
    print(f"Registry sizes: {registry_sizes}  |  k values: {k_values}")
    print(f"Output: {out_path}\n")

    call_num = 0
    t_run_start = time.perf_counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for size in registry_sizes:
            registry = build_registry(size)
            print(f"\n{'='*60}")
            print(f"Registry size: {size} tools")
            print(f"{'='*60}")

            # ── Naive condition ─────────────────────────────────────── #
            naive_harness = Harness(
                registry=registry,
                selector=NaiveSelector(),
                model=model,
                base_url=base_url,
            )
            print(f"\n[Naive] registry={size}")
            for task in tasks:
                call_num += 1
                result = naive_harness.run(task["query"])
                metrics = evaluate_result(result, task)
                row = {
                    "task_id": task["id"],
                    "case_type": task["case_type"],
                    "query": task["query"],
                    "correct_tool": task["correct_tool"],
                    "correct_args": task["correct_args"],
                    "registry_size": size,
                    "condition": "naive",
                    "k": None,
                    **metrics,
                    **{k: v for k, v in result.to_dict().items()
                       if k not in ("query", "model", "selector_name")},
                    "selector_name": result.selector_name,
                    "model": result.model,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                elapsed = time.perf_counter() - t_run_start
                eta = (elapsed / call_num) * (total_calls - call_num)
                status = "[OK]" if metrics["full_success"] else "[FAIL]"
                print(
                    f"  [{call_num}/{total_calls}] {status} {task['id']:<6}"
                    f" tool={result.tool_called or '(none)':<30}"
                    f" tokens={result.prompt_tokens} lat={result.latency_s:.1f}s"
                    f" ETA={eta/60:.1f}m"
                )

            # ── Retrieval conditions (one per k) ────────────────────── #
            for k in k_values:
                retrieval_harness = Harness(
                    registry=registry,
                    selector=retrieval_selector,
                    model=model,
                    base_url=base_url,
                )
                print(f"\n[Retrieval k={k}] registry={size}")
                for task in tasks:
                    call_num += 1
                    result = retrieval_harness.run(task["query"], selector_kwargs={"k": k})
                    metrics = evaluate_result(result, task)
                    row = {
                        "task_id": task["id"],
                        "case_type": task["case_type"],
                        "query": task["query"],
                        "correct_tool": task["correct_tool"],
                        "correct_args": task["correct_args"],
                        "registry_size": size,
                        "condition": "retrieval",
                        "k": k,
                        **metrics,
                        **{k2: v for k2, v in result.to_dict().items()
                           if k2 not in ("query", "model", "selector_name")},
                        "selector_name": result.selector_name,
                        "model": result.model,
                    }
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    elapsed = time.perf_counter() - t_run_start
                    eta = (elapsed / call_num) * (total_calls - call_num)
                    status = "[OK]" if metrics["full_success"] else "[FAIL]"
                    print(
                        f"  [{call_num}/{total_calls}] {status} {task['id']:<6}"
                        f" tool={result.tool_called or '(none)':<30}"
                        f" tokens={result.prompt_tokens} lat={result.latency_s:.1f}s"
                        f" ETA={eta/60:.1f}m"
                    )

    total_time = time.perf_counter() - t_run_start
    print(f"\nDone. {call_num} calls in {total_time/60:.1f} min -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────── #
# CLI                                                                           #
# ─────────────────────────────────────────────────────────────────────────── #


def main() -> None:
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Run tool-selection eval")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--registry-sizes", nargs="+", type=int, default=[10, 25, 50])
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--tasks-file", default=str(here / "eval_tasks.jsonl"))
    parser.add_argument("--out-dir", default=str(here / "results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    args = parser.parse_args()

    # Load tasks
    tasks_path = Path(args.tasks_file)
    if not tasks_path.exists():
        print(f"ERROR: tasks file not found: {tasks_path}", file=sys.stderr)
        sys.exit(1)
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_dir) / f"raw_results_{timestamp}.jsonl"

    run_eval(
        model=args.model,
        registry_sizes=args.registry_sizes,
        k_values=args.k_values,
        tasks=tasks,
        out_path=out_path,
        base_url=args.base_url,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
