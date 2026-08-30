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
    --model           Model name                 (default: qwen2.5:7b)
    --registry-sizes  Space-separated ints       (default: 10 25 50)
    --k-values        Space-separated ints       (default: 3 5 10)
    --tasks-file      Path to eval_tasks.jsonl   (default: ./eval_tasks.jsonl)
    --out-dir         Output directory           (default: ./results)
    --dry-run         Run only first 5 tasks per condition (for testing)
    --base-url        OpenAI-compatible base URL (default: http://localhost:11434/v1)
    --api-key         API Key if required        (default: env OPENAI_API_KEY)
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
            out[k] = [float(x) if isinstance(x, (int, float)) else str(x).lower().strip() for x in v]
        elif isinstance(v, str):
            out[k] = v.strip().lower()
        else:
            out[k] = v
    return out


def evaluate_result(result: HarnessResult, task: Dict) -> Dict[str, bool]:
    """Score a single HarnessResult against ground truth task definition."""
    expected_tool: Optional[str] = task["correct_tool"]
    expected_args: Optional[Dict] = task["correct_args"]

    # 1. Tool selection correctness
    if expected_tool is None:
        # Negative / trick case: correct behavior is to call NO tool
        tool_correct = (result.tool_called is None) or result.no_tool_called
    else:
        tool_correct = (result.tool_called == expected_tool)

    # 2. Arguments correctness
    if expected_tool is None:
        args_correct = True
    elif not tool_correct:
        args_correct = False
    elif expected_args is None:
        args_correct = True
    else:
        gen_norm = _normalise_args(result.arguments_generated)
        exp_norm = _normalise_args(expected_args)
        if gen_norm is None:
            args_correct = False
        else:
            args_correct = True
            for k, exp_val in exp_norm.items():
                if k not in gen_norm:
                    args_correct = False
                    break
                gen_val = gen_norm[k]
                if isinstance(exp_val, float) and isinstance(gen_val, float):
                    if abs(exp_val - gen_val) > 1e-3:
                        args_correct = False
                        break
                elif gen_val != exp_val:
                    args_correct = False
                    break

    # 3. Full task success
    full_success = tool_correct and args_correct

    return {
        "tool_correct": tool_correct,
        "args_correct": args_correct,
        "full_success": full_success,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Main eval loop                                                                #
# ─────────────────────────────────────────────────────────────────────────── #


def run_eval(
    model: str,
    registry_sizes: List[int],
    k_values: List[int],
    tasks: List[Dict],
    out_path: Path,
    base_url: str = "http://localhost:11434/v1",
    api_key: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Run full evaluation matrix across conditions, sizes, and tasks."""
    if dry_run:
        tasks = tasks[:5]
        print(f"[DRY RUN] Truncated to {len(tasks)} tasks.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_conditions = 1 + len(k_values)
    total_calls = len(registry_sizes) * total_conditions * len(tasks)
    call_num = 0

    print(f"\nStarting evaluation:")
    print(f"  Model:           {model}")
    print(f"  Base URL:        {base_url}")
    print(f"  Registry sizes:  {registry_sizes}")
    print(f"  Retrieval k:     {k_values}")
    print(f"  Tasks count:     {len(tasks)}")
    print(f"  Total LLM calls: {total_calls}")
    print(f"  Output path:     {out_path}\n")

    t_run_start = time.perf_counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for size in registry_sizes:
            registry = build_registry(size)
            naive_selector = NaiveSelector(registry)
            retrieval_selector = RetrievalSelector(registry)

            # ── 1. Naive condition ──────────────────────────────────── #
            naive_harness = Harness(
                registry=registry,
                selector=naive_selector,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )

            # Connection check
            if call_num == 0 and not naive_harness.check_connection():
                print(
                    f"WARNING: Could not connect to LLM endpoint at {base_url}/models\n"
                    f"Please ensure Ollama is running (`ollama run {model}`) or verify `--base-url`.\n"
                )

            print(f"\n[Naive Selection] registry_size={size}")
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
                eta = (elapsed / max(1, call_num)) * (total_calls - call_num)
                status = "[OK]" if metrics["full_success"] else "[FAIL]"
                print(
                    f"  [{call_num}/{total_calls}] {status} {task['id']:<6}"
                    f" tool={result.tool_called or '(none)':<30}"
                    f" tokens={result.prompt_tokens} lat={result.latency_s:.1f}s"
                    f" ETA={eta/60:.1f}m"
                )

            # ── 2. Retrieval conditions (per k) ─────────────────────── #
            for k in k_values:
                retrieval_harness = Harness(
                    registry=registry,
                    selector=retrieval_selector,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )
                print(f"\n[Retrieval k={k}] registry_size={size}")
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
                    eta = (elapsed / max(1, call_num)) * (total_calls - call_num)
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
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--registry-sizes", nargs="+", type=int, default=[10, 25, 50])
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--tasks-file", default=str(here / "eval_tasks.jsonl"))
    parser.add_argument("--out-dir", default=str(here / "results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

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
        api_key=args.api_key,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
