"""Benchmark harness: run every study under a fixed seed and write the results.

``python -m causa.bench`` produces ``benchmarks/results.json`` and ``benchmarks/results.md``. Every number in
the README is copied from that file by ``--update-readme`` (and ``tests/test_docs_consistency.py`` fails if
the two disagree). ``--check`` compares the results with ``benchmarks/thresholds.json`` and exits non-zero
on a violation, which is how CI uses the benchmark as a quality gate. ``--quick`` shrinks every study for
smoke tests; its numbers are not the published ones.

Work runs in at most ``--workers`` processes (default 2), each limited to one BLAS and OpenMP thread, so the
benchmark shares a laptop politely.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from causa import __version__
from causa.render import render_markdown, render_readme_section
from causa.studies.ate import ATEStudyConfig, run_ate_study, true_ate
from causa.studies.cate import CATEStudyConfig, run_cate_study
from causa.studies.experiments import (
    CupedStudyConfig,
    SequentialStudyConfig,
    SRMStudyConfig,
    cuped_study,
    sequential_study,
    srm_study,
)
from causa.studies.serving import ServingStudyConfig, serving_study

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = "results.json"
MARKDOWN_FILE = "results.md"
THRESHOLDS_FILE = "thresholds.json"
README_BEGIN = "<!-- RESULTS:BEGIN -->"
README_END = "<!-- RESULTS:END -->"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _environment(workers: int, quick: bool) -> dict[str, Any]:
    return {
        "causa": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
        "git_commit": _git_commit(),
        "workers": workers,
        "quick": quick,
    }


def _configs(seed: int, quick: bool) -> dict[str, Any]:
    """Study sizes: the published run, or a much smaller smoke-test run."""
    if quick:
        return {
            "ate": [ATEStudyConfig("moderate_overlap", 1, 1.0, 24, 2000, 50, seed)],
            "ate_draws": 400_000,
            "cate": CATEStudyConfig(seeds=(0, 1), n_train=8_000, n_holdout=8_000, seed=seed),
            "cuped": CupedStudyConfig(reps=200, seed=seed),
            "sequential": SequentialStudyConfig(reps=200, seed=seed),
            "srm": SRMStudyConfig(reps=1000, bias_reps=20, seed=seed),
            "serving": ServingStudyConfig(
                experiment_rows=4_000,
                experiment_requests=3,
                customers=100,
                targeting_requests=3,
                ate_rows=1_000,
                ate_boot=30,
                ate_requests=2,
                train_rows=3_000,
                seed=seed,
            ),
        }
    return {
        "ate": [
            ATEStudyConfig("moderate_overlap", 1, 1.0, seed=seed),
            ATEStudyConfig("weak_overlap", 2, 1.8, seed=seed),
        ],
        "ate_draws": 4_000_000,
        "cate": CATEStudyConfig(seed=seed),
        "cuped": CupedStudyConfig(seed=seed),
        "sequential": SequentialStudyConfig(seed=seed),
        "srm": SRMStudyConfig(seed=seed),
        "serving": ServingStudyConfig(seed=seed),
    }


def _timed(seconds: dict[str, float], key: str, fn: Any, *args: Any) -> Any:
    started = time.perf_counter()
    out = fn(*args)
    seconds[key] = round(time.perf_counter() - started, 2)
    print(f"  {key}: {seconds[key]} s", flush=True)
    return out


def run_benchmark(seed: int, workers: int, quick: bool) -> dict[str, Any]:
    """Run every study and return the results document."""
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = _configs(seed, quick)
    seconds: dict[str, float] = {}
    started = time.perf_counter()
    truth = _timed(seconds, "true_ate", true_ate, seed, cfg["ate_draws"])
    scenarios = {
        c.scenario: _timed(seconds, f"ate_{c.scenario}", run_ate_study, c, truth["ate"], workers)
        for c in cfg["ate"]
    }
    results = {
        "seed": seed,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment(workers, quick),
        "ate": {"truth": truth, "scenarios": scenarios},
        "cate": _timed(seconds, "cate", run_cate_study, cfg["cate"], workers),
        "cuped": _timed(seconds, "cuped", cuped_study, cfg["cuped"]),
        "sequential": _timed(seconds, "sequential", sequential_study, cfg["sequential"]),
        "srm": _timed(seconds, "srm", srm_study, cfg["srm"]),
        "serving": _timed(seconds, "serving", serving_study, cfg["serving"]),
    }
    seconds["total"] = round(time.perf_counter() - started, 2)
    results["timings_seconds"] = seconds
    return results


def write_results(results: dict[str, Any], out_dir: Path) -> None:
    """Write the JSON document and its Markdown rendering."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RESULTS_FILE).write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / MARKDOWN_FILE).write_text(render_markdown(results))


def update_readme(results: dict[str, Any], readme: Path) -> None:
    """Replace the block between the README markers with the rendered results section."""
    text = readme.read_text()
    start, end = text.index(README_BEGIN) + len(README_BEGIN), text.index(README_END)
    readme.write_text(text[:start] + "\n" + render_readme_section(results) + "\n" + text[end:])


def lookup(results: dict[str, Any], path: str) -> float:
    """Value at a dotted path such as ``cuped.variance_reduction``."""
    node: Any = results
    for part in path.split("."):
        node = node[part]
    return float(node)


def check_thresholds(results: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Print one line per check and return whether every check passed."""
    passed = True
    for check in thresholds["checks"]:
        try:
            value = lookup(results, check["path"])
        except (KeyError, TypeError):
            print(f"FAIL {check['path']} missing from results")
            passed = False
            continue
        ok = ("min" not in check or value >= check["min"]) and ("max" not in check or value <= check["max"])
        bounds = " and ".join(f"{op} {check[k]}" for k, op in (("min", ">="), ("max", "<=")) if k in check)
        print(f"{'PASS' if ok else 'FAIL'} {check['path']} = {value:.4f} (want {bounds})")
        passed &= ok
    print("quality gate:", "PASS" if passed else "FAIL")
    return passed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="causa benchmark harness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2, help="worker processes (each single-threaded)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "benchmarks")
    parser.add_argument("--quick", action="store_true", help="small smoke-test sizes, not the published run")
    parser.add_argument("--check", action="store_true", help="after running (or alone with --no-run), gate")
    parser.add_argument(
        "--no-run", action="store_true", help="skip the run and use the existing results.json"
    )
    parser.add_argument(
        "--update-readme", action="store_true", help="copy the measured tables into README.md"
    )
    args = parser.parse_args(argv)
    if not args.no_run:
        print(f"causa benchmark (seed {args.seed}, {args.workers} workers{', quick' if args.quick else ''})")
        write_results(run_benchmark(args.seed, args.workers, args.quick), args.out)
        print(f"wrote {args.out / RESULTS_FILE} and {args.out / MARKDOWN_FILE}")
    results = json.loads((args.out / RESULTS_FILE).read_text())
    if args.update_readme:
        update_readme(results, PROJECT_ROOT / "README.md")
        print("README results section updated")
    if args.check:
        thresholds = json.loads((PROJECT_ROOT / "benchmarks" / THRESHOLDS_FILE).read_text())
        return 0 if check_thresholds(results, thresholds) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
