#!/usr/bin/env python3
"""Run and summarize a cross-benchmark PPG suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppg.eval.suite import (  # noqa: E402
    discover_result_files,
    latest_records_by_benchmark,
    render_markdown_summary,
    summarize_records,
    write_summary_files,
)


DEFAULT_BENCHMARKS = (
    "gsm8k",
    "drop",
    "hotpotqa",
    "truthfulqa",
    "arc_challenge",
    "mmlu",
)

PROFILES = {
    "smoke": {
        "train_n": 20,
        "val_n": 20,
        "test_n": 50,
        "warmup": 20,
        "train_ep": 60,
        "finetune": 20,
        "ppg_path_candidates": 20,
        "ppg_ensemble_paths": 1,
    },
    "standard": {
        "train_n": 100,
        "val_n": 100,
        "test_n": 300,
        "warmup": 200,
        "train_ep": 1000,
        "finetune": 200,
        "ppg_path_candidates": 80,
        "ppg_ensemble_paths": 3,
    },
    "push": {
        "train_n": 200,
        "val_n": 200,
        "test_n": 500,
        "warmup": 300,
        "train_ep": 2500,
        "finetune": 400,
        "ppg_path_candidates": 120,
        "ppg_ensemble_paths": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPG across a benchmark suite and summarize method rankings.",
    )
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS),
                        help="Comma-separated benchmark list")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard",
                        help="Budget profile")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--reflection-model", default="gpt-4o")
    parser.add_argument("--mmlu-subject", default="all")
    parser.add_argument("--production", action="store_true",
                        help="Forward --production to run_benchmark.py")
    parser.add_argument("--run-mipro", action="store_true",
                        help="Include MIPROv2 in each benchmark run")
    parser.add_argument("--run-gepa", action="store_true",
                        help="Include GEPA in each benchmark run")
    parser.add_argument("--include-ppg", action="store_true",
                        help="Include PPG when external baselines are requested")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-dir", default=".lm_cache")
    parser.add_argument("--output-dir", default="results_suite")
    parser.add_argument("--log-root", default="ppg_logs/suite")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--k-samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--parse-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-n", type=int, default=None)
    parser.add_argument("--val-n", type=int, default=None)
    parser.add_argument("--test-n", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--train-ep", type=int, default=None)
    parser.add_argument("--finetune", type=int, default=None)
    parser.add_argument("--ppg-path-candidates", type=int, default=None)
    parser.add_argument("--ppg-ensemble-paths", type=int, default=None)
    parser.add_argument("--ppg-calibration-patience", type=int, default=0)
    parser.add_argument("--diagnostic-report", action="store_true",
                        help="Forward --diagnostic-report to each benchmark run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Only summarize existing result JSON files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a benchmark when a result for model+benchmark already exists")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue remaining benchmarks if one command fails")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    benchmarks = _parse_csv(args.benchmarks)

    if not args.summarize_only:
        for benchmark in benchmarks:
            if args.skip_existing and _has_existing_result(args.output_dir, benchmark, args.model):
                print(f"[suite] skip existing: {benchmark}")
                continue
            cmd = build_command(args, benchmark, repo_root)
            print("[suite] " + " ".join(cmd))
            if args.dry_run:
                continue
            completed = subprocess.run(cmd, cwd=repo_root, check=False)
            if completed.returncode != 0:
                print(f"[suite] failed: {benchmark} exited {completed.returncode}", file=sys.stderr)
                if not args.continue_on_error:
                    return completed.returncode

    if args.dry_run:
        return 0

    result_files = discover_result_files(
        args.output_dir,
        benchmarks=benchmarks,
        model=args.model,
    )
    records = latest_records_by_benchmark(result_files)
    summary = summarize_records(records)
    prefix = f"suite_{args.model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    paths = write_summary_files(summary, args.output_dir, prefix=prefix)
    print()
    print(render_markdown_summary(summary))
    print(f"[suite] wrote {paths['json']}")
    print(f"[suite] wrote {paths['markdown']}")
    return 0


def build_command(args: argparse.Namespace, benchmark: str, repo_root: Path) -> list[str]:
    budget = _budget(args)
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_benchmark.py"),
        benchmark,
        "--provider", args.provider,
        "--model", args.model,
        "--train-n", str(budget["train_n"]),
        "--val-n", str(budget["val_n"]),
        "--test-n", str(budget["test_n"]),
        "--warmup", str(budget["warmup"]),
        "--train-ep", str(budget["train_ep"]),
        "--finetune", str(budget["finetune"]),
        "--workers", str(args.workers),
        "--temperature", str(args.temperature),
        "--sample-temperature", str(args.sample_temperature),
        "--k-samples", str(args.k_samples),
        "--timeout", str(args.timeout),
        "--max-retries", str(args.max_retries),
        "--parse-retries", str(args.parse_retries),
        "--seed", str(args.seed),
        "--ppg-calibration", "val_path",
        "--ppg-path-candidates", str(budget["ppg_path_candidates"]),
        "--ppg-calibration-patience", str(args.ppg_calibration_patience),
        "--ppg-ensemble-paths", str(budget["ppg_ensemble_paths"]),
        "--output-dir", args.output_dir,
        "--log-dir", str(Path(args.log_root) / f"{benchmark}_{time.strftime('%Y%m%d_%H%M%S')}"),
    ]
    if benchmark == "mmlu":
        cmd.extend(["--mmlu-subject", args.mmlu_subject])
    if args.production:
        cmd.append("--production")
    if args.diagnostic_report:
        cmd.append("--diagnostic-report")
    if args.no_cache:
        cmd.append("--no-cache")
    else:
        cmd.extend(["--cache-dir", args.cache_dir])
    if args.run_mipro:
        cmd.append("--run-mipro")
    if args.run_gepa:
        cmd.extend(["--run-gepa", "--reflection-model", args.reflection_model])
    if (args.run_mipro or args.run_gepa) and args.include_ppg:
        cmd.append("--include-ppg")
    return cmd


def _budget(args: argparse.Namespace) -> dict[str, int]:
    budget = dict(PROFILES[args.profile])
    overrides = {
        "train_n": args.train_n,
        "val_n": args.val_n,
        "test_n": args.test_n,
        "warmup": args.warmup,
        "train_ep": args.train_ep,
        "finetune": args.finetune,
        "ppg_path_candidates": args.ppg_path_candidates,
        "ppg_ensemble_paths": args.ppg_ensemble_paths,
    }
    for key, value in overrides.items():
        if value is not None:
            budget[key] = value
    return budget


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _has_existing_result(output_dir: str | Path, benchmark: str, model: str) -> bool:
    return bool(discover_result_files(output_dir, benchmarks=[benchmark], model=model))


if __name__ == "__main__":
    raise SystemExit(main())
