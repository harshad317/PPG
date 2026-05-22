"""Tests for cross-benchmark suite summaries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pytest

from ppg.eval.suite import (
    discover_result_files,
    latest_records_by_benchmark,
    load_suite_record,
    render_markdown_summary,
    summarize_records,
    write_summary_files,
)


def write_result(path: Path, benchmark: str, results: list[dict], *, model: str = "gpt-x") -> Path:
    payload = {
        "benchmark": benchmark,
        "model": model,
        "seed": 0,
        "results": results,
        "winner": max(results, key=lambda r: r["task_accuracy"])["name"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def row(name: str, score: float, *, tokens: float = 100.0, calls: int = 10) -> dict:
    return {
        "name": name,
        "task_accuracy": score,
        "std_task": 0.1,
        "mean_tokens": tokens,
        "mean_constraint": 0.0,
        "lm_calls": calls,
        "n_examples": 5,
    }


def test_load_suite_record(tmp_path: Path):
    path = write_result(tmp_path / "gsm.json", "gsm8k", [row("ppg", 0.8), row("gepa", 0.7)])

    record = load_suite_record(path)

    assert record.benchmark == "gsm8k"
    assert record.model == "gpt-x"
    assert record.methods["ppg"]["task_accuracy"] == pytest.approx(0.8)


def test_discover_and_latest_records_by_benchmark(tmp_path: Path):
    old = write_result(tmp_path / "old.json", "gsm8k", [row("ppg", 0.7)])
    new = write_result(tmp_path / "new.json", "gsm8k", [row("ppg", 0.8)])
    other = write_result(tmp_path / "hotpot.json", "hotpotqa", [row("ppg", 0.6)])
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    os.utime(other, (3, 3))

    files = discover_result_files(tmp_path, benchmarks=["gsm8k", "hotpotqa"], model="gpt-x")
    records = latest_records_by_benchmark(files)

    assert [r.benchmark for r in records] == ["gsm8k", "hotpotqa"]
    assert records[0].path.name == "new.json"


def test_summarize_records_computes_suite_rank_and_ppg_delta(tmp_path: Path):
    gsm = write_result(
        tmp_path / "gsm.json",
        "gsm8k",
        [row("ppg", 0.80), row("gepa", 0.75), row("base_model", 0.40)],
    )
    hotpot = write_result(
        tmp_path / "hotpot.json",
        "hotpotqa",
        [row("ppg", 0.78), row("gepa", 0.80), row("base_model", 0.20)],
    )

    summary = summarize_records([load_suite_record(gsm), load_suite_record(hotpot)])
    ppg = next(m for m in summary["methods"] if m["name"] == "ppg")
    gepa = next(m for m in summary["methods"] if m["name"] == "gepa")

    assert ppg["mean_score"] == pytest.approx(0.79)
    assert ppg["average_rank"] == pytest.approx(1.5)
    assert ppg["mean_delta_vs_best_non_ppg"] == pytest.approx(0.015)
    assert gepa["mean_delta_vs_ppg"] == pytest.approx(-0.015)


def test_summarize_records_counts_score_ties_as_wins(tmp_path: Path):
    result = write_result(
        tmp_path / "mmlu.json",
        "mmlu",
        [row("ppg", 0.87), row("miprov2", 0.87), row("gepa", 0.86)],
    )

    summary = summarize_records([load_suite_record(result)])
    ppg = next(m for m in summary["methods"] if m["name"] == "ppg")
    miprov2 = next(m for m in summary["methods"] if m["name"] == "miprov2")
    per_benchmark = summary["per_benchmark"][0]

    assert ppg["wins"] == 1
    assert miprov2["wins"] == 1
    assert ppg["average_rank"] == pytest.approx(1.0)
    assert miprov2["average_rank"] == pytest.approx(1.0)
    assert per_benchmark["winner"] == "miprov2, ppg"
    assert per_benchmark["winners"] == ["miprov2", "ppg"]


def test_render_and_write_summary(tmp_path: Path):
    result = write_result(tmp_path / "gsm.json", "gsm8k", [row("ppg", 0.8), row("gepa", 0.7)])
    summary = summarize_records([load_suite_record(result)])

    text = render_markdown_summary(summary)
    paths = write_summary_files(summary, tmp_path, prefix="suite_test")

    assert "PPG Suite Summary" in text
    assert paths["json"].exists()
    assert paths["markdown"].exists()


def test_run_suite_build_command_includes_suite_controls():
    module = _load_run_suite_module()
    args = argparse.Namespace(
        profile="smoke",
        provider="openai",
        model="gpt-4.1-mini",
        reflection_model="gpt-4o",
        mmlu_subject="all",
        production=True,
        few_shot=True,
        run_mipro=True,
        run_gepa=True,
        include_ppg=True,
        no_cache=False,
        cache_dir=".lm_cache",
        output_dir="results_suite",
        log_root="ppg_logs/suite",
        workers=8,
        temperature=0.0,
        sample_temperature=0.8,
        k_samples=5,
        timeout=90.0,
        max_retries=6,
        parse_retries=5,
        seed=0,
        train_n=None,
        val_n=None,
        test_n=None,
        warmup=None,
        train_ep=None,
        finetune=None,
        ppg_path_candidates=33,
        ppg_ensemble_paths=3,
        ppg_calibration_patience=0,
        ppg_calibration_execution="deployment",
        diagnostic_report=True,
    )

    cmd = module.build_command(args, "hotpotqa", Path("/repo"))

    assert "--production" in cmd
    assert "--few-shot" in cmd
    assert "--include-ppg" in cmd
    assert "--diagnostic-report" in cmd
    assert cmd[cmd.index("--ppg-path-candidates") + 1] == "33"
    assert cmd[cmd.index("--ppg-ensemble-paths") + 1] == "3"
    assert cmd[cmd.index("--ppg-calibration-patience") + 1] == "0"
    assert cmd[cmd.index("--ppg-calibration-execution") + 1] == "deployment"


def _load_run_suite_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("run_suite_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
