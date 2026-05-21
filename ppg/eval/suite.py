"""Cross-benchmark suite summarization utilities."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class SuiteRecord:
    """One benchmark result file loaded from ``run_benchmark.py`` output."""

    path: Path
    benchmark: str
    model: str
    seed: int
    methods: dict[str, dict]
    winner: str
    modified_time: float


def load_suite_record(path: str | Path) -> SuiteRecord:
    """Load one JSON result produced by ``scripts/run_benchmark.py``."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    methods = {
        row["name"]: row
        for row in payload.get("results", [])
        if isinstance(row, dict) and "name" in row
    }
    return SuiteRecord(
        path=p,
        benchmark=str(payload.get("benchmark", "")),
        model=str(payload.get("model", "")),
        seed=int(payload.get("seed", 0)),
        methods=methods,
        winner=str(payload.get("winner", "")),
        modified_time=p.stat().st_mtime,
    )


def discover_result_files(
    output_dir: str | Path,
    *,
    benchmarks: Optional[Iterable[str]] = None,
    model: Optional[str] = None,
) -> list[Path]:
    """Find result JSON files matching optional benchmark/model filters."""
    wanted_benchmarks = set(benchmarks or [])
    result_files: list[Path] = []
    for path in Path(output_dir).glob("*.json"):
        try:
            record = load_suite_record(path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if wanted_benchmarks and record.benchmark not in wanted_benchmarks:
            continue
        if model is not None and record.model != model:
            continue
        result_files.append(path)
    return sorted(result_files)


def latest_records_by_benchmark(files: Iterable[str | Path]) -> list[SuiteRecord]:
    """Keep the newest result file for each benchmark."""
    latest: dict[str, SuiteRecord] = {}
    for path in files:
        record = load_suite_record(path)
        current = latest.get(record.benchmark)
        if current is None or record.modified_time > current.modified_time:
            latest[record.benchmark] = record
    return sorted(latest.values(), key=lambda r: r.benchmark)


def summarize_records(records: Iterable[SuiteRecord]) -> dict:
    """Compute suite-level method leaderboard and per-benchmark deltas."""
    rows = list(records)
    method_names = sorted({name for record in rows for name in record.methods})
    per_benchmark = [_summarize_benchmark(record) for record in rows]

    methods = []
    for method in method_names:
        method_rows = [
            bench
            for bench in per_benchmark
            if method in bench["scores"]
        ]
        scores = [bench["scores"][method] for bench in method_rows]
        ranks = [bench["ranks"][method] for bench in method_rows]
        tokens = [
            bench["mean_tokens"][method]
            for bench in method_rows
            if bench["mean_tokens"][method] is not None
        ]
        lm_calls = [
            bench["lm_calls"][method]
            for bench in method_rows
            if bench["lm_calls"][method] is not None
        ]
        deltas_vs_ppg = [
            bench["scores"][method] - bench["scores"]["ppg"]
            for bench in method_rows
            if method != "ppg" and "ppg" in bench["scores"]
        ]

        entry = {
            "name": method,
            "coverage": len(method_rows),
            "mean_score": _mean(scores),
            "average_rank": _mean(ranks),
            "wins": sum(1 for rank in ranks if rank == 1),
            "mean_tokens": _mean(tokens),
            "mean_lm_calls": _mean(lm_calls),
        }
        if method == "ppg":
            entry["mean_delta_vs_best_non_ppg"] = _mean([
                bench["ppg_delta_vs_best_non_ppg"]
                for bench in per_benchmark
                if bench["ppg_delta_vs_best_non_ppg"] is not None
            ])
        else:
            entry["mean_delta_vs_ppg"] = _mean(deltas_vs_ppg)
        methods.append(entry)

    methods.sort(key=lambda m: (
        -m["coverage"],
        m["average_rank"] if m["average_rank"] is not None else 1e9,
        -(m["mean_score"] or 0.0),
        m["name"],
    ))

    return {
        "n_benchmarks": len(rows),
        "benchmarks": [record.benchmark for record in rows],
        "methods": methods,
        "per_benchmark": per_benchmark,
    }


def render_markdown_summary(summary: dict) -> str:
    """Render a compact markdown leaderboard."""
    lines = [
        "# PPG Suite Summary",
        "",
        f"Benchmarks: {', '.join(summary.get('benchmarks', [])) or 'none'}",
        "",
        "| Method | Coverage | Mean Score | Avg Rank | Wins | PPG Delta | Mean Tokens | Mean LM Calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary.get("methods", []):
        delta = row.get("mean_delta_vs_best_non_ppg")
        if row["name"] != "ppg":
            delta = row.get("mean_delta_vs_ppg")
        lines.append(
            "| {name} | {coverage} | {mean_score} | {average_rank} | {wins} | "
            "{delta} | {mean_tokens} | {mean_lm_calls} |".format(
                name=row["name"],
                coverage=row["coverage"],
                mean_score=_fmt(row.get("mean_score")),
                average_rank=_fmt(row.get("average_rank")),
                wins=row["wins"],
                delta=_fmt(delta),
                mean_tokens=_fmt(row.get("mean_tokens"), digits=1),
                mean_lm_calls=_fmt(row.get("mean_lm_calls"), digits=1),
            )
        )

    lines.extend([
        "",
        "## Per Benchmark",
        "",
        "| Benchmark | Winner | PPG | Best Non-PPG | PPG Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in summary.get("per_benchmark", []):
        lines.append(
            "| {benchmark} | {winner} | {ppg} | {best_non_ppg} | {delta} |".format(
                benchmark=row["benchmark"],
                winner=row["winner"],
                ppg=_fmt(row.get("ppg_score")),
                best_non_ppg=_fmt(row.get("best_non_ppg_score")),
                delta=_fmt(row.get("ppg_delta_vs_best_non_ppg")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_summary_files(summary: dict, output_dir: str | Path, *, prefix: str = "suite") -> dict[str, Path]:
    """Write JSON and markdown summaries."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{prefix}_summary.json"
    md_path = out / f"{prefix}_summary.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    md_path.write_text(render_markdown_summary(summary), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _summarize_benchmark(record: SuiteRecord) -> dict:
    method_scores = {
        name: float(row.get("task_accuracy", 0.0))
        for name, row in record.methods.items()
    }
    sorted_methods = sorted(method_scores, key=lambda name: (-method_scores[name], name))
    ranks = {name: i + 1 for i, name in enumerate(sorted_methods)}
    best_non_ppg_name = next((name for name in sorted_methods if name != "ppg"), None)
    ppg_score = method_scores.get("ppg")
    best_non_ppg_score = (
        method_scores[best_non_ppg_name]
        if best_non_ppg_name is not None else None
    )
    delta = (
        ppg_score - best_non_ppg_score
        if ppg_score is not None and best_non_ppg_score is not None else None
    )

    return {
        "benchmark": record.benchmark,
        "path": str(record.path),
        "winner": sorted_methods[0] if sorted_methods else record.winner,
        "scores": method_scores,
        "ranks": ranks,
        "mean_tokens": {
            name: _maybe_float(row.get("mean_tokens"))
            for name, row in record.methods.items()
        },
        "lm_calls": {
            name: _maybe_float(row.get("lm_calls"))
            for name, row in record.methods.items()
        },
        "ppg_score": ppg_score,
        "best_non_ppg": best_non_ppg_name,
        "best_non_ppg_score": best_non_ppg_score,
        "ppg_delta_vs_best_non_ppg": delta,
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | int | None, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"
