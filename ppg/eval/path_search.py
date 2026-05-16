"""Validation-based path selection for deployable PPG prompts."""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from ppg.core.executor import LMClient, PromptAssembler
from ppg.core.graph import PPGraph
from ppg.core.tokenizer import count_tokens
from ppg.eval.harness import EvalExample
from ppg.training.reward import ConstraintChecker, TaskMetric


@dataclass
class PathSearchResult:
    """Best validation path plus lightweight diagnostics."""

    path: list[str]
    val_score: float
    mean_tokens: float
    utility: float
    n_examples: int
    n_paths_scored: int
    total_paths: int


def path_utility(graph: PPGraph, path: list[str]) -> float:
    """Sum learned fragment utilities for deterministic pre-ranking."""
    return sum(graph.nodes[nid].utility for nid in path)


def ranked_paths(graph: PPGraph, max_candidates: Optional[int] = None) -> list[list[str]]:
    """
    Enumerate complete graph paths, ranked by learned utility.

    ``max_candidates=None`` keeps all paths.  When capped, the search spends
    validation calls on the routes most supported by LOO utility estimates.
    """
    paths = [list(p) for p in graph.all_paths()]
    paths.sort(
        key=lambda p: (
            -path_utility(graph, p),
            len(p),
            " ".join(graph.nodes[nid].type.value for nid in p),
            " ".join(graph.nodes[nid].template for nid in p),
        )
    )
    if max_candidates is not None and max_candidates > 0:
        return paths[:max_candidates]
    return paths


def select_path_by_validation(
    graph: PPGraph,
    examples: list[EvalExample],
    lm: LMClient,
    metric: TaskMetric,
    constraint_checker: Optional[ConstraintChecker] = None,
    *,
    max_candidates: Optional[int] = None,
    n_workers: int = 1,
    show_progress: bool = False,
) -> PathSearchResult:
    """
    Select the complete graph path with highest validation score.

    Ties prefer higher learned utility, then fewer tokens.  Test examples are
    never used; this is an optimizer/calibration step over the validation split.
    """
    if not examples:
        raise ValueError("examples must be non-empty")

    total_paths = graph.path_count()
    paths = ranked_paths(graph, max_candidates=max_candidates)
    if not paths:
        raise ValueError("graph has no complete paths")

    iterator = paths
    bar = None
    if show_progress:
        try:
            from tqdm import tqdm
            bar = tqdm(paths, desc="calibrate ppg", total=len(paths),
                       unit="path", ncols=100, leave=True)
            iterator = bar
        except ImportError:
            bar = None

    best: Optional[PathSearchResult] = None
    for path in iterator:
        score, mean_tokens = score_path(
            graph=graph,
            path=path,
            examples=examples,
            lm=lm,
            metric=metric,
            constraint_checker=constraint_checker,
            n_workers=n_workers,
        )
        candidate = PathSearchResult(
            path=path,
            val_score=score,
            mean_tokens=mean_tokens,
            utility=path_utility(graph, path),
            n_examples=len(examples),
            n_paths_scored=len(paths),
            total_paths=total_paths,
        )
        if best is None or _is_better(candidate, best):
            best = candidate
        if bar is not None:
            bar.set_postfix(best=f"{best.val_score:.3f}")

    if bar is not None:
        bar.close()

    assert best is not None
    return best


def score_path(
    graph: PPGraph,
    path: list[str],
    examples: list[EvalExample],
    lm: LMClient,
    metric: TaskMetric,
    constraint_checker: Optional[ConstraintChecker] = None,
    *,
    n_workers: int = 1,
) -> tuple[float, float]:
    """Return ``(mean_score, mean_prompt_tokens)`` for a fixed path."""
    asm = PromptAssembler(graph)

    def _run(ex: EvalExample) -> tuple[float, int]:
        prompt = asm.assemble(path, {"input": ex.x})
        response = lm.complete(prompt)
        return _score_response(response, ex, metric, constraint_checker), count_tokens(prompt)

    if n_workers <= 1:
        out = [_run(ex) for ex in examples]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            out = list(pool.map(_run, examples))

    scores = [s for s, _ in out]
    tokens = [t for _, t in out]
    return statistics.mean(scores), statistics.mean(tokens)


def _score_response(
    response: str,
    example: EvalExample,
    metric: TaskMetric,
    constraint_checker: Optional[ConstraintChecker],
) -> float:
    test_list = example.metadata.get("test_list") if example.metadata else None
    if test_list and hasattr(metric, "score_with_tests"):
        return metric.score_with_tests(response, test_list)
    if example.constraints and constraint_checker is not None:
        return constraint_checker.check(response, example.constraints, example.metadata or {})
    return metric.score(response, example.y_star)


def _is_better(candidate: PathSearchResult, incumbent: PathSearchResult) -> bool:
    return (
        candidate.val_score,
        candidate.utility,
        -candidate.mean_tokens,
        -len(candidate.path),
    ) > (
        incumbent.val_score,
        incumbent.utility,
        -incumbent.mean_tokens,
        -len(incumbent.path),
    )
