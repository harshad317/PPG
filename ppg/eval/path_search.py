"""Validation-based path selection for deployable PPG prompts."""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from ppg.core.executor import LMClient, PromptAssembler
from ppg.core.graph import PPGraph
from ppg.core.tokenizer import count_tokens
from ppg.eval.harness import EvalExample
from ppg.training.reward import ConstraintChecker, TaskMetric


@dataclass
class PathCandidate:
    """One validation-scored path candidate."""

    path: list[str]
    val_score: float
    mean_tokens: float
    utility: float
    adjusted_score: float


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
    candidates: list[PathCandidate] = field(default_factory=list)


def path_utility(graph: PPGraph, path: list[str]) -> float:
    """Sum learned fragment utilities for deterministic pre-ranking."""
    return sum(graph.nodes[nid].utility for nid in path)


def ranked_paths(graph: PPGraph, max_candidates: Optional[int] = None) -> list[list[str]]:
    """
    Enumerate complete graph paths with a diversified candidate set.

    When ``max_candidates`` is set, the candidate pool is split:
      - First half: top paths by learned utility (explore best-known routes)
      - Second half: shortest paths by node count (explore concise routes)

    This prevents F1-inflated utilities from filling the entire candidate set
    with verbose paths, ensuring calibration can compare both strategies.

    ``max_candidates=None`` keeps all paths sorted by utility.
    """
    paths = [list(p) for p in graph.all_paths()]

    def _sort_key(p):
        return (
            -path_utility(graph, p),
            len(p),
            " ".join(graph.nodes[nid].type.value for nid in p),
            " ".join(graph.nodes[nid].template for nid in p),
        )

    if max_candidates is None or max_candidates <= 0:
        paths.sort(key=_sort_key)
        return paths

    by_utility = sorted(paths, key=_sort_key)
    by_length = sorted(paths, key=lambda p: (
        len(p),
        -path_utility(graph, p),
    ))

    half = max_candidates // 2
    seen: set[tuple[str, ...]] = set()
    result: list[list[str]] = []

    for source in (by_utility, by_length):
        budget = half if source is by_utility else max_candidates
        for p in source:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                result.append(p)
            if len(result) >= budget:
                break

    return result[:max_candidates]


def _estimate_input_tokens(examples: list[EvalExample]) -> float:
    """Mean token count of raw inputs (no prompt fragments)."""
    if not examples:
        return 0.0
    return statistics.mean(count_tokens(ex.x) for ex in examples)


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
    early_stop_patience: int = 10,
    token_efficiency_weight: float = 0.02,
    max_tokens_ref: int = 2048,
    return_top_k: int = 1,
) -> PathSearchResult:
    """
    Select the complete graph path with highest overhead-adjusted validation score.

    Token penalty is computed on FRAGMENT OVERHEAD only (total tokens minus
    average input tokens), not total prompt length. This prevents penalizing
    inherently long inputs (e.g. HotpotQA's 10-passage context).

    Adjusted score = val_score - weight * max(0, mean_tokens - input_tokens) / ref

    When ``max_candidates`` exceeds total paths, evaluates all paths.
    Early stopping fires when the best score hasn't improved for
    ``early_stop_patience`` consecutive paths.

    Test examples are never used; this is a calibration step over the val split.
    """
    if not examples:
        raise ValueError("examples must be non-empty")

    total_paths = graph.path_count()

    # Evaluate all paths when graph is small enough
    effective_max = max_candidates
    if effective_max is not None and total_paths <= effective_max * 2:
        effective_max = None

    paths = ranked_paths(graph, max_candidates=effective_max)
    if not paths:
        raise ValueError("graph has no complete paths")

    input_tokens = _estimate_input_tokens(examples)

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
    candidates: list[PathCandidate] = []
    no_improve_count = 0
    n_scored = 0
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
        n_scored += 1
        candidate = PathSearchResult(
            path=path,
            val_score=score,
            mean_tokens=mean_tokens,
            utility=path_utility(graph, path),
            n_examples=len(examples),
            n_paths_scored=n_scored,
            total_paths=total_paths,
        )
        adjusted = _adjusted_score(
            candidate,
            token_efficiency_weight,
            max_tokens_ref,
            input_tokens,
        )
        candidates.append(PathCandidate(
            path=path,
            val_score=score,
            mean_tokens=mean_tokens,
            utility=candidate.utility,
            adjusted_score=adjusted,
        ))
        if best is None or _is_better(candidate, best,
                                       token_efficiency_weight, max_tokens_ref,
                                       input_tokens):
            best = candidate
            no_improve_count = 0
        else:
            no_improve_count += 1
        if bar is not None:
            bar.set_postfix(best=f"{best.val_score:.3f}")
        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            break

    if bar is not None:
        bar.close()

    assert best is not None
    best.n_paths_scored = n_scored
    top_k = max(1, return_top_k)
    best.candidates = sorted(
        candidates,
        key=lambda c: (
            c.adjusted_score,
            c.val_score,
            c.utility,
            -c.mean_tokens,
            -len(c.path),
        ),
        reverse=True,
    )[:top_k]
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


def _is_better(
    candidate: PathSearchResult,
    incumbent: PathSearchResult,
    token_weight: float = 0.02,
    max_tokens_ref: int = 2048,
    input_tokens: float = 0.0,
) -> bool:
    c_adj = _adjusted_score(candidate, token_weight, max_tokens_ref, input_tokens)
    i_adj = _adjusted_score(incumbent, token_weight, max_tokens_ref, input_tokens)
    return (
        c_adj,
        candidate.utility,
        -candidate.mean_tokens,
        -len(candidate.path),
    ) > (
        i_adj,
        incumbent.utility,
        -incumbent.mean_tokens,
        -len(incumbent.path),
    )


def _adjusted_score(
    result: PathSearchResult,
    token_weight: float = 0.02,
    max_tokens_ref: int = 2048,
    input_tokens: float = 0.0,
) -> float:
    overhead = max(0.0, result.mean_tokens - input_tokens)
    return result.val_score - token_weight * (overhead / max_tokens_ref)
