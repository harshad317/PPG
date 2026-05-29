"""Validation-based path selection for deployable PPG prompts."""

from __future__ import annotations

import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from ppg.core.executor import LMClient, PromptAssembler
from ppg.core.features import AnswerNormalizer, default_normalizer
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
    ensemble_val_score: Optional[float] = None


@dataclass
class _PathEvaluation:
    candidate: PathCandidate
    responses: list[str]


PathRunner = Callable[[list[str], EvalExample], tuple[str, int]]


def effective_majority_ensemble_size(top_k: int) -> int:
    """
    Return a usable majority-vote ensemble size.

    Even-size majority ensembles tie whenever the voters disagree; this makes
    two-path deployments collapse to the first path in common cases. Keep
    singleton deployments unchanged, and round larger even budgets up to the
    next odd number so validation can test a real majority.
    """
    top_k = max(1, top_k)
    if top_k > 1 and top_k % 2 == 0:
        return top_k + 1
    return top_k


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
    path_runner: Optional[PathRunner] = None,
    normalizer: AnswerNormalizer = default_normalizer,
    racing_subset: int = 0,
    racing_survivors: int = 0,
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

    # Successive-halving (racing): cheaply pre-filter candidates on a small
    # validation subset, then only full-score the survivors. The final winner is
    # still validated on the full set below, so quality is preserved; only
    # clearly-dominated paths are dropped early. Saves calls when there are many
    # candidate paths and the full val split is large.
    if (
        racing_subset > 0
        and racing_survivors > 0
        and len(paths) > racing_survivors
        and racing_subset < len(examples)
    ):
        paths = _race_paths(
            graph=graph,
            paths=paths,
            subset=examples[:racing_subset],
            lm=lm,
            metric=metric,
            constraint_checker=constraint_checker,
            n_workers=n_workers,
            path_runner=path_runner,
            survivors=racing_survivors,
        )

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
    evaluations: list[_PathEvaluation] = []
    no_improve_count = 0
    n_scored = 0
    for path in iterator:
        score, mean_tokens, responses = _score_path_detailed(
            graph=graph,
            path=path,
            examples=examples,
            lm=lm,
            metric=metric,
            constraint_checker=constraint_checker,
            n_workers=n_workers,
            path_runner=path_runner,
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
        evaluations.append(_PathEvaluation(
            candidate=candidates[-1],
            responses=responses,
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
    best.candidates, best.ensemble_val_score = _select_ensemble_candidates(
        evaluations=evaluations,
        examples=examples,
        metric=metric,
        constraint_checker=constraint_checker,
        top_k=effective_majority_ensemble_size(return_top_k),
        normalizer=normalizer,
    )
    return best


def _race_paths(
    graph: PPGraph,
    paths: list[list[str]],
    subset: list[EvalExample],
    lm: LMClient,
    metric: TaskMetric,
    constraint_checker: Optional[ConstraintChecker],
    n_workers: int,
    path_runner: Optional[PathRunner],
    survivors: int,
) -> list[list[str]]:
    """
    Score every path on a small subset, return the top ``survivors`` paths.

    Ranking mirrors the main loop's preferences: higher subset score first,
    then higher learned utility, then fewer nodes (cheaper). Original candidate
    order is preserved among the survivors so downstream early-stopping behaves
    the same as without racing.
    """
    scored: list[tuple[float, float, int, int, list[str]]] = []
    for order, path in enumerate(paths):
        score, _mean_tokens = score_path(
            graph, path, subset, lm, metric, constraint_checker,
            n_workers=n_workers, path_runner=path_runner,
        )
        scored.append((score, path_utility(graph, path), -len(path), order, path))

    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    survivor_set = {id(t[4]) for t in scored[:survivors]}
    return [p for p in paths if id(p) in survivor_set]


def score_path(
    graph: PPGraph,
    path: list[str],
    examples: list[EvalExample],
    lm: LMClient,
    metric: TaskMetric,
    constraint_checker: Optional[ConstraintChecker] = None,
    *,
    n_workers: int = 1,
    path_runner: Optional[PathRunner] = None,
) -> tuple[float, float]:
    """Return ``(mean_score, mean_prompt_tokens)`` for a fixed path."""
    score, mean_tokens, _responses = _score_path_detailed(
        graph=graph,
        path=path,
        examples=examples,
        lm=lm,
        metric=metric,
        constraint_checker=constraint_checker,
        n_workers=n_workers,
        path_runner=path_runner,
    )
    return score, mean_tokens


def _score_path_detailed(
    graph: PPGraph,
    path: list[str],
    examples: list[EvalExample],
    lm: LMClient,
    metric: TaskMetric,
    constraint_checker: Optional[ConstraintChecker] = None,
    *,
    n_workers: int = 1,
    path_runner: Optional[PathRunner] = None,
) -> tuple[float, float, list[str]]:
    """Return score, mean prompt tokens, and per-example responses."""
    asm = PromptAssembler(graph)

    def _run(ex: EvalExample) -> tuple[float, int, str]:
        if path_runner is None:
            prompt = asm.assemble(path, {"input": ex.x})
            response = lm.complete(prompt)
            tokens = count_tokens(prompt)
        else:
            response, tokens = path_runner(path, ex)
        score = _score_response(response, ex, metric, constraint_checker)
        return score, tokens, response

    if n_workers <= 1:
        out = [_run(ex) for ex in examples]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            out = list(pool.map(_run, examples))

    scores = [s for s, _, _ in out]
    tokens = [t for _, t, _ in out]
    responses = [r for _, _, r in out]
    return statistics.mean(scores), statistics.mean(tokens), responses


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


def _candidate_sort_key(c: PathCandidate) -> tuple[float, float, float, float, int]:
    return (
        c.adjusted_score,
        c.val_score,
        c.utility,
        -c.mean_tokens,
        -len(c.path),
    )


def _select_ensemble_candidates(
    evaluations: list[_PathEvaluation],
    examples: list[EvalExample],
    metric: TaskMetric,
    constraint_checker: Optional[ConstraintChecker],
    top_k: int,
    normalizer: AnswerNormalizer = default_normalizer,
) -> tuple[list[PathCandidate], float]:
    if not evaluations:
        return [], 0.0

    ranked = sorted(
        evaluations,
        key=lambda ev: _candidate_sort_key(ev.candidate),
        reverse=True,
    )
    if top_k <= 1:
        return [ranked[0].candidate], ranked[0].candidate.val_score

    best_selected: list[_PathEvaluation] = []
    best_key: tuple[float, float, float, float, float] | None = None

    # Try several high-quality seeds. The best individual path is not always
    # the best first voter because tie-breaking returns the first response.
    # Track every prefix size and let validation choose the smallest ensemble
    # that maximizes score. This avoids paying for extra paths when they only
    # tie, or worse, dilute the best route.
    for seed in ranked[:min(10, len(ranked))]:
        selected = [seed]
        remaining = [ev for ev in ranked if ev is not seed]
        best_selected, best_key = _maybe_update_best_ensemble(
            selected,
            examples,
            metric,
            constraint_checker,
            normalizer,
            best_selected,
            best_key,
        )

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_step_key: tuple[float, float, float, float, float, int] | None = None
            for i, ev in enumerate(remaining):
                score = _ensemble_score(
                    selected + [ev],
                    examples,
                    metric,
                    constraint_checker,
                    normalizer,
                )
                c = ev.candidate
                key = (
                    score,
                    c.adjusted_score,
                    c.val_score,
                    c.utility,
                    -c.mean_tokens,
                    -len(c.path),
                )
                if best_step_key is None or key > best_step_key:
                    best_step_key = key
                    best_idx = i
            selected.append(remaining.pop(best_idx))
            best_selected, best_key = _maybe_update_best_ensemble(
                selected,
                examples,
                metric,
                constraint_checker,
                normalizer,
                best_selected,
                best_key,
            )

    return [ev.candidate for ev in best_selected], best_key[0] if best_key else 0.0


def _maybe_update_best_ensemble(
    selected: list[_PathEvaluation],
    examples: list[EvalExample],
    metric: TaskMetric,
    constraint_checker: Optional[ConstraintChecker],
    normalizer: AnswerNormalizer,
    best_selected: list[_PathEvaluation],
    best_key: Optional[tuple[float, float, float, float, float]],
) -> tuple[list[_PathEvaluation], tuple[float, float, float, float, float]]:
    score = _ensemble_score(selected, examples, metric, constraint_checker, normalizer)
    total_tokens = sum(ev.candidate.mean_tokens for ev in selected)
    mean_adjusted = statistics.mean(ev.candidate.adjusted_score for ev in selected)
    mean_val = statistics.mean(ev.candidate.val_score for ev in selected)
    key = (
        score,
        -total_tokens,
        mean_adjusted,
        mean_val,
        selected[0].candidate.adjusted_score,
    )
    if best_key is None or key > best_key:
        return list(selected), key
    return best_selected, best_key


def _ensemble_score(
    evaluations: list[_PathEvaluation],
    examples: list[EvalExample],
    metric: TaskMetric,
    constraint_checker: Optional[ConstraintChecker],
    normalizer: AnswerNormalizer = default_normalizer,
) -> float:
    if not evaluations:
        return 0.0
    scores = []
    for i, ex in enumerate(examples):
        responses = [ev.responses[i] for ev in evaluations]
        response = _select_majority_response(responses, normalizer)
        scores.append(_score_response(response, ex, metric, constraint_checker))
    return statistics.mean(scores) if scores else 0.0


def _select_majority_response(
    responses: list[str],
    normalizer: AnswerNormalizer = default_normalizer,
) -> str:
    if not responses:
        return ""
    normalized = [normalizer(response) for response in responses]
    counts = Counter(normalized)
    best_answer, _count = counts.most_common(1)[0]
    for answer, response in zip(normalized, responses):
        if answer == best_answer:
            return response
    return responses[0]
