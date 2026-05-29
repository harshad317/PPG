"""Validation-gated deployment selection utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ppg.eval.harness import BaselineMetrics


@dataclass(frozen=True)
class DeploymentCandidate:
    """One deployable method scored on a validation split."""

    name: str
    val_score: float
    mean_tokens: float
    lm_calls: int = 0
    priority: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DeploymentSelection:
    """Selected deployment plus the validation evidence behind the decision."""

    selected: DeploymentCandidate
    candidates: tuple[DeploymentCandidate, ...]
    runner_up: DeploymentCandidate | None
    margin_to_runner_up: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "selected": self.selected.name,
            "runner_up": self.runner_up.name if self.runner_up else None,
            "margin_to_runner_up": round(self.margin_to_runner_up, 6),
            "reason": self.reason,
            "candidates": [
                {
                    "name": c.name,
                    "val_score": round(c.val_score, 6),
                    "mean_tokens": round(c.mean_tokens, 3),
                    "lm_calls": c.lm_calls,
                    "priority": c.priority,
                    "metadata": dict(c.metadata),
                }
                for c in self.candidates
            ],
        }


def candidate_from_metrics(
    metrics: BaselineMetrics,
    *,
    name: str | None = None,
    priority: int = 0,
    metadata: Mapping[str, object] | None = None,
) -> DeploymentCandidate:
    """Build a deployment candidate from aggregate validation metrics."""
    return DeploymentCandidate(
        name=name or metrics.name,
        val_score=metrics.task_accuracy,
        mean_tokens=metrics.mean_tokens,
        lm_calls=metrics.lm_calls,
        priority=priority,
        metadata=metadata or {},
    )


def select_deployment_by_validation(
    candidates: Iterable[DeploymentCandidate],
    *,
    incumbent_name: str = "ppg",
    min_margin: float = 0.0,
) -> DeploymentSelection:
    """
    Pick the deployable method with the strongest validation evidence.

    Selection uses only validation-visible metrics:
      1. higher validation score,
      2. lower token count on ties,
      3. fewer LM calls on ties,
      4. explicit priority,
      5. deterministic name tie-break.

    ``min_margin`` is a stability guard: when the incumbent is within that
    score margin of the best candidate, keep the incumbent instead of switching.
    This prevents noisy validation slices from changing deployment for tiny,
    statistically weak gains.
    """
    ranked = _rank_candidates(tuple(candidates))
    if not ranked:
        raise ValueError("candidates must be non-empty")
    _validate_unique_names(ranked)
    if min_margin < 0:
        raise ValueError("min_margin must be >= 0")

    best = ranked[0]
    selected = best
    reason = "best_validation_score"

    incumbent = next((c for c in ranked if c.name == incumbent_name), None)
    if incumbent is not None and best.name != incumbent_name:
        margin_over_incumbent = best.val_score - incumbent.val_score
        if margin_over_incumbent < min_margin:
            selected = incumbent
            reason = "incumbent_within_margin"

    ordered = (selected,) + tuple(c for c in ranked if c.name != selected.name)
    runner_up = ordered[1] if len(ordered) > 1 else None
    margin = selected.val_score - runner_up.val_score if runner_up else 0.0
    return DeploymentSelection(
        selected=selected,
        candidates=ordered,
        runner_up=runner_up,
        margin_to_runner_up=margin,
        reason=reason,
    )


def _rank_candidates(candidates: tuple[DeploymentCandidate, ...]) -> tuple[DeploymentCandidate, ...]:
    return tuple(sorted(candidates, key=_candidate_key))


def _candidate_key(c: DeploymentCandidate) -> tuple[float, float, int, int, str]:
    return (
        -c.val_score,
        c.mean_tokens,
        c.lm_calls,
        -c.priority,
        c.name,
    )


def _validate_unique_names(candidates: tuple[DeploymentCandidate, ...]) -> None:
    names = [c.name for c in candidates]
    dupes = sorted({name for name in names if names.count(name) > 1})
    if dupes:
        raise ValueError(f"candidate names must be unique: {dupes}")
