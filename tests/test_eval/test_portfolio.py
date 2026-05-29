"""Tests for validation-gated deployment selection."""

from __future__ import annotations

import pytest

from ppg.eval.harness import BaselineMetrics
from ppg.eval.portfolio import (
    DeploymentCandidate,
    candidate_from_metrics,
    select_deployment_by_validation,
)


def test_selects_highest_validation_score():
    selection = select_deployment_by_validation([
        DeploymentCandidate("ppg", 0.70, 120),
        DeploymentCandidate("static_best", 0.75, 100),
    ])

    assert selection.selected.name == "static_best"
    assert selection.reason == "best_validation_score"


def test_tie_breaks_to_lower_token_count():
    selection = select_deployment_by_validation([
        DeploymentCandidate("ppg", 0.80, 150),
        DeploymentCandidate("flat_all", 0.80, 300),
        DeploymentCandidate("static_best", 0.80, 90),
    ])

    assert selection.selected.name == "static_best"


def test_keeps_incumbent_when_best_is_inside_margin():
    selection = select_deployment_by_validation(
        [
            DeploymentCandidate("ppg", 0.80, 150),
            DeploymentCandidate("static_best", 0.805, 90),
        ],
        min_margin=0.01,
    )

    assert selection.selected.name == "ppg"
    assert selection.reason == "incumbent_within_margin"


def test_switches_when_challenger_beats_margin():
    selection = select_deployment_by_validation(
        [
            DeploymentCandidate("ppg", 0.80, 150),
            DeploymentCandidate("static_best", 0.82, 90),
        ],
        min_margin=0.01,
    )

    assert selection.selected.name == "static_best"


def test_candidate_from_metrics_uses_aggregate_fields():
    metrics = BaselineMetrics(
        name="ppg",
        task_scores=[1.0, 0.0, 1.0],
        token_counts=[100, 200, 300],
        constraint_scores=[],
        lm_calls=3,
    )

    candidate = candidate_from_metrics(metrics, metadata={"split": "val"})

    assert candidate.name == "ppg"
    assert candidate.val_score == pytest.approx(2 / 3)
    assert candidate.mean_tokens == pytest.approx(200)
    assert candidate.lm_calls == 3
    assert candidate.metadata["split"] == "val"


def test_empty_candidates_raise():
    with pytest.raises(ValueError, match="non-empty"):
        select_deployment_by_validation([])


def test_duplicate_candidate_names_raise():
    with pytest.raises(ValueError, match="unique"):
        select_deployment_by_validation([
            DeploymentCandidate("ppg", 0.7, 100),
            DeploymentCandidate("ppg", 0.8, 90),
        ])
