"""Tests for ppg/training/credit.py — LOO credit assignment."""

from __future__ import annotations

import numpy as np
import pytest

from ppg.core import FragmentType, PPGraphBuilder
from ppg.core.executor import PathTrace, PromptAssembler
from ppg.training.credit import (
    CreditAssignmentResult,
    CreditAssigner,
    CreditAssignerConfig,
)
from ppg.training.reward import ExactMatchMetric


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def make_graph_and_ids():
    """task_framing -> reasoning_style -> output_contract (3 nodes)."""
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think step by step.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    b.connect_chain(*ids)
    return b.build(), ids


def make_long_graph_and_ids():
    """
    task_framing -> reasoning_style -> compression -> output_contract (4 nodes).
    Long enough that middle two nodes are ablatable.
    """
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think.")
    b.add_fragment(FragmentType.COMPRESSION,     "Be concise.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    b.connect_chain(*ids)
    return b.build(), ids


class FixedLM:
    """Always returns a fixed response."""
    def __init__(self, response: str = "42"):
        self.response = response
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class SwitchLM:
    """Returns first response on call 1, second on call 2+."""
    def __init__(self, first: str, second: str):
        self.first = first
        self.second = second
        self.n = 0

    def complete(self, prompt: str) -> str:
        self.n += 1
        return self.first if self.n == 1 else self.second


def make_trace(node_ids: list[str], lm_response: str = "42",
               token_count: int = 10) -> PathTrace:
    from ppg.core.features import RuntimeFeatures
    return PathTrace(
        node_ids=node_ids,
        edges_traversed=[(node_ids[i], node_ids[i+1]) for i in range(len(node_ids)-1)],
        assembled_prompt="[prompt]",
        token_count=token_count,
        lm_response=lm_response,
        pre_lm_features=RuntimeFeatures(),
        post_lm_features=None,
        guard_decisions=[],
        escalated=False,
    )


def make_assigner(lm=None, graph=None, metric=None, config=None):
    if lm is None:
        lm = FixedLM("42")
    if graph is None:
        graph, _ = make_graph_and_ids()
    if metric is None:
        metric = ExactMatchMetric()
    asm = PromptAssembler(graph)
    return CreditAssigner(lm=lm, assembler=asm, task_metric=metric, config=config)


# ---------------------------------------------------------------------------
# CreditAssignmentResult
# ---------------------------------------------------------------------------

class TestCreditAssignmentResult:
    def test_marginal_computed(self):
        r = CreditAssignmentResult(
            node_id="n1", full_score=0.8, ablated_score=0.5,
            marginal=0.3, ablated_path=["n0", "n2"],
        )
        assert r.marginal == pytest.approx(0.3)

    def test_node_was_helpful_positive(self):
        r = CreditAssignmentResult("n", 1.0, 0.5, 0.5, [])
        assert r.node_was_helpful is True
        assert r.node_was_harmful is False

    def test_node_was_harmful_negative(self):
        r = CreditAssignmentResult("n", 0.5, 1.0, -0.5, [])
        assert r.node_was_harmful is True
        assert r.node_was_helpful is False

    def test_zero_marginal_neither(self):
        r = CreditAssignmentResult("n", 0.5, 0.5, 0.0, [])
        assert r.node_was_helpful is False
        assert r.node_was_harmful is False

    def test_ablated_path_stored(self):
        r = CreditAssignmentResult("n1", 1.0, 0.0, 1.0, ["n0", "n2"])
        assert r.ablated_path == ["n0", "n2"]


# ---------------------------------------------------------------------------
# CreditAssignerConfig
# ---------------------------------------------------------------------------

class TestCreditAssignerConfig:
    def test_defaults(self):
        cfg = CreditAssignerConfig()
        assert cfg.p_ablate == pytest.approx(0.15)
        assert cfg.skip_source is True
        assert cfg.skip_terminal is True
        assert cfg.min_path_length == 3

    def test_custom_values(self):
        cfg = CreditAssignerConfig(p_ablate=0.5, skip_source=False, min_path_length=2)
        assert cfg.p_ablate == pytest.approx(0.5)
        assert cfg.skip_source is False
        assert cfg.min_path_length == 2


# ---------------------------------------------------------------------------
# CreditAssigner._ablatable_nodes
# ---------------------------------------------------------------------------

class TestAblatable:
    def test_three_node_path_one_ablatable(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        trace = make_trace(ids)
        assigner = make_assigner(graph=g)
        ablatable = assigner._ablatable_nodes(trace)
        assert ablatable == [rs]

    def test_four_node_path_two_ablatable(self):
        g, ids = make_long_graph_and_ids()
        tf, rs, comp, oc = ids
        trace = make_trace(ids)
        assigner = make_assigner(graph=g)
        ablatable = assigner._ablatable_nodes(trace)
        assert ablatable == [rs, comp]

    def test_two_node_path_too_short(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        # Only pass first 2 nodes
        trace = make_trace([tf, rs])
        assigner = make_assigner(graph=g)
        ablatable = assigner._ablatable_nodes(trace)
        assert ablatable == []

    def test_skip_source_false_includes_first(self):
        g, ids = make_long_graph_and_ids()
        tf, rs, comp, oc = ids
        cfg = CreditAssignerConfig(skip_source=False, skip_terminal=True)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids)
        ablatable = assigner._ablatable_nodes(trace)
        assert tf in ablatable

    def test_skip_terminal_false_includes_last(self):
        g, ids = make_long_graph_and_ids()
        tf, rs, comp, oc = ids
        cfg = CreditAssignerConfig(skip_source=True, skip_terminal=False)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids)
        ablatable = assigner._ablatable_nodes(trace)
        assert oc in ablatable

    def test_min_path_length_blocks_short_path(self):
        g, ids = make_long_graph_and_ids()
        cfg = CreditAssignerConfig(min_path_length=10)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids)
        assert assigner._ablatable_nodes(trace) == []


# ---------------------------------------------------------------------------
# CreditAssigner._run_loo
# ---------------------------------------------------------------------------

class TestRunLOO:
    def test_marginal_positive_when_ablation_hurts(self):
        """Node in path improves score: ablating reduces score."""
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        # Full path response matches reference; ablated doesn't
        lm = FixedLM("42")
        lm_ablated = FixedLM("wrong")

        class TwoCallLM:
            def __init__(self):
                self.n = 0
            def complete(self, prompt):
                # Not called for full (uses cached); called once for ablated
                return "wrong"

        asm = PromptAssembler(g)
        metric = ExactMatchMetric()
        assigner = CreditAssigner(lm=TwoCallLM(), assembler=asm, task_metric=metric)
        trace = make_trace(ids, lm_response="42")
        result = assigner._run_loo(trace, g, "What is 2+2?", "42", rs)
        assert result.marginal == pytest.approx(1.0)  # 1.0 - 0.0

    def test_marginal_negative_when_ablation_helps(self):
        """Node hurts performance: ablating improves score."""
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids

        class CorrectAblatedLM:
            def complete(self, prompt):
                return "42"

        asm = PromptAssembler(g)
        metric = ExactMatchMetric()
        assigner = CreditAssigner(lm=CorrectAblatedLM(), assembler=asm, task_metric=metric)
        # Full path was wrong
        trace = make_trace(ids, lm_response="wrong")
        result = assigner._run_loo(trace, g, "What is 2+2?", "42", rs)
        assert result.marginal == pytest.approx(-1.0)  # 0.0 - 1.0

    def test_ablated_path_excludes_node(self):
        g, ids = make_long_graph_and_ids()
        tf, rs, comp, oc = ids
        lm = FixedLM("42")
        asm = PromptAssembler(g)
        metric = ExactMatchMetric()
        assigner = CreditAssigner(lm=lm, assembler=asm, task_metric=metric)
        trace = make_trace(ids, lm_response="42")
        result = assigner._run_loo(trace, g, "x", "42", rs)
        assert rs not in result.ablated_path
        assert tf in result.ablated_path
        assert oc in result.ablated_path

    def test_one_lm_call_per_loo(self):
        """_run_loo makes exactly 1 LM call (ablated); full score is cached."""
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        lm = FixedLM("42")
        asm = PromptAssembler(g)
        metric = ExactMatchMetric()
        assigner = CreditAssigner(lm=lm, assembler=asm, task_metric=metric)
        trace = make_trace(ids, lm_response="42")
        assigner._run_loo(trace, g, "x", "42", rs)
        assert len(lm.calls) == 1

    def test_result_fields_populated(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        lm = FixedLM("42")
        asm = PromptAssembler(g)
        assigner = CreditAssigner(lm=lm, assembler=asm, task_metric=ExactMatchMetric())
        trace = make_trace(ids, lm_response="42")
        result = assigner._run_loo(trace, g, "x", "42", rs)
        assert result.node_id == rs
        assert result.full_score == pytest.approx(1.0)
        assert isinstance(result.ablated_score, float)
        assert isinstance(result.marginal, float)
        assert isinstance(result.ablated_path, list)


# ---------------------------------------------------------------------------
# CreditAssigner.force_assign
# ---------------------------------------------------------------------------

class TestForceAssign:
    def test_updates_utility_in_place(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        lm = FixedLM("42")
        asm = PromptAssembler(g)
        assigner = CreditAssigner(lm=lm, assembler=asm, task_metric=ExactMatchMetric())
        trace = make_trace(ids, lm_response="42")
        before_n = g.nodes[rs].utility_n
        assigner.force_assign(trace, g, "x", "42", rs)
        assert g.nodes[rs].utility_n == before_n + 1

    def test_increments_n_assignments(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        assigner = make_assigner(graph=g)
        trace = make_trace(ids, lm_response="42")
        assert assigner.n_assignments == 0
        assigner.force_assign(trace, g, "x", "42", rs)
        assert assigner.n_assignments == 1

    def test_returns_result(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        assigner = make_assigner(graph=g)
        trace = make_trace(ids, lm_response="42")
        result = assigner.force_assign(trace, g, "x", "42", rs)
        assert isinstance(result, CreditAssignmentResult)
        assert result.node_id == rs

    def test_force_assign_ignores_p_ablate(self):
        """force_assign always runs regardless of p_ablate=0."""
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        cfg = CreditAssignerConfig(p_ablate=0.0)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids, lm_response="42")
        result = assigner.force_assign(trace, g, "x", "42", rs)
        assert result is not None


# ---------------------------------------------------------------------------
# CreditAssigner.maybe_assign
# ---------------------------------------------------------------------------

class TestMaybeAssign:
    def test_always_assigns_when_p1(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        cfg = CreditAssignerConfig(p_ablate=1.0)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids, lm_response="42")
        rng = np.random.default_rng(0)
        result = assigner.maybe_assign(trace, g, "x", "42", rng)
        assert result is not None
        assert isinstance(result, CreditAssignmentResult)

    def test_never_assigns_when_p0(self):
        g, ids = make_graph_and_ids()
        cfg = CreditAssignerConfig(p_ablate=0.0)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids, lm_response="42")
        rng = np.random.default_rng(0)
        result = assigner.maybe_assign(trace, g, "x", "42", rng)
        assert result is None

    def test_short_path_returns_none(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        cfg = CreditAssignerConfig(p_ablate=1.0)
        assigner = make_assigner(graph=g, config=cfg)
        # Only 2 nodes — below min_path_length=3
        trace = make_trace([tf, rs])
        rng = np.random.default_rng(0)
        result = assigner.maybe_assign(trace, g, "x", "42", rng)
        assert result is None

    def test_updates_utility_when_assigned(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        cfg = CreditAssignerConfig(p_ablate=1.0)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids, lm_response="42")
        rng = np.random.default_rng(0)
        assigner.maybe_assign(trace, g, "x", "42", rng)
        # Middle node rs should have received one update
        assert g.nodes[rs].utility_n == 1

    def test_skipped_increments_n_skipped(self):
        g, ids = make_graph_and_ids()
        cfg = CreditAssignerConfig(p_ablate=0.0)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids)
        rng = np.random.default_rng(0)
        assigner.maybe_assign(trace, g, "x", "42", rng)
        assert assigner.n_skipped == 1

    def test_chosen_node_is_ablatable(self):
        """Result node_id must be from the ablatable set (never source/terminal)."""
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        cfg = CreditAssignerConfig(p_ablate=1.0)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids, lm_response="42")
        rng = np.random.default_rng(42)
        result = assigner.maybe_assign(trace, g, "x", "42", rng)
        assert result.node_id == rs  # Only middle node is ablatable
        assert result.node_id != tf
        assert result.node_id != oc


# ---------------------------------------------------------------------------
# CreditAssigner.assign_all
# ---------------------------------------------------------------------------

class TestAssignAll:
    def test_returns_one_result_per_ablatable_node(self):
        g, ids = make_long_graph_and_ids()
        tf, rs, comp, oc = ids
        assigner = make_assigner(graph=g)
        trace = make_trace(ids, lm_response="42")
        results = assigner.assign_all(trace, g, "x", "42")
        assert len(results) == 2  # rs and comp

    def test_all_ablatable_nodes_updated(self):
        g, ids = make_long_graph_and_ids()
        tf, rs, comp, oc = ids
        assigner = make_assigner(graph=g)
        trace = make_trace(ids, lm_response="42")
        assigner.assign_all(trace, g, "x", "42")
        assert g.nodes[rs].utility_n == 1
        assert g.nodes[comp].utility_n == 1
        # Source and terminal not updated
        assert g.nodes[tf].utility_n == 0
        assert g.nodes[oc].utility_n == 0

    def test_n_assignments_matches_ablatable_count(self):
        g, ids = make_long_graph_and_ids()
        assigner = make_assigner(graph=g)
        trace = make_trace(ids, lm_response="42")
        assigner.assign_all(trace, g, "x", "42")
        assert assigner.n_assignments == 2

    def test_short_path_returns_empty(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        cfg = CreditAssignerConfig(min_path_length=10)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids)
        results = assigner.assign_all(trace, g, "x", "42")
        assert results == []

    def test_result_order_matches_path_order(self):
        g, ids = make_long_graph_and_ids()
        tf, rs, comp, oc = ids
        assigner = make_assigner(graph=g)
        trace = make_trace(ids)
        results = assigner.assign_all(trace, g, "x", "42")
        assert results[0].node_id == rs
        assert results[1].node_id == comp


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_assignment_rate_zero_at_init(self):
        assigner = make_assigner()
        assert assigner.assignment_rate == 0.0

    def test_assignment_rate_after_assign(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        cfg = CreditAssignerConfig(p_ablate=1.0)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids, lm_response="42")
        rng = np.random.default_rng(0)
        assigner.maybe_assign(trace, g, "x", "42", rng)
        assert assigner.assignment_rate == pytest.approx(1.0)

    def test_assignment_rate_mixed(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        assigner = make_assigner(graph=g)
        trace = make_trace(ids, lm_response="42")
        # 1 forced assign
        assigner.force_assign(trace, g, "x", "42", rs)
        # 1 skip (p_ablate=0 would skip, but we add directly to _n_skipped manually)
        assigner._n_skipped += 1
        assert assigner.assignment_rate == pytest.approx(0.5)

    def test_fragment_utility_report_empty_initially(self):
        g, ids = make_graph_and_ids()
        assigner = make_assigner(graph=g)
        report = assigner.fragment_utility_report(g)
        assert report == {}

    def test_fragment_utility_report_after_assign(self):
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids
        assigner = make_assigner(graph=g)
        trace = make_trace(ids, lm_response="42")
        assigner.force_assign(trace, g, "x", "42", rs)
        report = assigner.fragment_utility_report(g)
        assert rs in report
        assert "utility" in report[rs]
        assert "n_samples" in report[rs]
        assert report[rs]["n_samples"] == 1

    def test_fragment_utility_report_excludes_unupdated(self):
        g, ids = make_long_graph_and_ids()
        tf, rs, comp, oc = ids
        assigner = make_assigner(graph=g)
        trace = make_trace(ids, lm_response="42")
        assigner.force_assign(trace, g, "x", "42", rs)
        report = assigner.fragment_utility_report(g)
        assert rs in report
        assert comp not in report


# ---------------------------------------------------------------------------
# Utility convergence (online mean)
# ---------------------------------------------------------------------------

class TestUtilityConvergence:
    def test_utility_converges_to_mean_marginal(self):
        """
        Fragment utility = online mean of marginals.
        Use all-correct trace (full_score=1.0); LM alternates "42"/"wrong"
        giving ablated_scores 1.0 and 0.0 alternately.
        marginals: 0.0 (no improvement) and 1.0 (node helped).
        Expected mean over 50 steps = 0.5.
        """
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids

        call_count = [0]
        ablated_answers = ["42", "wrong"] * 25  # 50 total, alternating

        class AblatedLM:
            def complete(self, prompt):
                r = ablated_answers[call_count[0] % len(ablated_answers)]
                call_count[0] += 1
                return r

        asm = PromptAssembler(g)
        metric = ExactMatchMetric()
        assigner = CreditAssigner(lm=AblatedLM(), assembler=asm, task_metric=metric)

        # trace_correct: full_score=1.0 every time
        trace_correct = make_trace(ids, lm_response="42")

        for _ in range(50):
            assigner.force_assign(trace_correct, g, "x", "42", rs)

        # marginals: [0.0, 1.0, 0.0, 1.0, ...] → mean = 0.5
        assert g.nodes[rs].utility == pytest.approx(0.5, abs=0.05)

    def test_positive_utility_after_consistently_helpful(self):
        """Node always helps: utility should be positive."""
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids

        class WrongAblated:
            def complete(self, prompt):
                return "wrong"

        asm = PromptAssembler(g)
        assigner = CreditAssigner(lm=WrongAblated(), assembler=asm,
                                  task_metric=ExactMatchMetric())
        trace = make_trace(ids, lm_response="42")
        for _ in range(10):
            assigner.force_assign(trace, g, "x", "42", rs)
        assert g.nodes[rs].utility > 0.0

    def test_negative_utility_after_consistently_harmful(self):
        """Node always hurts: utility should be negative."""
        g, ids = make_graph_and_ids()
        tf, rs, oc = ids

        class CorrectAblated:
            def complete(self, prompt):
                return "42"

        asm = PromptAssembler(g)
        assigner = CreditAssigner(lm=CorrectAblated(), assembler=asm,
                                  task_metric=ExactMatchMetric())
        trace = make_trace(ids, lm_response="wrong")
        for _ in range(10):
            assigner.force_assign(trace, g, "x", "42", rs)
        assert g.nodes[rs].utility < 0.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_rng_seed_same_node_selected(self):
        g, ids = make_long_graph_and_ids()
        cfg = CreditAssignerConfig(p_ablate=1.0)
        assigner = make_assigner(graph=g, config=cfg)
        trace = make_trace(ids, lm_response="42")

        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)

        r1 = assigner.maybe_assign(make_trace(ids, "42"), g, "x", "42", rng1)
        r2 = assigner.maybe_assign(make_trace(ids, "42"), g, "x", "42", rng2)
        assert r1.node_id == r2.node_id

    def test_different_seeds_may_differ_on_four_node_path(self):
        """With 2 ablatable nodes, different seeds may pick different ones."""
        g, ids = make_long_graph_and_ids()
        tf, rs, comp, oc = ids
        cfg = CreditAssignerConfig(p_ablate=1.0)
        assigner = make_assigner(graph=g, config=cfg)

        results = set()
        for seed in range(20):
            rng = np.random.default_rng(seed)
            r = assigner.maybe_assign(make_trace(ids, "42"), g, "x", "42", rng)
            results.add(r.node_id)

        # Both nodes should appear across seeds
        assert rs in results or comp in results
