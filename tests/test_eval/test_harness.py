"""Tests for ppg/eval/harness.py."""

from __future__ import annotations

import pytest

from ppg.bandits.linucb import LinUCBPolicy
from ppg.core import (
    ExecutorConfig,
    FeatureExtractor,
    FragmentType,
    PPGExecutor,
    PPGraphBuilder,
)
from ppg.eval.harness import (
    SUPPORTED_BASELINES,
    BaselineMetrics,
    EvalConfig,
    EvalExample,
    EvalHarness,
    EvalReport,
)
from ppg.training.reward import ExactMatchMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FixedLM:
    def __init__(self, response: str = "42"):
        self.response = response
        self.n_calls  = 0

    def complete(self, prompt: str) -> str:
        self.n_calls += 1
        return self.response


class RoutingLM:
    """Returns different responses based on prompt length (longer = different)."""
    def complete(self, prompt: str) -> str:
        return "42" if len(prompt) < 50 else "43"


class PromptAwareLM:
    def complete(self, prompt: str) -> str:
        return "42" if "good route" in prompt else "wrong"


class EnsembleRouteLM:
    def __init__(self):
        self.n_calls = 0

    def complete(self, prompt: str) -> str:
        self.n_calls += 1
        if "good route" in prompt or "also good route" in prompt:
            return "42"
        return "wrong"


class SamplingLM:
    def __init__(self):
        self.sample_calls = 0
        self.complete_calls = 0

    def complete(self, prompt: str) -> str:
        self.complete_calls += 1
        return "42"

    def sample(self, prompt: str, n: int) -> list[str]:
        self.sample_calls += 1
        return ["wrong", "42", "42"][:n]


def make_graph():
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think step by step.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    b.connect_chain(*ids)
    return b.build(), ids


def make_branching_graph():
    """task_framing -> reasoning_style -> output_contract
                    -> compression    -> output_contract"""
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think step by step.")
    b.add_fragment(FragmentType.COMPRESSION,     "Be brief.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    tf, rs, comp, oc = ids
    b.connect(tf, rs)
    b.connect(tf, comp)
    b.connect(rs, oc)
    b.connect(comp, oc)
    return b.build(), ids


def make_calibration_graph():
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING, "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "bad route")
    b.add_fragment(FragmentType.REASONING_STYLE, "good route")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    tf, bad, good, oc = ids
    b.connect(tf, bad)
    b.connect(tf, good)
    b.connect(bad, oc)
    b.connect(good, oc)
    return b.build(), ids


def make_ensemble_graph():
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING, "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "bad route")
    b.add_fragment(FragmentType.REASONING_STYLE, "good route")
    b.add_fragment(FragmentType.REASONING_STYLE, "also good route")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    tf, bad, good, also_good, oc = ids
    for route in (bad, good, also_good):
        b.connect(tf, route)
        b.connect(route, oc)
    return b.build(), ids


def make_executor(graph, lm, policy=None):
    if policy is None:
        policy = LinUCBPolicy(graph)
    return PPGExecutor(
        graph=graph,
        selector=policy,
        lm=lm,
        feature_extractor=FeatureExtractor(),
        config=ExecutorConfig(escalation_enabled=False),
    )


def make_harness(lm=None, graph=None, ids=None, baselines=None, cfg=None):
    if graph is None:
        graph, ids = make_graph()
    if lm is None:
        lm = FixedLM("42")
    executor = make_executor(graph, lm)
    metric   = ExactMatchMetric()
    config   = cfg or EvalConfig(
        baselines=baselines or ["flat_all", "static_best", "random_gating",
                                "highest_utility"],
        static_best_path=ids,
        seed=0,
    )
    return EvalHarness(executor=executor, metric=metric, lm=lm, config=config)


def make_examples(n: int = 5, answer: str = "42") -> list[EvalExample]:
    return [EvalExample(x=f"q{i}", y_star=answer) for i in range(n)]


# ---------------------------------------------------------------------------
# EvalExample
# ---------------------------------------------------------------------------

class TestEvalExample:
    def test_defaults(self):
        ex = EvalExample(x="hi", y_star="lo")
        assert ex.constraints == []

    def test_with_constraints(self):
        ex = EvalExample(x="q", y_star="a", constraints=["bullet"])
        assert ex.constraints == ["bullet"]


# ---------------------------------------------------------------------------
# BaselineMetrics
# ---------------------------------------------------------------------------

class TestBaselineMetrics:
    def _make(self, scores, tokens, name="test"):
        return BaselineMetrics(
            name=name,
            task_scores=scores,
            token_counts=tokens,
            constraint_scores=[],
            lm_calls=len(scores),
        )

    def test_task_accuracy_mean(self):
        m = self._make([1.0, 0.0, 1.0], [10, 10, 10])
        assert m.task_accuracy == pytest.approx(2 / 3)

    def test_task_accuracy_empty(self):
        m = self._make([], [])
        assert m.task_accuracy == pytest.approx(0.0)

    def test_mean_tokens(self):
        m = self._make([1.0], [100, 200])
        assert m.mean_tokens == pytest.approx(150.0)

    def test_std_task_single(self):
        m = self._make([0.7], [10])
        assert m.std_task == pytest.approx(0.0)

    def test_std_task_multiple(self):
        m = self._make([0.0, 1.0], [10, 10])
        assert m.std_task > 0.0

    def test_mean_constraint_empty(self):
        m = self._make([0.5], [10])
        assert m.mean_constraint == pytest.approx(0.0)

    def test_mean_constraint_nonempty(self):
        m = BaselineMetrics("x", [1.0], [10], [0.8, 0.6], 1)
        assert m.mean_constraint == pytest.approx(0.7)

    def test_as_dict_keys(self):
        m = self._make([1.0, 0.5], [10, 20])
        d = m.as_dict()
        for key in ("name", "task_accuracy", "std_task", "mean_tokens",
                    "mean_constraint", "lm_calls", "n_examples"):
            assert key in d

    def test_as_dict_n_examples(self):
        m = self._make([0.5, 0.8, 1.0], [10, 10, 10])
        assert m.as_dict()["n_examples"] == 3


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------

class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert "flat_all" in cfg.baselines
        assert cfg.seed == 0

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError, match="Unknown baselines"):
            EvalConfig(baselines=["nonexistent_baseline"])

    def test_valid_baselines_accepted(self):
        cfg = EvalConfig(baselines=["flat_all", "random_gating"])
        assert cfg.baselines == ["flat_all", "random_gating"]

    def test_supported_baselines_constant(self):
        assert "flat_all"        in SUPPORTED_BASELINES
        assert "static_best"     in SUPPORTED_BASELINES
        assert "random_gating"   in SUPPORTED_BASELINES
        assert "highest_utility" in SUPPORTED_BASELINES
        assert "miprov2"         in SUPPORTED_BASELINES
        assert "gepa"            in SUPPORTED_BASELINES


# ---------------------------------------------------------------------------
# EvalHarness.evaluate — PPG metrics
# ---------------------------------------------------------------------------

class TestEvalHarnessPPG:
    def test_returns_eval_report(self):
        h = make_harness(baselines=[])
        report = h.evaluate(make_examples(3))
        assert isinstance(report, EvalReport)

    def test_empty_examples_raises(self):
        h = make_harness(baselines=[])
        with pytest.raises(ValueError):
            h.evaluate([])

    def test_ppg_n_examples_matches(self):
        h = make_harness(baselines=[])
        report = h.evaluate(make_examples(7))
        assert len(report.ppg.task_scores) == 7

    def test_ppg_correct_lm_gives_accuracy_1(self):
        h = make_harness(lm=FixedLM("42"), baselines=[])
        report = h.evaluate(make_examples(5, answer="42"))
        assert report.ppg.task_accuracy == pytest.approx(1.0)

    def test_ppg_wrong_lm_gives_accuracy_0(self):
        h = make_harness(lm=FixedLM("WRONG"), baselines=[])
        report = h.evaluate(make_examples(5, answer="42"))
        assert report.ppg.task_accuracy == pytest.approx(0.0)

    def test_ppg_lm_calls_equals_n_examples(self):
        h = make_harness(baselines=[])
        report = h.evaluate(make_examples(4))
        assert report.ppg.lm_calls == 4

    def test_ppg_token_counts_nonempty(self):
        h = make_harness(baselines=[])
        report = h.evaluate(make_examples(3))
        assert all(t > 0 for t in report.ppg.token_counts)

    def test_ppg_uses_calibrated_path_when_configured(self):
        graph, ids = make_calibration_graph()
        tf, _bad, good, oc = ids
        lm = PromptAwareLM()
        executor = make_executor(graph, lm)
        cfg = EvalConfig(baselines=[], ppg_path=[tf, good, oc])
        harness = EvalHarness(
            executor=executor,
            metric=ExactMatchMetric(),
            lm=lm,
            config=cfg,
        )
        report = harness.evaluate(make_examples(3, answer="42"))
        assert report.ppg.task_accuracy == pytest.approx(1.0)

    def test_calibrated_ppg_path_keeps_executor_sampling(self):
        graph, ids = make_graph()
        lm = SamplingLM()
        executor = PPGExecutor(
            graph=graph,
            selector=LinUCBPolicy(graph),
            lm=lm,
            feature_extractor=FeatureExtractor(),
            config=ExecutorConfig(
                escalation_enabled=True,
                k_samples=3,
                escalation_threshold=1.0,
                sample_aggregation="majority",
            ),
        )
        cfg = EvalConfig(baselines=[], ppg_path=ids)
        harness = EvalHarness(
            executor=executor,
            metric=ExactMatchMetric(),
            lm=lm,
            config=cfg,
        )

        report = harness.evaluate(make_examples(2, answer="42"))

        assert report.ppg.task_accuracy == pytest.approx(1.0)
        assert lm.sample_calls == 2
        assert lm.complete_calls == 0

    def test_ppg_path_ensemble_majority_votes_responses(self):
        graph, ids = make_ensemble_graph()
        tf, bad, good, also_good, oc = ids
        lm = EnsembleRouteLM()
        executor = make_executor(graph, lm)
        cfg = EvalConfig(
            baselines=[],
            ppg_paths=[
                [tf, bad, oc],
                [tf, good, oc],
                [tf, also_good, oc],
            ],
        )
        harness = EvalHarness(
            executor=executor,
            metric=ExactMatchMetric(),
            lm=lm,
            config=cfg,
        )

        report = harness.evaluate(make_examples(2, answer="42"))

        assert report.ppg.task_accuracy == pytest.approx(1.0)
        assert report.ppg.lm_calls == 6
        assert lm.n_calls == 6


# ---------------------------------------------------------------------------
# EvalHarness.evaluate — flat_all baseline
# ---------------------------------------------------------------------------

class TestFlatAllBaseline:
    def test_flat_all_in_report(self):
        h = make_harness(baselines=["flat_all"])
        report = h.evaluate(make_examples(3))
        assert "flat_all" in report.baselines

    def test_flat_all_n_examples(self):
        h = make_harness(baselines=["flat_all"])
        report = h.evaluate(make_examples(5))
        assert len(report.baselines["flat_all"].task_scores) == 5

    def test_flat_all_lm_calls_equals_n_examples(self):
        lm     = FixedLM("42")
        h      = make_harness(lm=lm, baselines=["flat_all"])
        report = h.evaluate(make_examples(4))
        assert report.baselines["flat_all"].lm_calls == 4

    def test_flat_all_longer_prompt_than_single_node(self):
        """Flat-all uses all nodes → prompt longer than any single-node path."""
        graph, ids = make_graph()
        lm = FixedLM("42")
        executor = make_executor(graph, lm)
        metric   = ExactMatchMetric()
        # Use flat_all and highest_utility (which may pick fewer nodes)
        cfg    = EvalConfig(baselines=["flat_all", "highest_utility"],
                            static_best_path=ids, seed=0)
        harness = EvalHarness(executor=executor, metric=metric, lm=lm, config=cfg)
        report  = harness.evaluate(make_examples(3))
        flat_tokens = report.baselines["flat_all"].mean_tokens
        ppg_tokens  = report.ppg.mean_tokens
        # flat_all should use >= tokens as PPG (uses all nodes)
        assert flat_tokens >= ppg_tokens

    def test_flat_all_correct_lm_accuracy(self):
        h = make_harness(lm=FixedLM("42"), baselines=["flat_all"])
        report = h.evaluate(make_examples(5, answer="42"))
        assert report.baselines["flat_all"].task_accuracy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EvalHarness.evaluate — static_best baseline
# ---------------------------------------------------------------------------

class TestStaticBestBaseline:
    def test_static_best_uses_supplied_path(self):
        graph, ids = make_graph()
        lm = FixedLM("42")
        executor = make_executor(graph, lm)
        metric   = ExactMatchMetric()
        cfg      = EvalConfig(baselines=["static_best"], static_best_path=ids)
        harness  = EvalHarness(executor=executor, metric=metric, lm=lm, config=cfg)
        report   = harness.evaluate(make_examples(3))
        assert "static_best" in report.baselines
        assert len(report.baselines["static_best"].task_scores) == 3

    def test_static_best_no_path_uses_utility_default(self):
        """When static_best_path=None, harness computes best utility path."""
        graph, ids = make_graph()
        lm = FixedLM("42")
        executor = make_executor(graph, lm)
        metric   = ExactMatchMetric()
        cfg      = EvalConfig(baselines=["static_best"], static_best_path=None)
        harness  = EvalHarness(executor=executor, metric=metric, lm=lm, config=cfg)
        report   = harness.evaluate(make_examples(3))
        # Should not raise; returns metrics
        assert report.baselines["static_best"].task_accuracy >= 0.0


# ---------------------------------------------------------------------------
# EvalHarness.evaluate — random_gating baseline
# ---------------------------------------------------------------------------

class TestRandomGatingBaseline:
    def test_random_gating_in_report(self):
        graph, ids = make_graph()
        h = make_harness(graph=graph, ids=ids, baselines=["random_gating"])
        report = h.evaluate(make_examples(4))
        assert "random_gating" in report.baselines

    def test_random_gating_n_examples(self):
        graph, ids = make_graph()
        h = make_harness(graph=graph, ids=ids, baselines=["random_gating"])
        report = h.evaluate(make_examples(6))
        assert len(report.baselines["random_gating"].task_scores) == 6

    def test_random_gating_lm_calls_matched(self):
        graph, ids = make_graph()
        h = make_harness(graph=graph, ids=ids, baselines=["random_gating"])
        report = h.evaluate(make_examples(3))
        assert report.baselines["random_gating"].lm_calls == 3


# ---------------------------------------------------------------------------
# EvalHarness.evaluate — highest_utility baseline
# ---------------------------------------------------------------------------

class TestHighestUtilityBaseline:
    def test_highest_utility_in_report(self):
        graph, ids = make_graph()
        h = make_harness(graph=graph, ids=ids, baselines=["highest_utility"])
        report = h.evaluate(make_examples(3))
        assert "highest_utility" in report.baselines

    def test_highest_utility_accuracy_with_correct_lm(self):
        graph, ids = make_graph()
        h = make_harness(lm=FixedLM("42"), graph=graph, ids=ids,
                         baselines=["highest_utility"])
        report = h.evaluate(make_examples(4, answer="42"))
        assert report.baselines["highest_utility"].task_accuracy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EvalHarness.evaluate — stub baselines
# ---------------------------------------------------------------------------

class TestStubBaselines:
    def test_miprov2_raises_not_implemented(self):
        graph, ids = make_graph()
        h = make_harness(graph=graph, ids=ids, baselines=["miprov2"])
        with pytest.raises(NotImplementedError, match="MIPROv2|DSPy|dspy"):
            h.evaluate(make_examples(2))

    def test_gepa_raises_not_implemented(self):
        graph, ids = make_graph()
        h = make_harness(graph=graph, ids=ids, baselines=["gepa"])
        with pytest.raises(NotImplementedError, match="GEPA"):
            h.evaluate(make_examples(2))


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------

class TestEvalReport:
    def _make_report(self, ppg_acc=0.8, baseline_accs=None):
        if baseline_accs is None:
            baseline_accs = {"flat_all": 0.6, "random_gating": 0.5}

        def _metrics(name, acc):
            n = 5
            scores = [acc] * n
            return BaselineMetrics(name=name, task_scores=scores,
                                   token_counts=[10]*n, constraint_scores=[],
                                   lm_calls=n)

        ppg = _metrics("ppg", ppg_acc)
        bases = {k: _metrics(k, v) for k, v in baseline_accs.items()}
        return EvalReport(ppg=ppg, baselines=bases)

    def test_all_metrics_includes_ppg_and_baselines(self):
        report = self._make_report()
        names = {m.name for m in report.all_metrics()}
        assert "ppg" in names
        assert "flat_all" in names

    def test_comparison_table_sorted_descending(self):
        report = self._make_report(ppg_acc=0.8,
                                   baseline_accs={"flat_all": 0.6,
                                                  "random_gating": 0.5})
        table = report.comparison_table()
        accuracies = [row["task_accuracy"] for row in table]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_winner_is_highest_accuracy(self):
        report = self._make_report(ppg_acc=0.9,
                                   baseline_accs={"flat_all": 0.6})
        assert report.winner() == "ppg"

    def test_winner_baseline_when_ppg_loses(self):
        report = self._make_report(ppg_acc=0.5,
                                   baseline_accs={"flat_all": 0.9})
        assert report.winner() == "flat_all"

    def test_ppg_delta_positive_when_ppg_wins(self):
        report = self._make_report(ppg_acc=0.8,
                                   baseline_accs={"flat_all": 0.6})
        assert report.ppg_delta("flat_all") == pytest.approx(0.2)

    def test_ppg_delta_negative_when_ppg_loses(self):
        report = self._make_report(ppg_acc=0.4,
                                   baseline_accs={"flat_all": 0.7})
        assert report.ppg_delta("flat_all") == pytest.approx(-0.3)

    def test_comparison_table_has_all_systems(self):
        report = self._make_report(
            baseline_accs={"flat_all": 0.6, "random_gating": 0.5}
        )
        table = report.comparison_table()
        names = {row["name"] for row in table}
        assert names == {"ppg", "flat_all", "random_gating"}


# ---------------------------------------------------------------------------
# Integration: full evaluate call
# ---------------------------------------------------------------------------

class TestEvalHarnessIntegration:
    def test_full_evaluate_all_baselines_except_stubs(self):
        graph, ids = make_graph()
        h = make_harness(
            lm=FixedLM("42"),
            graph=graph, ids=ids,
            baselines=["flat_all", "static_best", "random_gating",
                       "highest_utility"],
        )
        report = h.evaluate(make_examples(5, answer="42"))
        assert set(report.baselines.keys()) == {
            "flat_all", "static_best", "random_gating", "highest_utility"
        }

    def test_report_winner_with_perfect_lm(self):
        """All baselines should score 1.0 when LM is always correct."""
        graph, ids = make_graph()
        h = make_harness(
            lm=FixedLM("42"), graph=graph, ids=ids,
            baselines=["flat_all", "random_gating"],
        )
        report = h.evaluate(make_examples(5, answer="42"))
        for m in report.all_metrics():
            assert m.task_accuracy == pytest.approx(1.0), m.name

    def test_lm_calls_per_baseline_matched(self):
        """Each baseline makes exactly n_examples calls."""
        graph, ids = make_graph()
        h = make_harness(
            lm=FixedLM("42"), graph=graph, ids=ids,
            baselines=["flat_all", "static_best", "random_gating",
                       "highest_utility"],
        )
        n = 6
        report = h.evaluate(make_examples(n))
        for name, bm in report.baselines.items():
            assert bm.lm_calls == n, f"{name}: expected {n} calls, got {bm.lm_calls}"

    def test_executor_selector_unchanged_after_evaluate(self):
        """Baseline runs that swap the selector must restore it."""
        from ppg.bandits.linucb import LinUCBPolicy
        graph, ids = make_graph()
        lm     = FixedLM("42")
        policy = LinUCBPolicy(graph)
        executor = make_executor(graph, lm, policy)
        metric   = ExactMatchMetric()
        cfg      = EvalConfig(baselines=["random_gating", "highest_utility"],
                              static_best_path=ids)
        harness  = EvalHarness(executor=executor, metric=metric, lm=lm, config=cfg)
        harness.evaluate(make_examples(3))
        assert executor.selector is policy
