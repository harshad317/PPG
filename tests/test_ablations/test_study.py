"""Tests for ppg/ablations/study.py."""

from __future__ import annotations

import pytest

from ppg.ablations.study import (
    ABLATIONS,
    AblationConfig,
    AblationReport,
    AblationResult,
    AblationStudy,
    available_ablations,
    build_ablation_components,
)
from ppg.bandits.linucb import LinUCBPolicy
from ppg.core.executor import RandomSelector
from ppg.data.fragments import build_graph
from ppg.eval.harness import BaselineMetrics, EvalExample
from ppg.training.reward import ExactMatchMetric
from ppg.training.trainer import TrainerConfig, TrainingExample


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


FAST_TRAINER_CFG = TrainerConfig(
    n_warmup_episodes=2,
    n_train_episodes=3,
    n_finetune_episodes=1,
    seed=0,
)

SMALL_TRAIN = [TrainingExample(x=f"q{i}", y_star="42") for i in range(5)]
SMALL_TEST  = [EvalExample(x=f"q{i}", y_star="42") for i in range(3)]


def make_study(
    ablations=None,
    correct_response="42",
    train=None,
    test=None,
):
    return AblationStudy(
        lm=FixedLM(correct_response),
        metric=ExactMatchMetric(),
        train_dataset=train or SMALL_TRAIN,
        test_dataset=test or SMALL_TEST,
        benchmark="gsm8k",
        ablations=ablations or ["ppg_full"],
        trainer_cfg=FAST_TRAINER_CFG,
    )


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------

class TestAblationConfig:
    def test_credit_disabled_when_p_ablate_zero(self):
        cfg = AblationConfig("x", "desc", p_ablate=0.0)
        assert cfg.credit_disabled is True

    def test_credit_enabled_when_p_ablate_nonzero(self):
        cfg = AblationConfig("x", "desc", p_ablate=0.15)
        assert cfg.credit_disabled is False

    def test_bandit_disabled_when_use_random(self):
        cfg = AblationConfig("x", "desc", use_random=True)
        assert cfg.bandit_disabled is True

    def test_bandit_enabled_by_default(self):
        cfg = AblationConfig("x", "desc")
        assert cfg.bandit_disabled is False


# ---------------------------------------------------------------------------
# ABLATIONS registry
# ---------------------------------------------------------------------------

class TestAblationsRegistry:
    def test_all_expected_ablations_present(self):
        expected = {"ppg_full", "no_credit", "no_variance",
                    "no_bandit", "lean_topology"}
        assert expected.issubset(set(ABLATIONS.keys()))

    def test_ppg_full_has_all_components_enabled(self):
        cfg = ABLATIONS["ppg_full"]
        assert cfg.p_ablate > 0.0
        assert cfg.skip_variance is False
        assert cfg.use_random is False
        assert cfg.topology == "rich"

    def test_no_credit_disables_only_credit(self):
        cfg = ABLATIONS["no_credit"]
        assert cfg.p_ablate == 0.0
        assert cfg.skip_variance is False
        assert cfg.use_random is False

    def test_no_variance_disables_only_variance(self):
        cfg = ABLATIONS["no_variance"]
        assert cfg.skip_variance is True
        assert cfg.p_ablate > 0.0
        assert cfg.use_random is False

    def test_no_bandit_uses_random(self):
        cfg = ABLATIONS["no_bandit"]
        assert cfg.use_random is True
        assert cfg.skip_variance is False

    def test_lean_topology_uses_lean_graph(self):
        cfg = ABLATIONS["lean_topology"]
        assert cfg.topology == "lean"
        assert cfg.use_random is False

    def test_each_config_has_unique_name(self):
        names = [cfg.name for cfg in ABLATIONS.values()]
        assert len(names) == len(set(names))

    def test_each_config_has_description(self):
        for cfg in ABLATIONS.values():
            assert cfg.description.strip()


# ---------------------------------------------------------------------------
# available_ablations
# ---------------------------------------------------------------------------

class TestAvailableAblations:
    def test_returns_sorted_list(self):
        result = available_ablations()
        assert result == sorted(result)

    def test_includes_all_registry_keys(self):
        result = available_ablations()
        for key in ABLATIONS:
            assert key in result


# ---------------------------------------------------------------------------
# build_ablation_components
# ---------------------------------------------------------------------------

class TestBuildAblationComponents:
    def test_returns_five_tuple(self):
        lm = FixedLM()
        result = build_ablation_components(
            config=ABLATIONS["ppg_full"],
            lm=lm,
            metric=ExactMatchMetric(),
            benchmark="gsm8k",
            trainer_cfg=FAST_TRAINER_CFG,
        )
        assert len(result) == 5

    def test_ppg_full_uses_linucb_selector(self):
        lm = FixedLM()
        executor, policy, *_ = build_ablation_components(
            config=ABLATIONS["ppg_full"],
            lm=lm,
            metric=ExactMatchMetric(),
            benchmark="gsm8k",
            trainer_cfg=FAST_TRAINER_CFG,
        )
        assert isinstance(executor.selector, LinUCBPolicy)

    def test_no_bandit_uses_random_selector(self):
        lm = FixedLM()
        executor, *_ = build_ablation_components(
            config=ABLATIONS["no_bandit"],
            lm=lm,
            metric=ExactMatchMetric(),
            benchmark="gsm8k",
            trainer_cfg=FAST_TRAINER_CFG,
        )
        assert isinstance(executor.selector, RandomSelector)

    def test_no_credit_sets_p_ablate_zero(self):
        lm = FixedLM()
        _, _, _, credit, _ = build_ablation_components(
            config=ABLATIONS["no_credit"],
            lm=lm,
            metric=ExactMatchMetric(),
            benchmark="gsm8k",
            trainer_cfg=FAST_TRAINER_CFG,
        )
        assert credit.cfg.p_ablate == pytest.approx(0.0)

    def test_no_variance_sets_skip_variance(self):
        lm = FixedLM()
        _, _, reward, _, _ = build_ablation_components(
            config=ABLATIONS["no_variance"],
            lm=lm,
            metric=ExactMatchMetric(),
            benchmark="gsm8k",
            trainer_cfg=FAST_TRAINER_CFG,
        )
        assert reward.cfg.skip_variance is True

    def test_lean_topology_builds_lean_graph(self):
        lm = FixedLM()
        executor, *_ = build_ablation_components(
            config=ABLATIONS["lean_topology"],
            lm=lm,
            metric=ExactMatchMetric(),
            benchmark="gsm8k",
            trainer_cfg=FAST_TRAINER_CFG,
        )
        # Lean graph has 3 nodes
        assert len(executor.graph.nodes) == 3

    def test_custom_graph_overrides_benchmark(self):
        lm    = FixedLM()
        graph = build_graph("mbpp", topology="lean")
        executor, *_ = build_ablation_components(
            config=ABLATIONS["ppg_full"],
            lm=lm,
            metric=ExactMatchMetric(),
            graph=graph,
            benchmark="gsm8k",   # ignored when graph provided
            trainer_cfg=FAST_TRAINER_CFG,
        )
        # Graph has mbpp fragments (just check it used the provided graph)
        assert executor.graph is graph


# ---------------------------------------------------------------------------
# AblationReport
# ---------------------------------------------------------------------------

class TestAblationReport:
    def _make_result(self, name, accuracy):
        cfg = AblationConfig(name, f"desc {name}")
        scores = [accuracy] * 5
        metrics = BaselineMetrics(
            name=name, task_scores=scores, token_counts=[10]*5,
            constraint_scores=[], lm_calls=5,
        )
        return AblationResult(config=cfg, metrics=metrics)

    def test_table_sorted_descending(self):
        report = AblationReport(results=[
            self._make_result("a", 0.5),
            self._make_result("b", 0.9),
            self._make_result("c", 0.7),
        ])
        table = report.table()
        accs = [row["task_accuracy"] for row in table]
        assert accs == sorted(accs, reverse=True)

    def test_table_has_description(self):
        report = AblationReport(results=[self._make_result("a", 0.5)])
        assert "description" in report.table()[0]

    def test_get_by_name(self):
        r = self._make_result("no_credit", 0.7)
        report = AblationReport(results=[r])
        assert report.get("no_credit") is r
        assert report.get("missing") is None

    def test_winner_highest_accuracy(self):
        report = AblationReport(results=[
            self._make_result("ppg_full", 0.9),
            self._make_result("no_credit", 0.7),
        ])
        assert report.winner() == "ppg_full"

    def test_delta_vs_full_positive(self):
        report = AblationReport(results=[
            self._make_result("ppg_full", 0.8),
            self._make_result("no_credit", 0.6),
        ])
        assert report.delta_vs_full("no_credit") == pytest.approx(0.2)

    def test_delta_vs_full_none_when_full_missing(self):
        report = AblationReport(results=[
            self._make_result("no_credit", 0.6),
        ])
        assert report.delta_vs_full("no_credit") is None

    def test_delta_vs_full_none_when_ablation_missing(self):
        report = AblationReport(results=[
            self._make_result("ppg_full", 0.8),
        ])
        assert report.delta_vs_full("no_credit") is None


# ---------------------------------------------------------------------------
# AblationStudy — construction
# ---------------------------------------------------------------------------

class TestAblationStudyConstruction:
    def test_unknown_ablation_raises(self):
        with pytest.raises(ValueError, match="Unknown ablations"):
            AblationStudy(
                lm=FixedLM(), metric=ExactMatchMetric(),
                train_dataset=SMALL_TRAIN, test_dataset=SMALL_TEST,
                ablations=["nonexistent"],
            )

    def test_valid_ablations_accepted(self):
        study = make_study(ablations=["ppg_full", "no_credit"])
        assert study is not None


# ---------------------------------------------------------------------------
# AblationStudy.run — correctness
# ---------------------------------------------------------------------------

class TestAblationStudyRun:
    def test_empty_train_raises(self):
        study = AblationStudy(
            lm=FixedLM(), metric=ExactMatchMetric(),
            train_dataset=[],        # explicitly empty
            test_dataset=SMALL_TEST,
            ablations=["ppg_full"], trainer_cfg=FAST_TRAINER_CFG,
        )
        with pytest.raises(ValueError):
            study.run()

    def test_empty_test_raises(self):
        study = AblationStudy(
            lm=FixedLM(), metric=ExactMatchMetric(),
            train_dataset=SMALL_TRAIN,
            test_dataset=[],         # explicitly empty
            ablations=["ppg_full"], trainer_cfg=FAST_TRAINER_CFG,
        )
        with pytest.raises(ValueError):
            study.run()

    def test_returns_ablation_report(self):
        study  = make_study(ablations=["ppg_full"])
        report = study.run()
        assert isinstance(report, AblationReport)

    def test_report_has_all_requested_ablations(self):
        study  = make_study(ablations=["ppg_full", "no_credit"])
        report = study.run()
        names  = {r.config.name for r in report.results}
        assert names == {"ppg_full", "no_credit"}

    def test_metrics_n_examples_matches_test_size(self):
        study  = make_study(ablations=["ppg_full"])
        report = study.run()
        assert len(report.results[0].metrics.task_scores) == len(SMALL_TEST)

    def test_correct_lm_gives_high_accuracy(self):
        study  = make_study(ablations=["ppg_full"], correct_response="42")
        report = study.run()
        assert report.results[0].metrics.task_accuracy == pytest.approx(1.0)

    def test_wrong_lm_gives_zero_accuracy(self):
        study  = make_study(ablations=["ppg_full"], correct_response="WRONG")
        report = study.run()
        assert report.results[0].metrics.task_accuracy == pytest.approx(0.0)

    def test_no_credit_runs_without_error(self):
        study  = make_study(ablations=["no_credit"])
        report = study.run()
        assert report.get("no_credit") is not None

    def test_no_variance_runs_without_error(self):
        study  = make_study(ablations=["no_variance"])
        report = study.run()
        assert report.get("no_variance") is not None

    def test_no_bandit_runs_without_error(self):
        study  = make_study(ablations=["no_bandit"])
        report = study.run()
        assert report.get("no_bandit") is not None

    def test_lean_topology_runs_without_error(self):
        study  = make_study(ablations=["lean_topology"])
        report = study.run()
        assert report.get("lean_topology") is not None

    def test_lm_calls_nonzero(self):
        study  = make_study(ablations=["ppg_full"])
        report = study.run()
        assert report.results[0].metrics.lm_calls > 0

    def test_ablation_name_in_metrics(self):
        study  = make_study(ablations=["no_credit"])
        report = study.run()
        assert report.results[0].metrics.name == "no_credit"


# ---------------------------------------------------------------------------
# AblationStudy — on_ablation callback
# ---------------------------------------------------------------------------

class TestOnAblationCallback:
    def test_callback_called_once_per_ablation(self):
        calls = []

        def cb(name, result):
            calls.append(name)

        study = AblationStudy(
            lm=FixedLM(), metric=ExactMatchMetric(),
            train_dataset=SMALL_TRAIN, test_dataset=SMALL_TEST,
            ablations=["ppg_full", "no_credit"],
            trainer_cfg=FAST_TRAINER_CFG,
            on_ablation=cb,
        )
        study.run()
        assert set(calls) == {"ppg_full", "no_credit"}

    def test_callback_receives_ablation_result(self):
        received = []

        def cb(name, result):
            received.append(result)

        study = AblationStudy(
            lm=FixedLM(), metric=ExactMatchMetric(),
            train_dataset=SMALL_TRAIN, test_dataset=SMALL_TEST,
            ablations=["ppg_full"],
            trainer_cfg=FAST_TRAINER_CFG,
            on_ablation=cb,
        )
        study.run()
        assert len(received) == 1
        assert isinstance(received[0], AblationResult)


# ---------------------------------------------------------------------------
# Integration: all 5 ablations run end-to-end
# ---------------------------------------------------------------------------

class TestAllAblationsIntegration:
    def test_all_ablations_complete(self):
        """Smoke test: all 5 ablations run without error."""
        study = AblationStudy(
            lm=FixedLM("42"),
            metric=ExactMatchMetric(),
            train_dataset=SMALL_TRAIN,
            test_dataset=SMALL_TEST,
            benchmark="gsm8k",
            ablations=list(ABLATIONS.keys()),
            trainer_cfg=FAST_TRAINER_CFG,
        )
        report = study.run()
        assert len(report.results) == len(ABLATIONS)

    def test_table_has_all_ablations(self):
        study = AblationStudy(
            lm=FixedLM("42"),
            metric=ExactMatchMetric(),
            train_dataset=SMALL_TRAIN,
            test_dataset=SMALL_TEST,
            benchmark="gsm8k",
            ablations=list(ABLATIONS.keys()),
            trainer_cfg=FAST_TRAINER_CFG,
        )
        report = study.run()
        table_names = {row["name"] for row in report.table()}
        assert table_names == set(ABLATIONS.keys())
