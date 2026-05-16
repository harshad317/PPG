"""Tests for ppg/eval/external.py — MIPROv2Baseline and GEPABaseline."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from ppg.eval.external import GEPABaseline, MIPROv2Baseline
from ppg.eval.harness import EvalExample
from ppg.training.reward import ExactMatchMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FixedLM:
    def __init__(self, response: str = "42"):
        self.response = response
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


def make_train(n: int = 5):
    return [EvalExample(x=f"What is {i}+{i}?", y_star=str(i * 2)) for i in range(n)]


def make_test(n: int = 3):
    return [EvalExample(x=f"Q{i}", y_star=f"A{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# MIPROv2Baseline
# ---------------------------------------------------------------------------

class TestMIPROv2Baseline:
    def test_run_before_compile_raises(self):
        b = MIPROv2Baseline(metric=ExactMatchMetric())
        with pytest.raises(RuntimeError, match="compile"):
            b.run(EvalExample(x="hi", y_star="there"))

    def test_import_error_when_dspy_missing(self):
        b = MIPROv2Baseline(metric=ExactMatchMetric())
        with patch.dict(sys.modules, {"dspy": None}):
            with pytest.raises(ImportError, match="dspy"):
                b.compile(trainset=make_train())

    def test_compile_and_run(self):
        """Smoke-test compile+run using a mock dspy module."""
        # Build a minimal dspy stub
        dspy_mod = types.ModuleType("dspy")

        class FakeExample:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            def with_inputs(self, *args):
                return self

        class FakeSig:
            def __init__(self, *args, **kwargs): pass
            def with_instructions(self, text): return self

        class FakePredict:
            def __init__(self, sig):
                self.signature = sig
            def __call__(self, **kwargs):
                return FakeExample(response="42")

        class FakeDSPyModule:
            def __init__(self): pass
            def __call__(self, **kwargs):
                return FakeExample(response="42")

        class FakeMIPROv2:
            def __init__(self, metric, auto="medium"):
                self.metric = metric
                self.auto   = auto
            def compile(self, program, trainset):
                return program

        dspy_mod.Example   = FakeExample
        dspy_mod.Signature = FakeSig
        dspy_mod.Predict   = FakePredict
        dspy_mod.Module    = FakeDSPyModule
        dspy_mod.MIPROv2   = FakeMIPROv2
        dspy_mod.configure = lambda **kwargs: None

        with patch.dict(sys.modules, {"dspy": dspy_mod}):
            b = MIPROv2Baseline(metric=ExactMatchMetric(), auto="light")
            b.compile(trainset=make_train(), seed_instructions="Solve: {input}")
            response, tokens = b.run(EvalExample(x="1+1", y_star="2"))

        assert isinstance(response, str)
        assert isinstance(tokens, int)
        assert tokens >= 0

    def test_auto_param_passed_to_miprov2(self):
        """Verify auto kwarg flows through to MIPROv2 constructor."""
        captured = {}

        dspy_mod = types.ModuleType("dspy")

        class FakeExample:
            def __init__(self, **kw):
                [setattr(self, k, v) for k, v in kw.items()]
            def with_inputs(self, *a):
                return self

        class FakeSig:
            def __init__(self, *a, **kw): pass
            def with_instructions(self, t): return self

        class FakeDSPyModule:
            def __init__(self): pass

        class FakeMIPROv2:
            def __init__(self, metric, auto="medium"):
                captured["auto"] = auto
            def compile(self, program, trainset):
                return program

        dspy_mod.Example   = FakeExample
        dspy_mod.Signature = FakeSig
        dspy_mod.Predict   = lambda sig: MagicMock(signature=sig)
        dspy_mod.Module    = FakeDSPyModule
        dspy_mod.MIPROv2   = FakeMIPROv2
        dspy_mod.configure = lambda **kw: None

        with patch.dict(sys.modules, {"dspy": dspy_mod}):
            b = MIPROv2Baseline(metric=ExactMatchMetric(), auto="heavy")
            b.compile(trainset=make_train()[:2])

        assert captured["auto"] == "heavy"


# ---------------------------------------------------------------------------
# GEPABaseline
# ---------------------------------------------------------------------------

class TestGEPABaseline:
    def test_run_before_compile_raises(self):
        b = GEPABaseline(metric=ExactMatchMetric(), lm_client=FixedLM())
        with pytest.raises(RuntimeError, match="compile"):
            b.run(EvalExample(x="q", y_star="a"))

    def test_import_error_when_gepa_missing(self):
        b = GEPABaseline(metric=ExactMatchMetric(), lm_client=FixedLM())
        with patch.dict(sys.modules, {
            "gepa": None,
            "gepa.optimize_anything": None,
        }):
            with pytest.raises((ImportError, TypeError)):
                b.compile(
                    trainset=make_train(),
                    valset=make_test(),
                    seed_prompt="Solve the following problem.",
                )

    def test_compile_calls_optimize_anything(self):
        """Verify optimize_anything is called with correct seed and evaluator."""
        captured = {}

        class FakeResult:
            best_candidate = "Optimized prompt text"

        def fake_optimize_anything(seed_candidate, evaluator, objective, config):
            captured["seed"]      = seed_candidate
            captured["objective"] = objective
            captured["config"]    = config
            score = evaluator(seed_candidate)
            captured["score"] = score
            return FakeResult()

        class FakeEngineConfig:
            def __init__(self, max_metric_calls=100, reflection_lm=None):
                self.max_metric_calls = max_metric_calls
                self.reflection_lm   = reflection_lm

        class FakeGEPAConfig:
            def __init__(self, engine=None):
                self.engine = engine

        gepa_mod      = types.ModuleType("gepa")
        gepa_oa_mod   = types.ModuleType("gepa.optimize_anything")
        gepa_oa_mod.optimize_anything = fake_optimize_anything
        gepa_oa_mod.GEPAConfig        = FakeGEPAConfig
        gepa_oa_mod.EngineConfig      = FakeEngineConfig
        gepa_oa_mod.log               = lambda msg: None

        with patch.dict(sys.modules, {
            "gepa": gepa_mod,
            "gepa.optimize_anything": gepa_oa_mod,
        }):
            lm = FixedLM("42")
            b  = GEPABaseline(
                metric=ExactMatchMetric(),
                lm_client=lm,
                reflection_lm="openai/gpt-4o",
                max_metric_calls=50,
                n_eval_examples=3,
                seed=0,
            )
            b.compile(
                trainset=make_train(),
                valset=make_test(),
                seed_prompt="Initial prompt.",
                objective="Maximize accuracy.",
            )

        assert captured["seed"] == "Initial prompt."
        assert captured["objective"] == "Maximize accuracy."
        assert isinstance(captured["score"], float)
        assert 0.0 <= captured["score"] <= 1.0
        # reflection_lm must reach EngineConfig
        assert captured["config"].engine.reflection_lm == "openai/gpt-4o"
        assert captured["config"].engine.max_metric_calls == 50

    def test_evaluator_samples_valset_not_trainset(self):
        """Evaluator must use valset examples when valset is non-empty."""
        seen_inputs: list[str] = []

        class FakeResult:
            best_candidate = "p"

        def fake_optimize_anything(seed_candidate, evaluator, objective, config):
            evaluator(seed_candidate)
            return FakeResult()

        class FakeEngineConfig:
            def __init__(self, max_metric_calls=100, reflection_lm=None): pass

        class FakeGEPAConfig:
            def __init__(self, engine=None): pass

        gepa_mod    = types.ModuleType("gepa")
        gepa_oa_mod = types.ModuleType("gepa.optimize_anything")
        gepa_oa_mod.optimize_anything = fake_optimize_anything
        gepa_oa_mod.GEPAConfig        = FakeGEPAConfig
        gepa_oa_mod.EngineConfig      = FakeEngineConfig
        gepa_oa_mod.log               = lambda msg: None

        # trainset inputs start with "What is", valset inputs start with "Q"
        trainset = make_train(5)
        valset   = make_test(3)

        class TrackingLM:
            def complete(self, prompt: str) -> str:
                seen_inputs.append(prompt)
                return "0"

        with patch.dict(sys.modules, {
            "gepa": gepa_mod,
            "gepa.optimize_anything": gepa_oa_mod,
        }):
            b = GEPABaseline(
                metric=ExactMatchMetric(),
                lm_client=TrackingLM(),
                n_eval_examples=10,  # more than valset size — takes all 3
                seed=0,
            )
            b.compile(trainset=trainset, valset=valset,
                      seed_prompt="p", objective="o")

        # All LM calls must use valset inputs (Q0, Q1, Q2), not trainset inputs
        for prompt in seen_inputs:
            assert any(f"Q{i}" in prompt for i in range(3)), (
                f"Expected valset input in prompt, got: {prompt!r}"
            )

    def test_evaluator_falls_back_to_trainset_when_valset_empty(self):
        """When valset is empty, evaluator uses trainset."""
        seen_inputs: list[str] = []

        class FakeResult:
            best_candidate = "p"

        def fake_optimize_anything(seed_candidate, evaluator, objective, config):
            evaluator(seed_candidate)
            return FakeResult()

        class FakeEngineConfig:
            def __init__(self, max_metric_calls=100, reflection_lm=None): pass

        class FakeGEPAConfig:
            def __init__(self, engine=None): pass

        gepa_mod    = types.ModuleType("gepa")
        gepa_oa_mod = types.ModuleType("gepa.optimize_anything")
        gepa_oa_mod.optimize_anything = fake_optimize_anything
        gepa_oa_mod.GEPAConfig        = FakeGEPAConfig
        gepa_oa_mod.EngineConfig      = FakeEngineConfig
        gepa_oa_mod.log               = lambda msg: None

        class TrackingLM:
            def complete(self, prompt: str) -> str:
                seen_inputs.append(prompt)
                return "0"

        with patch.dict(sys.modules, {
            "gepa": gepa_mod,
            "gepa.optimize_anything": gepa_oa_mod,
        }):
            b = GEPABaseline(
                metric=ExactMatchMetric(),
                lm_client=TrackingLM(),
                n_eval_examples=3,
                seed=0,
            )
            b.compile(trainset=make_train(5), valset=[],
                      seed_prompt="p", objective="o")

        assert len(seen_inputs) > 0  # evaluator ran using trainset

    def test_run_uses_optimized_prompt(self):
        """run() prepends optimized prompt to input before calling lm."""
        b = GEPABaseline(metric=ExactMatchMetric(), lm_client=FixedLM("answer"))
        b._optimized_prompt = "USE THIS PROMPT"

        response, tokens = b.run(EvalExample(x="test question", y_star="answer"))

        assert response == "answer"
        assert tokens > 0

    def test_run_lm_called_with_prompt_plus_input(self):
        lm = FixedLM("answer")
        b  = GEPABaseline(metric=ExactMatchMetric(), lm_client=lm)
        b._optimized_prompt = "PREFIX"

        b.run(EvalExample(x="MY_INPUT", y_star="x"))

        assert len(lm.calls) == 1
        assert "PREFIX" in lm.calls[0]
        assert "MY_INPUT" in lm.calls[0]

    def test_best_candidate_dict_extracted(self):
        """When best_candidate is a dict, extract system_prompt key."""
        b = GEPABaseline(metric=ExactMatchMetric(), lm_client=FixedLM())

        class FakeResult:
            best_candidate = {"system_prompt": "extracted prompt"}

        class FakeEngineConfig:
            def __init__(self, max_metric_calls=100, reflection_lm=None): pass

        class FakeGEPAConfig:
            def __init__(self, engine=None): pass

        gepa_mod    = types.ModuleType("gepa")
        gepa_oa_mod = types.ModuleType("gepa.optimize_anything")
        gepa_oa_mod.optimize_anything = lambda seed_candidate, evaluator, objective, config: FakeResult()
        gepa_oa_mod.GEPAConfig        = FakeGEPAConfig
        gepa_oa_mod.EngineConfig      = FakeEngineConfig
        gepa_oa_mod.log               = lambda msg: None

        with patch.dict(sys.modules, {
            "gepa": gepa_mod,
            "gepa.optimize_anything": gepa_oa_mod,
        }):
            b.compile(
                trainset=make_train()[:2],
                valset=make_test()[:1],
                seed_prompt="s",
            )

        assert b._optimized_prompt == "extracted prompt"


# ---------------------------------------------------------------------------
# EvalHarness external baseline routing
# ---------------------------------------------------------------------------

class TestHarnessExternalBaselines:
    """Verify the harness routes correctly to pre-compiled external baselines."""

    def _make_harness(self, external=None):
        from ppg.bandits.linucb import LinUCBPolicy
        from ppg.core import ExecutorConfig, FeatureExtractor, FragmentType, PPGExecutor, PPGraphBuilder
        from ppg.eval.harness import EvalConfig, EvalHarness

        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING,    "Task: {input}")
        b.add_fragment(FragmentType.REASONING_STYLE, "Think.")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
        ids = b.node_ids()
        b.connect_chain(*ids)
        graph = b.build()

        policy   = LinUCBPolicy(graph)
        executor = PPGExecutor(
            graph=graph,
            selector=policy,
            lm=FixedLM("42"),
            feature_extractor=FeatureExtractor(),
            config=ExecutorConfig(),
        )
        harness = EvalHarness(
            executor=executor,
            metric=ExactMatchMetric(),
            lm=FixedLM("42"),
            config=EvalConfig(baselines=list(external.keys()) if external else [],
                              show_progress=False),
            external_baselines=external,
        )
        return harness

    def test_external_baseline_run_called(self):
        class StubBaseline:
            calls = 0
            def run(self, example):
                StubBaseline.calls += 1
                return "42", 10

        stub    = StubBaseline()
        harness = self._make_harness(external={"miprov2": stub})
        examples = [EvalExample(x="q", y_star="42")] * 3
        report   = harness.evaluate(examples)

        assert "miprov2" in report.baselines
        assert StubBaseline.calls == 3
        assert report.baselines["miprov2"].task_accuracy == 1.0

    def test_notimplemented_without_external(self):
        from ppg.eval.harness import EvalConfig, EvalHarness
        harness = self._make_harness(external=None)
        # Re-configure with miprov2 in baselines but no external provided
        from ppg.bandits.linucb import LinUCBPolicy
        from ppg.core import ExecutorConfig, FeatureExtractor, FragmentType, PPGExecutor, PPGraphBuilder

        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING, "T: {input}")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT, "A:")
        ids = b.node_ids()
        b.connect_chain(*ids)
        graph    = b.build()
        policy   = LinUCBPolicy(graph)
        executor = PPGExecutor(
            graph=graph, selector=policy, lm=FixedLM(),
            feature_extractor=FeatureExtractor(), config=ExecutorConfig(),
        )
        harness2 = EvalHarness(
            executor=executor,
            metric=ExactMatchMetric(),
            lm=FixedLM(),
            config=EvalConfig(baselines=["miprov2"], show_progress=False),
        )
        with pytest.raises(NotImplementedError, match="MIPROv2"):
            harness2.evaluate([EvalExample(x="q", y_star="a")])

    def test_gepa_not_implemented_without_external(self):
        from ppg.bandits.linucb import LinUCBPolicy
        from ppg.core import ExecutorConfig, FeatureExtractor, FragmentType, PPGExecutor, PPGraphBuilder
        from ppg.eval.harness import EvalConfig, EvalHarness

        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING, "T: {input}")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT, "A:")
        ids = b.node_ids()
        b.connect_chain(*ids)
        graph    = b.build()
        policy   = LinUCBPolicy(graph)
        executor = PPGExecutor(
            graph=graph, selector=policy, lm=FixedLM(),
            feature_extractor=FeatureExtractor(), config=ExecutorConfig(),
        )
        harness = EvalHarness(
            executor=executor,
            metric=ExactMatchMetric(),
            lm=FixedLM(),
            config=EvalConfig(baselines=["gepa"], show_progress=False),
        )
        with pytest.raises(NotImplementedError, match="GEPA"):
            harness.evaluate([EvalExample(x="q", y_star="a")])
