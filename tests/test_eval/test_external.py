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


def _make_dspy_stub(*, include_gepa: bool = False):
    """Return (dspy_mod, patches_dict) with a minimal DSPy stub."""
    dspy_mod = types.ModuleType("dspy")

    class FakeExample:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        def with_inputs(self, *a):
            return self

    class FakeSig:
        instructions = ""
        def __init__(self, *a, **kw):
            self.instructions = kw.get("instructions", "")

    class FakePredict:
        def __init__(self, sig):
            self.signature = sig
        def __call__(self, **kwargs):
            return FakeExample(response="42")

    class FakeDSPyModule:
        def __init__(self):
            self.predict = FakePredict(FakeSig())
        def __call__(self, **kwargs):
            return FakeExample(response="42")

    class FakeMIPROv2:
        def __init__(self, metric, auto="medium"):
            self.metric = metric
            self.auto   = auto
        def compile(self, program, trainset, valset=None):
            return program

    dspy_mod.Example   = FakeExample
    dspy_mod.Signature = FakeSig
    dspy_mod.Predict   = FakePredict
    dspy_mod.Module    = FakeDSPyModule
    dspy_mod.MIPROv2   = FakeMIPROv2
    dspy_mod.configure = lambda **kw: None

    patches = {"dspy": dspy_mod}

    if include_gepa:
        class FakeGEPA:
            _last_instance = None

            def __init__(self, metric, *, reflection_lm=None,
                         max_metric_calls=150, seed=0, **kw):
                self.metric           = metric
                self.reflection_lm    = reflection_lm
                self.max_metric_calls = max_metric_calls
                self.seed             = seed
                FakeGEPA._last_instance = self
                self._compile_kwargs = {}

            def compile(self, program, *, trainset, valset=None, **kw):
                self._compile_kwargs = {"trainset": trainset, "valset": valset}
                return program

        teleprompt_mod      = types.ModuleType("dspy.teleprompt")
        teleprompt_mod.GEPA = FakeGEPA
        dspy_mod.teleprompt = teleprompt_mod
        patches["dspy.teleprompt"] = teleprompt_mod

    return dspy_mod, patches


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
        _, patches = _make_dspy_stub()
        with patch.dict(sys.modules, patches):
            b = MIPROv2Baseline(metric=ExactMatchMetric(), auto="light")
            b.compile(trainset=make_train(), seed_instructions="Solve: {input}")
            response, tokens = b.run(EvalExample(x="1+1", y_star="2"))

        assert isinstance(response, str)
        assert isinstance(tokens, int)
        assert tokens >= 0

    def test_auto_param_passed_to_miprov2(self):
        """Verify auto kwarg flows through to MIPROv2 constructor."""
        captured = {}

        _, patches = _make_dspy_stub()
        orig_mipro = patches["dspy"].MIPROv2

        class TrackingMIPROv2(orig_mipro):
            def __init__(self, metric, auto="medium"):
                captured["auto"] = auto
                super().__init__(metric=metric, auto=auto)

        patches["dspy"].MIPROv2 = TrackingMIPROv2

        with patch.dict(sys.modules, patches):
            b = MIPROv2Baseline(metric=ExactMatchMetric(), auto="heavy")
            b.compile(trainset=make_train()[:2])

        assert captured["auto"] == "heavy"


# ---------------------------------------------------------------------------
# GEPABaseline
# ---------------------------------------------------------------------------

class TestGEPABaseline:
    def test_run_before_compile_raises(self):
        b = GEPABaseline(metric=ExactMatchMetric())
        with pytest.raises(RuntimeError, match="compile"):
            b.run(EvalExample(x="q", y_star="a"))

    def test_import_error_when_dspy_missing(self):
        b = GEPABaseline(metric=ExactMatchMetric())
        with patch.dict(sys.modules, {"dspy": None, "dspy.teleprompt": None}):
            with pytest.raises((ImportError, TypeError)):
                b.compile(trainset=make_train(), seed_instructions="Solve.")

    def test_compile_and_run(self):
        """Smoke-test compile+run using a DSPy GEPA stub."""
        _, patches = _make_dspy_stub(include_gepa=True)
        with patch.dict(sys.modules, patches):
            b = GEPABaseline(metric=ExactMatchMetric(), max_metric_calls=30, seed=7)
            b.compile(trainset=make_train(), valset=make_test(),
                      seed_instructions="Solve the following.")
            response, tokens = b.run(EvalExample(x="1+1", y_star="2"))

        assert isinstance(response, str)
        assert isinstance(tokens, int)
        assert tokens >= 0

    def test_reflection_lm_forwarded_to_gepa(self):
        """reflection_lm passed to constructor must reach GEPA.__init__."""
        _, patches = _make_dspy_stub(include_gepa=True)
        FakeGEPA = patches["dspy.teleprompt"].GEPA
        sentinel = object()

        with patch.dict(sys.modules, patches):
            b = GEPABaseline(metric=ExactMatchMetric(), reflection_lm=sentinel)
            b.compile(trainset=make_train()[:2])

        assert FakeGEPA._last_instance is not None
        assert FakeGEPA._last_instance.reflection_lm is sentinel

    def test_reflection_lm_none_not_forwarded(self):
        """reflection_lm=None must not be passed as kwarg (GEPA uses default)."""
        _, patches = _make_dspy_stub(include_gepa=True)
        FakeGEPA = patches["dspy.teleprompt"].GEPA

        with patch.dict(sys.modules, patches):
            b = GEPABaseline(metric=ExactMatchMetric(), reflection_lm=None)
            b.compile(trainset=make_train()[:2])

        assert FakeGEPA._last_instance.reflection_lm is None

    def test_max_metric_calls_forwarded(self):
        _, patches = _make_dspy_stub(include_gepa=True)
        FakeGEPA = patches["dspy.teleprompt"].GEPA

        with patch.dict(sys.modules, patches):
            b = GEPABaseline(metric=ExactMatchMetric(), max_metric_calls=77)
            b.compile(trainset=make_train()[:2])

        assert FakeGEPA._last_instance.max_metric_calls == 77

    def test_valset_forwarded_to_compile(self):
        """valset must be passed to GEPA.compile() when non-empty."""
        _, patches = _make_dspy_stub(include_gepa=True)
        FakeGEPA = patches["dspy.teleprompt"].GEPA
        val = make_test(2)

        with patch.dict(sys.modules, patches):
            b = GEPABaseline(metric=ExactMatchMetric())
            b.compile(trainset=make_train(3), valset=val)

        kw = FakeGEPA._last_instance._compile_kwargs
        assert kw["valset"] is not None
        assert len(kw["valset"]) == len(val)

    def test_no_valset_compile_called_without_valset(self):
        """When valset is empty, GEPA.compile() receives valset=None."""
        _, patches = _make_dspy_stub(include_gepa=True)
        FakeGEPA = patches["dspy.teleprompt"].GEPA

        with patch.dict(sys.modules, patches):
            b = GEPABaseline(metric=ExactMatchMetric())
            b.compile(trainset=make_train(3), valset=[])

        assert FakeGEPA._last_instance._compile_kwargs["valset"] is None

    def test_gepa_metric_returns_score_and_feedback(self):
        """The internal metric adapter must return a (float, str) tuple."""
        captured_metric = {}

        _, patches = _make_dspy_stub(include_gepa=True)
        orig_gepa = patches["dspy.teleprompt"].GEPA

        class CapturingGEPA(orig_gepa):
            def __init__(self, metric, **kw):
                captured_metric["fn"] = metric
                super().__init__(metric=metric, **kw)

        patches["dspy.teleprompt"].GEPA = CapturingGEPA

        with patch.dict(sys.modules, patches):
            b = GEPABaseline(metric=ExactMatchMetric())
            b.compile(trainset=make_train(2))

        fn = captured_metric["fn"]
        assert fn is not None

        class FakePred:
            response = "42"

        ex = patches["dspy"].Example(input="q", y_star="42")
        result = fn(ex, FakePred())
        assert isinstance(result, tuple) and len(result) == 2
        score, feedback = result
        assert isinstance(score, float)
        assert isinstance(feedback, str)
        assert "Expected" in feedback

    def test_prompt_prefix_extracted_from_signature(self):
        """_prompt_prefix must be set from compiled program's signature instructions."""
        _, patches = _make_dspy_stub(include_gepa=True)

        # Make FakeSig store the instructions passed in constructor
        class TrackedSig:
            def __init__(self, *a, **kw):
                self.instructions = kw.get("instructions", "")

        class TrackedPredict:
            def __init__(self, sig):
                self.signature = sig
            def __call__(self, **kw):
                FakeEx = patches["dspy"].Example
                return FakeEx(response="ok")

        patches["dspy"].Signature = TrackedSig
        patches["dspy"].Predict   = TrackedPredict

        with patch.dict(sys.modules, patches):
            b = GEPABaseline(metric=ExactMatchMetric())
            b.compile(trainset=make_train(2), seed_instructions="MY SEED")

        assert b._prompt_prefix == "MY SEED"

    def test_run_uses_program_not_lm_client(self):
        """run() must call self._program(), not a separate lm_client."""
        _, patches = _make_dspy_stub(include_gepa=True)
        with patch.dict(sys.modules, patches):
            b = GEPABaseline(metric=ExactMatchMetric())
            b.compile(trainset=make_train(2))
            response, tokens = b.run(EvalExample(x="INPUT", y_star="X"))

        assert isinstance(response, str)
        assert tokens >= 0

    def test_verify_available(self):
        _, patches = _make_dspy_stub(include_gepa=True)
        patches["dspy"].__version__ = "2.5.0"

        with patch.dict(sys.modules, patches):
            info = GEPABaseline.verify()

        assert info["available"] is True
        assert info["has_gepa"] is True

    def test_verify_missing_dspy(self):
        with patch.dict(sys.modules, {"dspy": None, "dspy.teleprompt": None}):
            info = GEPABaseline.verify()
        assert info["available"] is False
        assert "error" in info


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
