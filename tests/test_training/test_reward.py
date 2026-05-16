"""Tests for ppg/training/reward.py."""

import numpy as np
import pytest

from ppg.core import (
    ExecutorConfig,
    FeatureExtractor,
    FragmentType,
    PPGExecutor,
    PPGraphBuilder,
    RandomSelector,
)
from ppg.training import (
    ExactMatchMetric,
    F1Metric,
    IFBenchConstraintChecker,
    KeywordConstraintChecker,
    MultipleChoiceMetric,
    NumericExactMatchMetric,
    PerturbationBuffer,
    RewardComponents,
    RewardComputer,
    RewardConfig,
    SubstringMatchMetric,
    TruncationPerturbator,
    WordShufflePerturbator,
    CompositePerturbator,
    default_perturbator,
    METRIC_REGISTRY,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FixedLM:
    """Always returns the same response."""
    def __init__(self, response: str = "42"):
        self.response = response
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        return self.response


class AlternatingLM:
    """Alternates between two responses."""
    def __init__(self, a: str = "correct", b: str = "wrong"):
        self.responses = [a, b]
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        r = self.responses[self.call_count % 2]
        self.call_count += 1
        return r


def make_graph_and_executor(lm):
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Solve: {input}")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    b.connect(*ids)
    g = b.build()
    executor = PPGExecutor(g, RandomSelector(0), lm, FeatureExtractor(),
                           ExecutorConfig())
    return g, executor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestExactMatchMetric:
    m = ExactMatchMetric()

    def test_exact_match(self):
        assert self.m.score("42", "42") == 1.0

    def test_case_insensitive(self):
        assert self.m.score("Paris", "paris") == 1.0

    def test_punctuation_stripped(self):
        assert self.m.score("42.", "42") == 1.0

    def test_extra_whitespace(self):
        assert self.m.score("  42  ", "42") == 1.0

    def test_mismatch(self):
        assert self.m.score("41", "42") == 0.0

    def test_empty_both(self):
        assert self.m.score("", "") == 1.0

    def test_empty_prediction(self):
        assert self.m.score("", "42") == 0.0


class TestNumericExactMatchMetric:
    m = NumericExactMatchMetric()

    def test_extracts_last_number(self):
        assert self.m.score("The answer is 42.", "42") == 1.0

    def test_extracts_decimal(self):
        assert self.m.score("Result: 3.14", "3.14") == 1.0

    def test_wrong_number(self):
        assert self.m.score("The answer is 41.", "42") == 0.0

    def test_negative_number(self):
        assert self.m.score("Answer: -7", "-7") == 1.0

    def test_last_number_wins(self):
        assert self.m.score("Step 1: 10, Step 2: 42", "42") == 1.0


class TestF1Metric:
    m = F1Metric()

    def test_perfect_match(self):
        assert self.m.score("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert self.m.score("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = self.m.score("the cat sat", "the cat")
        assert 0.0 < score < 1.0

    def test_empty_both(self):
        assert self.m.score("", "") == pytest.approx(1.0)

    def test_empty_prediction(self):
        assert self.m.score("", "answer") == pytest.approx(0.0)

    def test_symmetric(self):
        a, b = "quick brown fox", "brown fox jumped"
        assert self.m.score(a, b) == pytest.approx(self.m.score(b, a))


class TestSubstringMatchMetric:
    m = SubstringMatchMetric()

    def test_exact(self):
        assert self.m.score("42", "42") == 1.0

    def test_reference_in_prediction(self):
        assert self.m.score("the answer is 42 units", "42") == 1.0

    def test_reference_not_in_prediction(self):
        assert self.m.score("the answer is 41", "42") == 0.0

    def test_case_insensitive(self):
        assert self.m.score("Paris is the capital", "paris") == 1.0


class TestMultipleChoiceMetric:
    m = MultipleChoiceMetric()

    def test_exact_letter(self):
        assert self.m.score("A", "A") == 1.0

    def test_extracts_letter_from_prose(self):
        assert self.m.score("The answer is C.", "C") == 1.0

    def test_extracts_last_option(self):
        assert self.m.score("A is tempting, but the answer is D.", "D") == 1.0

    def test_answer_pattern_beats_later_explanation_letters(self):
        assert self.m.score("The answer is A because B is too broad.", "A") == 1.0

    def test_wrong_letter(self):
        assert self.m.score("The answer is B.", "C") == 0.0


class TestMetricRegistry:
    def test_all_keys_present(self):
        for key in ("exact_match", "numeric_exact_match", "f1", "substring", "multiple_choice"):
            assert key in METRIC_REGISTRY

    def test_registry_instances_work(self):
        for metric in METRIC_REGISTRY.values():
            score = metric.score("42", "42")
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ConstraintChecker
# ---------------------------------------------------------------------------

class TestKeywordConstraintChecker:
    c = KeywordConstraintChecker()

    def test_all_satisfied(self):
        assert self.c.check("use bullet points in your response", ["bullet", "response"]) == 1.0

    def test_none_satisfied(self):
        assert self.c.check("no matching words", ["xyz", "abc"]) == 0.0

    def test_partial(self):
        score = self.c.check("response with bullet points", ["bullet", "xyz"])
        assert score == pytest.approx(0.5)

    def test_empty_constraints(self):
        assert self.c.check("anything", []) == 1.0

    def test_case_insensitive(self):
        assert self.c.check("Answer in JSON format", ["json"]) == 1.0


class TestIFBenchConstraintChecker:
    c = IFBenchConstraintChecker()

    def test_negative_keyword_constraint_rewards_absence(self):
        objs = [{"constraint_type": "Keywords",
                 "constraint": "Do not include the word banana."}]
        assert self.c.check("Apple and pear.", [], {"constraint_objects": objs}) == 1.0

    def test_negative_keyword_constraint_penalizes_presence(self):
        objs = [{"constraint_type": "Keywords",
                 "constraint": "Do not include the word banana."}]
        assert self.c.check("Apple and banana.", [], {"constraint_objects": objs}) == 0.0

    def test_at_most_word_limit_is_inclusive(self):
        objs = [{"constraint_type": "Length",
                 "constraint": "Use at most 3 words."}]
        assert self.c.check("one two three", [], {"constraint_objects": objs}) == 1.0

    def test_at_least_word_limit_is_inclusive(self):
        objs = [{"constraint_type": "Length",
                 "constraint": "Use at least 3 words."}]
        assert self.c.check("one two three", [], {"constraint_objects": objs}) == 1.0

    def test_negative_bullet_format_rewards_plain_text(self):
        objs = [{"constraint_type": "Format",
                 "constraint": "Do not use bullet points."}]
        assert self.c.check("Plain sentence.", [], {"constraint_objects": objs}) == 1.0

    def test_negative_bullet_format_penalizes_bullets(self):
        objs = [{"constraint_type": "Format",
                 "constraint": "Do not use bullet points."}]
        assert self.c.check("- item", [], {"constraint_objects": objs}) == 0.0


# ---------------------------------------------------------------------------
# Perturbators
# ---------------------------------------------------------------------------

class TestWordShufflePerturbator:
    p = WordShufflePerturbator()
    rng = np.random.default_rng(0)

    def test_returns_n_variants(self):
        variants = self.p.perturb("the cat sat on the mat", n=3, rng=self.rng)
        assert len(variants) == 3

    def test_variants_have_same_words(self):
        text = "quick brown fox jumps over lazy dog"
        variants = self.p.perturb(text, n=5, rng=np.random.default_rng(1))
        for v in variants:
            assert sorted(v.split()) == sorted(text.split())

    def test_short_input_returns_original(self):
        rng = np.random.default_rng(0)
        variants = self.p.perturb("hi", n=2, rng=rng)
        assert all(v == "hi" for v in variants)


class TestTruncationPerturbator:
    p = TruncationPerturbator(min_frac=0.5, max_frac=0.8)
    rng = np.random.default_rng(0)

    def test_returns_n_variants(self):
        variants = self.p.perturb("a b c d e f g h", n=3, rng=self.rng)
        assert len(variants) == 3

    def test_variants_shorter_than_original(self):
        text = " ".join(str(i) for i in range(20))
        variants = self.p.perturb(text, n=5, rng=np.random.default_rng(2))
        for v in variants:
            assert len(v.split()) < 20

    def test_variants_at_least_half_original(self):
        text = " ".join(str(i) for i in range(20))
        variants = self.p.perturb(text, n=5, rng=np.random.default_rng(3))
        for v in variants:
            assert len(v.split()) >= 10


class TestCompositePerturbator:
    def test_round_robin(self):
        rng = np.random.default_rng(0)
        p = CompositePerturbator([
            TruncationPerturbator(0.5, 0.6),
            WordShufflePerturbator(),
        ])
        text = "the quick brown fox jumps over the lazy dog"
        variants = p.perturb(text, n=4, rng=rng)
        assert len(variants) == 4


# ---------------------------------------------------------------------------
# PerturbationBuffer
# ---------------------------------------------------------------------------

class TestPerturbationBuffer:
    def test_returns_m_variants(self):
        buf = PerturbationBuffer(m=3)
        variants = buf.get("hello world today", m=3)
        assert len(variants) == 3

    def test_cached_on_second_call(self):
        buf = PerturbationBuffer(m=2)
        v1 = buf.get("test input here")
        v2 = buf.get("test input here")
        assert v1 == v2

    def test_extends_cache_if_more_requested(self):
        buf = PerturbationBuffer(m=2)
        buf.get("input text", m=2)
        extended = buf.get("input text", m=4)
        assert len(extended) == 4

    def test_evicts_when_full(self):
        buf = PerturbationBuffer(m=1, max_size=2)
        buf.get("input one")
        buf.get("input two")
        buf.get("input three")   # should evict oldest
        assert buf.size <= 2

    def test_clear_empties_cache(self):
        buf = PerturbationBuffer(m=2)
        buf.get("some input text")
        buf.clear()
        assert buf.size == 0

    def test_different_inputs_different_variants(self):
        buf = PerturbationBuffer(m=2)
        v1 = buf.get("what is the capital of france")
        v2 = buf.get("solve the quadratic equation x squared")
        assert v1 != v2


# ---------------------------------------------------------------------------
# RewardComponents
# ---------------------------------------------------------------------------

class TestRewardComponents:
    def test_as_dict_has_all_keys(self):
        rc = RewardComponents(task=0.8, constraint=0.5, cost=-0.01, variance=-0.05,
                              total=0.74)
        d = rc.as_dict()
        for key in ("r_task", "r_constraint", "r_cost", "r_variance", "r_total"):
            assert key in d

    def test_total_matches_components(self):
        cfg = RewardConfig(lambda_constraint=0.2)
        task, constraint, cost, var = 0.8, 1.0, -0.01, -0.02
        total = task + cfg.lambda_constraint * constraint + cost + var
        rc = RewardComponents(task=task, constraint=constraint, cost=cost,
                              variance=var, total=total)
        assert rc.total == pytest.approx(total)


# ---------------------------------------------------------------------------
# RewardComputer
# ---------------------------------------------------------------------------

class TestRewardComputerBasic:
    def _setup(self, lm_response="42", skip_variance=True):
        lm = FixedLM(lm_response)
        g, executor = make_graph_and_executor(lm)
        from ppg.core.executor import PromptAssembler
        assembler = PromptAssembler(g)
        cfg = RewardConfig(
            lambda_cost=0.001,
            lambda_variance=0.1,
            lambda_constraint=0.2,
            max_tokens_ref=100,
            skip_variance=skip_variance,
        )
        reward_fn = RewardComputer(
            task_metric=NumericExactMatchMetric(),
            lm=lm,
            assembler=assembler,
            config=cfg,
        )
        trace = executor.execute("What is 6*7?")
        return reward_fn, trace, lm

    def test_returns_reward_components(self):
        reward_fn, trace, _ = self._setup()
        rc = reward_fn.compute(trace, "What is 6*7?", "42")
        assert isinstance(rc, RewardComponents)

    def test_correct_answer_task_score_one(self):
        reward_fn, trace, _ = self._setup(lm_response="42")
        rc = reward_fn.compute(trace, "q", "42")
        assert rc.task == pytest.approx(1.0)

    def test_wrong_answer_task_score_zero(self):
        reward_fn, trace, _ = self._setup(lm_response="99")
        rc = reward_fn.compute(trace, "q", "42")
        assert rc.task == pytest.approx(0.0)

    def test_cost_is_negative(self):
        reward_fn, trace, _ = self._setup()
        rc = reward_fn.compute(trace, "q", "42")
        assert rc.cost < 0.0

    def test_cost_scales_with_tokens(self):
        reward_fn, trace, _ = self._setup()
        short_trace = trace
        # Fake a long prompt by inflating token_count
        import dataclasses
        long_trace = dataclasses.replace(trace, token_count=trace.token_count * 10)
        rc_short = reward_fn.compute(short_trace, "q", "42")
        rc_long  = reward_fn.compute(long_trace,  "q", "42")
        assert rc_long.cost < rc_short.cost

    def test_no_constraint_checker_gives_zero_constraint(self):
        reward_fn, trace, _ = self._setup()
        rc = reward_fn.compute(trace, "q", "42", constraints=["bullet"])
        assert rc.constraint == pytest.approx(0.0)

    def test_skip_variance_gives_zero_variance(self):
        reward_fn, trace, _ = self._setup(skip_variance=True)
        rc = reward_fn.compute(trace, "q", "42")
        assert rc.variance == pytest.approx(0.0)

    def test_total_equals_sum_of_components(self):
        reward_fn, trace, _ = self._setup()
        rc = reward_fn.compute(trace, "q", "42")
        expected_total = rc.task + 0.2 * rc.constraint + rc.cost + rc.variance
        assert rc.total == pytest.approx(expected_total, rel=1e-6)


class TestRewardComputerWithConstraint:
    def _make(self, lm_response):
        lm = FixedLM(lm_response)
        g, executor = make_graph_and_executor(lm)
        from ppg.core.executor import PromptAssembler
        assembler = PromptAssembler(g)
        cfg = RewardConfig(lambda_constraint=0.5, skip_variance=True)
        reward_fn = RewardComputer(
            task_metric=ExactMatchMetric(),
            lm=lm,
            assembler=assembler,
            constraint_checker=KeywordConstraintChecker(),
            config=cfg,
        )
        trace = executor.execute("q")
        return reward_fn, trace

    def test_satisfied_constraints_positive(self):
        reward_fn, trace = self._make("use bullet points here")
        rc = reward_fn.compute(trace, "q", "ref", constraints=["bullet"])
        assert rc.constraint == pytest.approx(1.0)

    def test_unsatisfied_constraints_zero(self):
        reward_fn, trace = self._make("no keywords here")
        rc = reward_fn.compute(trace, "q", "ref", constraints=["xyz"])
        assert rc.constraint == pytest.approx(0.0)

    def test_constraint_contribution_in_total(self):
        reward_fn, trace = self._make("use bullet points here")
        rc_satisfied   = reward_fn.compute(trace, "q", "ref", constraints=["bullet"])
        rc_unsatisfied = reward_fn.compute(trace, "q", "ref", constraints=["xyz_absent"])
        # satisfied constraint should produce higher total
        assert rc_satisfied.total > rc_unsatisfied.total


class TestRewardComputerVariance:
    def test_variance_nonzero_when_lm_alternates(self):
        """Alternating LM produces non-zero variance across perturbations."""
        lm = AlternatingLM(a="42", b="99")
        g, executor = make_graph_and_executor(lm)
        from ppg.core.executor import PromptAssembler
        assembler = PromptAssembler(g)
        cfg = RewardConfig(
            lambda_variance=1.0,
            m_perturbation=2,
            skip_variance=False,
        )
        reward_fn = RewardComputer(
            task_metric=ExactMatchMetric(),
            lm=lm,
            assembler=assembler,
            config=cfg,
        )
        trace = executor.execute("q")
        rc = reward_fn.compute(trace, "q input text here", "42")
        assert rc.variance < 0.0   # negative penalty

    def test_variance_zero_when_lm_consistent(self):
        """Consistent LM: all perturbations give same score -> variance=0."""
        lm = FixedLM("42")
        g, executor = make_graph_and_executor(lm)
        from ppg.core.executor import PromptAssembler
        assembler = PromptAssembler(g)
        cfg = RewardConfig(lambda_variance=1.0, m_perturbation=3, skip_variance=False)
        reward_fn = RewardComputer(
            task_metric=ExactMatchMetric(),
            lm=lm,
            assembler=assembler,
            config=cfg,
        )
        trace = executor.execute("q")
        rc = reward_fn.compute(trace, "q the input text", "42")
        assert rc.variance == pytest.approx(0.0, abs=1e-9)

    def test_variance_calls_lm_m_times(self):
        lm = FixedLM("42")
        g, executor = make_graph_and_executor(lm)
        from ppg.core.executor import PromptAssembler
        assembler = PromptAssembler(g)
        cfg = RewardConfig(m_perturbation=3, skip_variance=False)
        reward_fn = RewardComputer(
            task_metric=ExactMatchMetric(),
            lm=lm,
            assembler=assembler,
            config=cfg,
        )
        trace = executor.execute("q")
        lm.call_count = 0          # reset after executor call
        reward_fn.compute(trace, "variance test input", "42")
        assert lm.call_count == 3  # m_perturbation=3


# ---------------------------------------------------------------------------
# Integration: full episode -> reward -> update
# ---------------------------------------------------------------------------

class TestEndToEndEpisode:
    def test_execute_compute_update(self):
        """Simulate one full training step without assertions on values."""
        from ppg.bandits import LinUCBPolicy
        from ppg.core.executor import PromptAssembler

        lm = FixedLM("42")
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING,    "Solve: {input}")
        b.add_fragment(FragmentType.REASONING_STYLE, "Think.")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
        ids = b.node_ids()
        b.connect_chain(*ids)
        g = b.build()

        policy   = LinUCBPolicy(g, alpha=0.5)
        executor = PPGExecutor(g, policy, lm, FeatureExtractor(), ExecutorConfig())
        assembler = PromptAssembler(g)
        reward_fn = RewardComputer(
            task_metric=NumericExactMatchMetric(),
            lm=lm,
            assembler=assembler,
            config=RewardConfig(skip_variance=True),
        )

        trace = executor.execute("What is 6*7?")
        rc    = reward_fn.compute(trace, "What is 6*7?", "42")
        phi   = trace.pre_lm_features.as_vector()

        policy.update_path(trace.edges_traversed, phi, rc.total)

        assert rc.task == pytest.approx(1.0)
        assert rc.total > 0.0
        assert policy.total_updates == len(trace.edges_traversed)
