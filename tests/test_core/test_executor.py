"""Tests for ppg/core/executor.py."""

import numpy as np
import pytest

from ppg.core import (
    FragmentType,
    Guard,
    PPGraphBuilder,
    FeatureExtractor,
    ExecutorConfig,
    PPGExecutor,
    PathTrace,
    RandomSelector,
    HighestUtilitySelector,
    PromptAssembler,
    FEATURE_DIM,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class EchoLM:
    """Returns prompt verbatim. No API calls."""
    def complete(self, prompt: str) -> str:
        return f"RESPONSE:{prompt[:20]}"


class CountingLM:
    """Counts calls and returns canned responses."""
    def __init__(self, responses: list[str]):
        self._responses = responses
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        resp = self._responses[self.call_count % len(self._responses)]
        self.call_count += 1
        return resp


class SamplingLM:
    """Returns canned sample batches, then optional complete responses."""
    def __init__(self, samples: list[str], complete_response: str = "final"):
        self.samples = samples
        self.complete_response = complete_response
        self.sample_calls = 0
        self.complete_calls = 0

    def complete(self, prompt: str) -> str:
        self.complete_calls += 1
        return self.complete_response

    def sample(self, prompt: str, n: int) -> list[str]:
        self.sample_calls += 1
        return self.samples[:n]


def make_linear_graph():
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Solve: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think step by step.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    b.connect_chain(*ids)
    return b.build(), ids


def make_branching_graph():
    """
    task_framing --(all-pass)--> reasoning_style --> output_contract
                 --(blocked)---> compression     --> output_contract
    """
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Solve: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think step by step.")
    b.add_fragment(FragmentType.COMPRESSION,     "Be concise.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    tf, rs, comp, oc = ids
    b.connect(tf, rs)                                        # all-pass
    b.connect(tf, comp, guard=Guard(                         # blocked (bias=1e9)
        weights=np.zeros(FEATURE_DIM), bias=1e9))
    b.connect(rs, oc)
    b.connect(comp, oc)
    return b.build(), ids


def make_escalation_graph():
    """
    task_framing --> output_contract
    (uncertainty_escalation node also present for escalation tests)
    """
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,           "Solve: {input}")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT,        "Answer:")
    b.add_fragment(FragmentType.UNCERTAINTY_ESCALATION, "Double-check your answer.")
    ids = b.node_ids()
    tf, oc, esc = ids
    b.connect(tf, oc)
    # esc node is connected so validator accepts it, but executor only adds
    # it during escalation — connect to oc so graph is valid
    b.connect(esc, oc)
    # But that makes oc reachable from two paths and esc needs a predecessor...
    # Simpler: make esc a direct successor of tf too (guard blocked normally)
    return b.build(), ids


def make_executor(graph, lm=None, seed=0, config=None):
    return PPGExecutor(
        graph=graph,
        selector=RandomSelector(seed=seed),
        lm=lm or EchoLM(),
        feature_extractor=FeatureExtractor(),
        config=config,
    )


# ---------------------------------------------------------------------------
# PromptAssembler
# ---------------------------------------------------------------------------

class TestPromptAssembler:
    def test_assembles_linear_path(self):
        g, ids = make_linear_graph()
        asm = PromptAssembler(g)
        prompt = asm.assemble(ids, {"input": "2+2"})
        assert "Solve: 2+2" in prompt
        assert "Think step by step." in prompt
        assert "Answer:" in prompt

    def test_separator_between_fragments(self):
        g, ids = make_linear_graph()
        asm = PromptAssembler(g, separator=" | ")
        prompt = asm.assemble(ids, {"input": "x"})
        assert " | " in prompt

    def test_skips_blank_fragments(self):
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING,    "Solve: {input}")
        b.add_fragment(FragmentType.REASONING_STYLE, "")   # blank
        b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
        ids = b.node_ids()
        b.connect_chain(*ids)
        g = b.build()
        asm = PromptAssembler(g)
        prompt = asm.assemble(ids, {"input": "x"})
        assert "Solve: x" in prompt
        assert "Answer:" in prompt
        # blank fragment should not produce double separator
        assert "\n\n\n\n" not in prompt

    def test_single_node_prompt(self):
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING,   "Solve: {input}")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Go!")
        ids = b.node_ids()
        b.connect(*ids)
        g = b.build()
        asm = PromptAssembler(g)
        prompt = asm.assemble([ids[0]], {"input": "test"})
        assert prompt == "Solve: test"


# ---------------------------------------------------------------------------
# PathTrace
# ---------------------------------------------------------------------------

class TestPathTrace:
    def test_path_length(self):
        g, ids = make_linear_graph()
        trace = make_executor(g).execute("hello")
        assert trace.path_length == len(trace.node_ids)

    def test_node_set(self):
        g, ids = make_linear_graph()
        trace = make_executor(g).execute("hello")
        assert trace.node_set() == frozenset(trace.node_ids)

    def test_with_reward(self):
        g, ids = make_linear_graph()
        trace = make_executor(g).execute("hello")
        assert trace.reward is None
        t2 = trace.with_reward(0.85)
        assert t2.reward == pytest.approx(0.85)
        assert trace.reward is None   # original unchanged


# ---------------------------------------------------------------------------
# PPGExecutor — basic execution
# ---------------------------------------------------------------------------

class TestPPGExecutorBasic:
    def test_returns_path_trace(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("2+2")
        assert isinstance(trace, PathTrace)

    def test_linear_graph_visits_all_nodes(self):
        g, ids = make_linear_graph()
        trace = make_executor(g).execute("2+2")
        assert trace.node_ids == ids

    def test_linear_graph_one_lm_call(self):
        g, _ = make_linear_graph()
        lm = CountingLM(["answer"])
        make_executor(g, lm=lm).execute("q")
        assert lm.call_count == 1

    def test_prompt_contains_rendered_fragments(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("2+2")
        assert "Solve: 2+2" in trace.assembled_prompt
        assert "Think step by step." in trace.assembled_prompt
        assert "Answer:" in trace.assembled_prompt

    def test_response_from_lm(self):
        g, _ = make_linear_graph()
        lm = CountingLM(["42"])
        trace = make_executor(g, lm=lm).execute("q")
        assert trace.lm_response == "42"

    def test_reward_none_before_trainer(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("q")
        assert trace.reward is None

    def test_token_count_positive(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("q")
        assert trace.token_count > 0

    def test_default_context_uses_input(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("my_question")
        assert "my_question" in trace.assembled_prompt

    def test_custom_context_overrides(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("ignored", context={"input": "custom"})
        assert "custom" in trace.assembled_prompt

    def test_pre_lm_features_populated(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("hello world")
        assert trace.pre_lm_features is not None
        assert trace.pre_lm_features.input_length_norm > 0.0

    def test_post_lm_features_none_in_fast_mode(self):
        g, _ = make_linear_graph()
        trace = make_executor(g, config=ExecutorConfig(escalation_enabled=False)).execute("q")
        assert trace.post_lm_features is None

    def test_escalated_false_in_fast_mode(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("q")
        assert trace.escalated is False


# ---------------------------------------------------------------------------
# Guard decisions
# ---------------------------------------------------------------------------

class TestGuardDecisions:
    def test_guard_decisions_recorded(self):
        g, ids = make_linear_graph()
        trace = make_executor(g).execute("q")
        # linear graph: 2 edges -> 2 guard decisions
        assert len(trace.guard_decisions) == 2

    def test_all_pass_guards_fire(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("q")
        assert all(gd.fired for gd in trace.guard_decisions)

    def test_blocked_guard_not_in_path(self):
        g, ids = make_branching_graph()
        tf, rs, comp, oc = ids
        trace = make_executor(g, seed=0).execute("q")
        # compression path is blocked -> should not appear in path
        assert comp not in trace.node_ids

    def test_blocked_guard_recorded_as_not_fired(self):
        g, ids = make_branching_graph()
        tf, rs, comp, oc = ids
        trace = make_executor(g).execute("q")
        comp_decisions = [gd for gd in trace.guard_decisions
                          if gd.src == tf and gd.dst == comp]
        assert len(comp_decisions) == 1
        assert comp_decisions[0].fired is False

    def test_guard_phi_shape(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("q")
        for gd in trace.guard_decisions:
            assert gd.phi.shape == (FEATURE_DIM,)


# ---------------------------------------------------------------------------
# Edge traversal
# ---------------------------------------------------------------------------

class TestEdgeTraversal:
    def test_edges_traversed_match_path(self):
        g, ids = make_linear_graph()
        trace = make_executor(g).execute("q")
        expected_edges = list(zip(ids[:-1], ids[1:]))
        assert trace.edges_traversed == expected_edges

    def test_edges_traversed_one_fewer_than_nodes(self):
        g, _ = make_linear_graph()
        trace = make_executor(g).execute("q")
        assert len(trace.edges_traversed) == len(trace.node_ids) - 1

    def test_branching_takes_only_active_branch(self):
        g, ids = make_branching_graph()
        tf, rs, comp, oc = ids
        trace = make_executor(g).execute("q")
        # Only tf->rs edge (all-pass); comp is blocked
        assert (tf, comp) not in trace.edges_traversed
        assert (tf, rs) in trace.edges_traversed


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

class TestEscalation:
    def _make_disagreeing_lm(self):
        """Returns a different answer each call -> max disagreement."""
        return CountingLM(["answer_A", "answer_B", "answer_C", "answer_D"])

    def test_no_escalation_when_agreement(self):
        """LM always returns same answer -> sc_disagreement=0 -> no escalation."""
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING,           "Solve: {input}")
        b.add_fragment(FragmentType.UNCERTAINTY_ESCALATION, "Double-check.")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT,        "Answer:")
        ids = b.node_ids()
        tf, esc, oc = ids
        b.connect(tf, esc)
        b.connect(esc, oc)
        g = b.build()

        lm = CountingLM(["42"])   # always same
        cfg = ExecutorConfig(escalation_enabled=True, k_samples=3,
                             escalation_threshold=0.3)
        executor = PPGExecutor(g, RandomSelector(0), lm,
                               FeatureExtractor(), cfg)
        trace = executor.execute("q")
        assert trace.escalated is False

    def test_escalation_fires_on_high_disagreement(self):
        """
        Build minimal graph where UNCERTAINTY_ESCALATION is reachable.
        LM returns all-different answers -> sc_disagreement=1.0 -> escalate.
        """
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING,           "Solve: {input}")
        b.add_fragment(FragmentType.UNCERTAINTY_ESCALATION, "Double-check.")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT,        "Answer:")
        ids = b.node_ids()
        tf, esc, oc = ids
        # Primary path: tf -> oc (escaping esc)
        b.connect(tf, oc)
        # Esc is reachable from tf too (for graph validity)
        b.connect(tf, esc)
        b.connect(esc, oc)
        g = b.build()

        lm = self._make_disagreeing_lm()
        cfg = ExecutorConfig(
            escalation_enabled=True,
            k_samples=4,
            escalation_threshold=0.3,
        )
        # Force path tf->oc (primary) by blocking tf->esc guard
        g.edges[(tf, esc)] = Guard(weights=np.zeros(FEATURE_DIM), bias=1e9)

        executor = PPGExecutor(g, RandomSelector(0), lm, FeatureExtractor(), cfg)
        trace = executor.execute("q")

        assert trace.post_lm_features is not None
        assert trace.post_lm_features.sc_disagreement > 0.3
        assert trace.escalated is True

    def test_escalation_adds_extra_lm_call(self):
        """Escalation requires at least k_samples + 1 total LM calls."""
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING,           "Solve: {input}")
        b.add_fragment(FragmentType.UNCERTAINTY_ESCALATION, "Double-check.")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT,        "Answer:")
        ids = b.node_ids()
        tf, esc, oc = ids
        b.connect(tf, oc)
        b.connect(tf, esc)
        b.connect(esc, oc)
        g = b.build()
        g.edges[(tf, esc)] = Guard(weights=np.zeros(FEATURE_DIM), bias=1e9)

        lm = self._make_disagreeing_lm()
        cfg = ExecutorConfig(escalation_enabled=True, k_samples=3,
                             escalation_threshold=0.0)  # always escalate
        executor = PPGExecutor(g, RandomSelector(0), lm, FeatureExtractor(), cfg)
        executor.execute("q")
        # k_samples responses + 1 escalation call
        assert lm.call_count >= cfg.k_samples + 1

    def test_majority_aggregation_returns_modal_sample(self):
        """Self-consistency can return the modal normalized answer."""
        g, _ = make_linear_graph()
        lm = SamplingLM(["#### 7", "#### 42", "The answer is 42."])
        cfg = ExecutorConfig(
            escalation_enabled=True,
            k_samples=3,
            escalation_threshold=1.0,
            sample_aggregation="majority",
        )
        executor = PPGExecutor(g, RandomSelector(0), lm, FeatureExtractor(), cfg)

        trace = executor.execute("q")

        assert trace.lm_response == "#### 42"
        assert trace.escalated is False
        assert lm.sample_calls == 1
        assert lm.complete_calls == 0

    def test_builtin_escalation_template_works_without_graph_node(self):
        """Production fallback escalation does not require a special graph node."""
        g, _ = make_linear_graph()
        lm = SamplingLM(["#### 7", "#### 41", "#### 42"], complete_response="#### 42")
        cfg = ExecutorConfig(
            escalation_enabled=True,
            k_samples=3,
            escalation_threshold=0.0,
            sample_aggregation="majority",
            escalation_template="Candidates:\n{candidate_answers}\nChoose final.",
        )
        executor = PPGExecutor(g, RandomSelector(0), lm, FeatureExtractor(), cfg)

        trace = executor.execute("q")

        assert trace.escalated is True
        assert trace.lm_response == "#### 42"
        assert "Candidates:" in trace.assembled_prompt
        assert "1. #### 7" in trace.assembled_prompt

    def test_no_escalation_node_no_crash(self):
        """If graph has no UNCERTAINTY_ESCALATION node, escalation is skipped."""
        g, _ = make_linear_graph()
        lm = self._make_disagreeing_lm()
        cfg = ExecutorConfig(escalation_enabled=True, k_samples=3,
                             escalation_threshold=0.0)
        executor = PPGExecutor(g, RandomSelector(0), lm, FeatureExtractor(), cfg)
        trace = executor.execute("q")
        assert trace.escalated is False


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

class TestSelectors:
    def test_random_selector_valid_choice(self):
        g, _ = make_linear_graph()
        from ppg.core.executor import RandomSelector
        sel = RandomSelector(seed=42)
        candidates = ["a", "b", "c"]
        phi = np.zeros(FEATURE_DIM)
        result = sel.select(None, candidates, phi)
        assert result in candidates

    def test_random_selector_update_no_error(self):
        sel = RandomSelector()
        sel.update(("a", "b"), np.zeros(FEATURE_DIM), 1.0)  # should not raise

    def test_highest_utility_selector_picks_best(self):
        g, ids = make_linear_graph()
        tf, rs, oc = ids
        # Set utility scores
        g.nodes[rs].utility = 0.9
        g.nodes[oc].utility = 0.1
        sel = HighestUtilitySelector(g)
        phi = np.zeros(FEATURE_DIM)
        chosen = sel.select(tf, [rs, oc], phi)
        assert chosen == rs

    def test_highest_utility_selector_update_no_error(self):
        g, _ = make_linear_graph()
        sel = HighestUtilitySelector(g)
        sel.update(("a", "b"), np.zeros(FEATURE_DIM), 1.0)  # should not raise


# ---------------------------------------------------------------------------
# Max path length safety cap
# ---------------------------------------------------------------------------

class TestMaxPathLength:
    def test_max_path_length_respected(self):
        """Even with all-pass guards, path should not exceed max_path_length."""
        # Build a long chain: tf -> rs -> uc -> ver -> comp -> oc
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING,           "Task: {input}")
        b.add_fragment(FragmentType.REASONING_STYLE,        "Reason.")
        b.add_fragment(FragmentType.UNCERTAINTY_ESCALATION, "Check.")
        b.add_fragment(FragmentType.VERIFICATION,           "Verify.")
        b.add_fragment(FragmentType.COMPRESSION,            "Compress.")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT,        "Answer:")
        ids = b.node_ids()
        b.connect_chain(*ids)
        g = b.build()

        cfg = ExecutorConfig(max_path_length=3)
        trace = make_executor(g, config=cfg).execute("q")
        assert trace.path_length <= cfg.max_path_length + 1  # +1 for the start node
