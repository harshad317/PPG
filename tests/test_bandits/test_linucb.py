"""Tests for ppg/bandits/linucb.py."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ppg.bandits import LinUCBArm, LinUCBPolicy
from ppg.core import (
    FEATURE_DIM,
    FragmentType,
    PPGraphBuilder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_graph():
    """task_framing -> reasoning_style -> output_contract."""
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    b.connect_chain(*ids)
    return b.build(), ids


def make_branching_graph():
    """
    task_framing -> reasoning_style -> output_contract
                 -> compression    -> output_contract
    """
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think.")
    b.add_fragment(FragmentType.COMPRESSION,     "Short.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    tf, rs, comp, oc = ids
    b.connect(tf, rs)
    b.connect(tf, comp)
    b.connect(rs, oc)
    b.connect(comp, oc)
    return b.build(), ids


def rand_phi(seed=0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(FEATURE_DIM)


# ---------------------------------------------------------------------------
# LinUCBArm
# ---------------------------------------------------------------------------

class TestLinUCBArm:
    def test_init_A_is_identity_scaled(self):
        arm = LinUCBArm(feature_dim=4, lambda_reg=2.0)
        expected = 2.0 * np.eye(4)
        assert np.allclose(arm.A, expected)

    def test_init_b_is_zero(self):
        arm = LinUCBArm(feature_dim=4)
        assert np.allclose(arm.b, 0.0)

    def test_n_updates_starts_zero(self):
        arm = LinUCBArm(FEATURE_DIM)
        assert arm.n_updates == 0

    def test_score_train_mode_greater_than_greedy(self):
        """UCB bonus should make train score >= greedy score."""
        arm = LinUCBArm(FEATURE_DIM, lambda_reg=1.0)
        phi = rand_phi()
        train_score = arm.score(phi, alpha=1.0, train_mode=True)
        greedy_score = arm.score(phi, alpha=1.0, train_mode=False)
        assert train_score >= greedy_score

    def test_score_alpha_zero_equals_greedy(self):
        arm = LinUCBArm(FEATURE_DIM)
        phi = rand_phi()
        assert arm.score(phi, alpha=0.0, train_mode=True) == pytest.approx(
            arm.score(phi, alpha=0.0, train_mode=False)
        )

    def test_score_uninit_arm_is_zero(self):
        """Uninitialised arm: b=0, A=I => mean=0, score=alpha*||phi||."""
        arm = LinUCBArm(FEATURE_DIM, lambda_reg=1.0)
        phi = np.ones(FEATURE_DIM)
        greedy = arm.score(phi, alpha=1.0, train_mode=False)
        assert greedy == pytest.approx(0.0)

    def test_update_increments_n_updates(self):
        arm = LinUCBArm(FEATURE_DIM)
        phi = rand_phi()
        arm.update(phi, reward=1.0)
        arm.update(phi, reward=0.5)
        assert arm.n_updates == 2

    def test_update_changes_A_and_b(self):
        arm = LinUCBArm(FEATURE_DIM)
        A_before = arm.A.copy()
        b_before = arm.b.copy()
        phi = rand_phi()
        arm.update(phi, reward=1.0)
        assert not np.allclose(arm.A, A_before)
        assert not np.allclose(arm.b, b_before)

    def test_A_remains_symmetric_after_updates(self):
        arm = LinUCBArm(FEATURE_DIM)
        rng = np.random.default_rng(7)
        for _ in range(10):
            phi = rng.standard_normal(FEATURE_DIM)
            arm.update(phi, reward=rng.uniform())
        assert np.allclose(arm.A, arm.A.T)

    def test_A_remains_positive_definite(self):
        """All eigenvalues should stay > 0."""
        arm = LinUCBArm(FEATURE_DIM, lambda_reg=0.1)
        rng = np.random.default_rng(3)
        for _ in range(20):
            arm.update(rng.standard_normal(FEATURE_DIM), reward=rng.uniform(-1, 1))
        eigvals = np.linalg.eigvalsh(arm.A)
        assert np.all(eigvals > 0)

    def test_mu_hat_shape(self):
        arm = LinUCBArm(FEATURE_DIM)
        assert arm.mu_hat.shape == (FEATURE_DIM,)

    def test_arm_learns_reward_direction(self):
        """
        After many updates with phi=e1 and reward=1, arm should assign
        high score to e1 and near-zero to e2.
        """
        arm = LinUCBArm(feature_dim=2, lambda_reg=0.01)
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        for _ in range(200):
            arm.update(e1, reward=1.0)
            arm.update(e2, reward=0.0)
        score_e1 = arm.score(e1, alpha=0.0, train_mode=False)
        score_e2 = arm.score(e2, alpha=0.0, train_mode=False)
        assert score_e1 > score_e2

    def test_serialization_roundtrip(self):
        arm = LinUCBArm(FEATURE_DIM, lambda_reg=0.5)
        rng = np.random.default_rng(0)
        for _ in range(5):
            arm.update(rng.standard_normal(FEATURE_DIM), reward=rng.uniform())
        arrays = arm.to_arrays()
        arm2 = LinUCBArm.from_arrays(arrays)
        assert np.allclose(arm2.A, arm.A)
        assert np.allclose(arm2.b, arm.b)
        assert arm2.n_updates == arm.n_updates
        assert arm2.feature_dim == arm.feature_dim
        assert arm2.lambda_reg == arm.lambda_reg


# ---------------------------------------------------------------------------
# LinUCBPolicy
# ---------------------------------------------------------------------------

class TestLinUCBPolicyInit:
    def test_pre_creates_arms_for_all_edges(self):
        g, _ = make_graph()
        policy = LinUCBPolicy(g)
        for edge in g.edges:
            assert edge in policy._arms

    def test_arm_count_matches_edge_count(self):
        g, _ = make_branching_graph()
        policy = LinUCBPolicy(g)
        assert len(policy._arms) == len(g.edges)

    def test_total_updates_zero_at_init(self):
        g, _ = make_graph()
        policy = LinUCBPolicy(g)
        assert policy.total_updates == 0


class TestLinUCBPolicySelect:
    def test_single_candidate_returned_immediately(self):
        g, ids = make_graph()
        policy = LinUCBPolicy(g)
        phi = rand_phi()
        result = policy.select(ids[0], [ids[1]], phi)
        assert result == ids[1]

    def test_returns_valid_candidate(self):
        g, ids = make_branching_graph()
        tf, rs, comp, oc = ids
        policy = LinUCBPolicy(g)
        phi = rand_phi()
        result = policy.select(tf, [rs, comp], phi)
        assert result in (rs, comp)

    def test_none_current_uses_start_sentinel(self):
        """select(None, ...) should not raise and should return a valid candidate."""
        g, ids = make_graph()
        policy = LinUCBPolicy(g)
        phi = rand_phi()
        result = policy.select(None, [ids[0]], phi)
        assert result == ids[0]

    def test_greedy_mode_no_exploration(self):
        """
        In eval mode (train_mode=False) with alpha>0, the score has no
        UCB bonus. Both calls should return the same arm (deterministic).
        """
        g, ids = make_branching_graph()
        tf, rs, comp, oc = ids
        policy = LinUCBPolicy(g, alpha=5.0)
        phi = rand_phi(seed=42)
        # eval mode is deterministic given same phi
        r1 = policy.select(tf, [rs, comp], phi, train_mode=False)
        r2 = policy.select(tf, [rs, comp], phi, train_mode=False)
        assert r1 == r2

    def test_learned_preference_respected(self):
        """
        After many updates rewarding rs and punishing comp, policy should
        prefer rs in greedy mode.
        """
        g, ids = make_branching_graph()
        tf, rs, comp, oc = ids
        policy = LinUCBPolicy(g, alpha=0.0, lambda_reg=0.01)
        # Fixed phi so result is deterministic regardless of FEATURE_DIM.
        phi = np.ones(FEATURE_DIM)

        for _ in range(300):
            policy.update((tf, rs),   phi, reward=1.0)
            policy.update((tf, comp), phi, reward=0.0)

        chosen = policy.select(tf, [rs, comp], phi, train_mode=False)
        assert chosen == rs


class TestLinUCBPolicyUpdate:
    def test_update_increments_arm(self):
        g, ids = make_graph()
        tf, rs, oc = ids
        policy = LinUCBPolicy(g)
        phi = rand_phi()
        policy.update((tf, rs), phi, reward=1.0)
        assert policy._arms[(tf, rs)].n_updates == 1

    def test_update_total_updates(self):
        g, ids = make_graph()
        tf, rs, oc = ids
        policy = LinUCBPolicy(g)
        phi = rand_phi()
        policy.update((tf, rs), phi, 1.0)
        policy.update((rs, oc), phi, 0.5)
        assert policy.total_updates == 2

    def test_update_unknown_edge_creates_arm(self):
        """Unknown edges created lazily (e.g. escalation edge)."""
        g, _ = make_graph()
        policy = LinUCBPolicy(g)
        phi = rand_phi()
        policy.update(("unknown_src", "unknown_dst"), phi, 0.7)
        assert ("unknown_src", "unknown_dst") in policy._arms

    def test_update_path_updates_all_edges(self):
        g, ids = make_graph()
        tf, rs, oc = ids
        policy = LinUCBPolicy(g)
        phi = rand_phi()
        policy.update_path([(tf, rs), (rs, oc)], phi, reward=0.9)
        assert policy._arms[(tf, rs)].n_updates == 1
        assert policy._arms[(rs, oc)].n_updates == 1

    def test_only_traversed_edge_updated(self):
        g, ids = make_branching_graph()
        tf, rs, comp, oc = ids
        policy = LinUCBPolicy(g)
        phi = rand_phi()
        policy.update((tf, rs), phi, 1.0)
        # comp arm should be untouched
        assert policy._arms[(tf, comp)].n_updates == 0


class TestLinUCBPolicySaveLoad:
    def test_save_load_roundtrip(self):
        g, ids = make_graph()
        tf, rs, oc = ids
        policy = LinUCBPolicy(g)
        rng = np.random.default_rng(1)
        for _ in range(10):
            phi = rng.standard_normal(FEATURE_DIM)
            policy.update((tf, rs), phi, rng.uniform())

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        policy.save(path)
        policy2 = LinUCBPolicy(g)
        policy2.load(path)

        assert np.allclose(policy._arms[(tf, rs)].A, policy2._arms[(tf, rs)].A)
        assert np.allclose(policy._arms[(tf, rs)].b, policy2._arms[(tf, rs)].b)
        assert policy._arms[(tf, rs)].n_updates == policy2._arms[(tf, rs)].n_updates

    def test_save_load_preserves_scores(self):
        g, ids = make_graph()
        tf, rs, oc = ids
        policy = LinUCBPolicy(g)
        rng = np.random.default_rng(2)
        phi = rng.standard_normal(FEATURE_DIM)
        for _ in range(20):
            policy.update((tf, rs), rng.standard_normal(FEATURE_DIM), rng.uniform())

        score_before = policy._arms[(tf, rs)].score(phi, 1.0, False)

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "policy.npz"
            policy.save(str(p))
            policy2 = LinUCBPolicy(g)
            policy2.load(str(p))

        score_after = policy2._arms[(tf, rs)].score(phi, 1.0, False)
        assert score_before == pytest.approx(score_after, rel=1e-6)


class TestLinUCBPolicyDiagnostics:
    def test_arm_stats_keys_readable(self):
        g, _ = make_graph()
        policy = LinUCBPolicy(g)
        stats = policy.arm_stats()
        assert len(stats) == len(g.edges)
        for key in stats:
            assert "->" in key

    def test_arm_stats_n_updates_after_training(self):
        g, ids = make_graph()
        tf, rs, oc = ids
        policy = LinUCBPolicy(g)
        phi = rand_phi()
        for _ in range(5):
            policy.update((tf, rs), phi, 1.0)
        stats = policy.arm_stats()
        tf_type = g.nodes[tf].type.value
        rs_type = g.nodes[rs].type.value
        key = f"{tf_type}->{rs_type}"
        assert stats[key]["n_updates"] == 5

    def test_guard_weights_shape(self):
        g, ids = make_graph()
        tf, rs, oc = ids
        policy = LinUCBPolicy(g)
        w = policy.guard_weights_for_edge(tf, rs)
        assert w.shape == (FEATURE_DIM,)


# ---------------------------------------------------------------------------
# Integration: LinUCBPolicy as executor's NodeSelector
# ---------------------------------------------------------------------------

class TestLinUCBWithExecutor:
    def test_linucb_satisfies_node_selector_protocol(self):
        from ppg.core.executor import NodeSelector
        g, _ = make_graph()
        policy = LinUCBPolicy(g)
        assert isinstance(policy, NodeSelector)

    def test_linucb_drives_executor(self):
        from ppg.core import (
            ExecutorConfig, FeatureExtractor, PPGExecutor,
        )
        from ppg.core.executor import LMClient

        class FixedLM:
            def complete(self, prompt: str) -> str:
                return "42"

        g, ids = make_graph()
        policy = LinUCBPolicy(g, alpha=0.5)
        executor = PPGExecutor(
            graph=g,
            selector=policy,
            lm=FixedLM(),
            feature_extractor=FeatureExtractor(),
            config=ExecutorConfig(),
        )
        trace = executor.execute("What is 2+2?")
        assert trace.lm_response == "42"
        assert trace.node_ids == ids

    def test_update_after_execute(self):
        """Simulate one training step: execute -> compute reward -> update policy."""
        from ppg.core import ExecutorConfig, FeatureExtractor, PPGExecutor

        class FixedLM:
            def complete(self, prompt: str) -> str:
                return "42"

        g, ids = make_graph()
        tf, rs, oc = ids
        policy = LinUCBPolicy(g, alpha=0.5)
        executor = PPGExecutor(g, policy, FixedLM(), FeatureExtractor(),
                               ExecutorConfig())

        trace = executor.execute("q")
        phi = trace.pre_lm_features.as_vector()
        reward = 1.0

        policy.update_path(trace.edges_traversed, phi, reward)
        assert policy.total_updates == len(trace.edges_traversed)
