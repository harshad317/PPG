"""
Tests for the API cost-reduction features:

  * BatchLMClient / DiskCached / Memoizing complete_batch + dedup
  * Anthropic prompt-cache system block
  * GRPO adaptive-k + fine-tune disable (call-count reduction)
  * Mixed-model aux_lm routing for credit + variance
  * Adaptive (early-exit) self-consistency
  * Racing path-calibration survivors

These assert *call-count* / routing behaviour, which is what drives cost. They
do not require any real API access.
"""

from __future__ import annotations

import numpy as np

from ppg.core.executor import ExecutorConfig, PPGExecutor, PromptAssembler
from ppg.core.features import FeatureExtractor, default_normalizer
from ppg.lm.clients import (
    BatchLMClient,
    CountingLMClient,
    DiskCachedLMClient,
    MemoizingLMClient,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class CountingFakeLM:
    """Deterministic echo LM that counts complete() calls."""

    def __init__(self, response="answer A"):
        self.calls = 0
        self.batch_calls = 0
        self._response = response

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._response

    def sample(self, prompt: str, n: int):
        self.calls += n
        return [self._response] * n


class CyclingLM:
    """Returns responses from a fixed list, cycling — for adaptive SC tests."""

    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def complete(self, prompt: str) -> str:
        r = self._responses[self.calls % len(self._responses)]
        self.calls += 1
        return r


# ---------------------------------------------------------------------------
# Batch / dedup
# ---------------------------------------------------------------------------

def test_batch_client_threaded_fallback_one_call_per_prompt():
    fake = CountingFakeLM()
    batch = BatchLMClient(fake, max_workers=4)
    out = batch.complete_batch(["a", "b", "c"])
    assert out == ["answer A"] * 3
    assert fake.calls == 3


def test_disk_cache_complete_batch_dedupes_and_caches(tmp_path):
    fake = CountingFakeLM()
    cache = DiskCachedLMClient(fake, str(tmp_path / "c.json"))
    # Duplicate prompt within the batch must collapse to one real call.
    out = cache.complete_batch(["x", "x", "y"])
    assert out == ["answer A"] * 3
    assert fake.calls == 2  # "x" once + "y" once
    # Second batch hits cache entirely → no new real calls.
    cache.complete_batch(["x", "y"])
    assert fake.calls == 2


def test_memoizing_client_collapses_repeats():
    fake = CountingFakeLM()
    memo = MemoizingLMClient(fake)
    for _ in range(5):
        memo.complete("same prompt")
    assert fake.calls == 1
    memo.complete("different")
    assert fake.calls == 2


def test_counting_client_counts_batch():
    fake = CountingFakeLM()
    counting = CountingLMClient(fake)
    counting.complete_batch(["a", "b"])
    assert counting.call_count == 2


# ---------------------------------------------------------------------------
# Prompt caching (system block)
# ---------------------------------------------------------------------------

def test_anthropic_system_param_cached_when_enabled():
    from ppg.lm.clients import AnthropicClient, AnthropicConfig

    client = AnthropicClient.__new__(AnthropicClient)  # bypass SDK import
    client.cfg = AnthropicConfig(enable_prompt_cache=True, system_msg="SYS")
    param = client._system_param()
    assert isinstance(param, list)
    assert param[0]["cache_control"] == {"type": "ephemeral"}
    assert param[0]["text"] == "SYS"

    client.cfg = AnthropicConfig(enable_prompt_cache=False, system_msg="SYS")
    assert client._system_param() == "SYS"


# ---------------------------------------------------------------------------
# Adaptive self-consistency
# ---------------------------------------------------------------------------

def _make_executor(lm, cfg):
    from ppg.data.fragments import build_graph

    graph = build_graph("gsm8k", topology="lean")
    fx = FeatureExtractor(normalizer=default_normalizer)
    from ppg.core.executor import HighestUtilitySelector
    return PPGExecutor(graph, HighestUtilitySelector(graph), lm, fx, cfg)


def test_adaptive_sampling_stops_early_on_agreement():
    # All samples agree → should stop after the minimum needed, not draw k=5.
    lm = CountingFakeLM(response="42")
    cfg = ExecutorConfig(
        escalation_enabled=True, k_samples=5,
        adaptive_sampling=True, adaptive_min_samples=1, adaptive_confidence=0.7,
        escalation_threshold=1.1,  # never escalate
    )
    ex = _make_executor(lm, cfg)
    ex.execute("question", train_mode=False)
    # With unanimous agreement and confidence 0.7, 2 draws (top/n = 1.0) suffice.
    assert lm.calls <= 2


def test_adaptive_sampling_uses_full_budget_on_disagreement():
    # Alternating answers never reach a settled majority → draws full budget.
    lm = CyclingLM(["A", "B", "C", "D", "E"])
    cfg = ExecutorConfig(
        escalation_enabled=True, k_samples=5,
        adaptive_sampling=True, adaptive_min_samples=1, adaptive_confidence=0.9,
        escalation_threshold=1.1,
    )
    ex = _make_executor(lm, cfg)
    ex.execute("question", train_mode=False)
    assert lm.calls == 5


# ---------------------------------------------------------------------------
# GRPO adaptive-k + fine-tune disable
# ---------------------------------------------------------------------------

def test_grpo_disabled_in_finetune_phase():
    from ppg.training.trainer import PPGTrainer, TrainerConfig

    cfg = TrainerConfig(k_grpo_paths=4, k_grpo_paths_finetune=1, grpo_adaptive=False)
    trainer = PPGTrainer.__new__(PPGTrainer)
    trainer.cfg = cfg

    class _Pol:
        def path_uncertainty(self, edges, phi):
            return 1.0  # very uncertain
    trainer.policy = _Pol()

    edges = [("a", "b"), ("b", "c")]
    phi = np.ones(4)

    class _Trace:
        edges_traversed = edges
    trace = _Trace()

    # Train phase keeps full budget; fine-tune collapses to 1 (no GRPO).
    assert trainer._grpo_k("train", trace, phi) == 4
    assert trainer._grpo_k("finetune", trace, phi) == 1


def test_grpo_adaptive_k_collapses_when_certain():
    from ppg.training.trainer import PPGTrainer, TrainerConfig

    cfg = TrainerConfig(
        k_grpo_paths=4, grpo_adaptive=True,
        grpo_uncertainty_threshold=0.05, k_grpo_min=2,
    )
    trainer = PPGTrainer.__new__(PPGTrainer)
    trainer.cfg = cfg

    class _Pol:
        def __init__(self, u):
            self._u = u
        def path_uncertainty(self, edges, phi):
            return self._u

    edges = [("a", "b")]
    phi = np.ones(4)

    class _Trace:
        edges_traversed = edges
    trace = _Trace()

    trainer.policy = _Pol(0.5)   # uncertain → full budget
    assert trainer._grpo_k("train", trace, phi) == 4
    trainer.policy = _Pol(0.0)   # converged → collapse to k_grpo_min
    assert trainer._grpo_k("train", trace, phi) == 2


# ---------------------------------------------------------------------------
# Mixed-model aux routing
# ---------------------------------------------------------------------------

def test_credit_assigner_uses_aux_lm_for_ablation():
    from ppg.data.fragments import build_graph
    from ppg.core.executor import PathTrace
    from ppg.core.features import RuntimeFeatures
    from ppg.training.credit import CreditAssigner, CreditAssignerConfig
    from ppg.training.reward import ExactMatchMetric

    graph = build_graph("gsm8k", topology="lean")
    asm = PromptAssembler(graph)
    main = CountingFakeLM(response="42")
    aux = CountingFakeLM(response="42")

    assigner = CreditAssigner(
        lm=main, assembler=asm, task_metric=ExactMatchMetric(),
        config=CreditAssignerConfig(skip_source=False, skip_terminal=False, min_path_length=2),
        aux_lm=aux,
    )

    node_ids = list(graph.nodes.keys())[:3]
    trace = PathTrace(
        node_ids=node_ids,
        edges_traversed=list(zip(node_ids, node_ids[1:])),
        assembled_prompt="p", token_count=10, lm_response="42",
        pre_lm_features=RuntimeFeatures(), post_lm_features=None,
        guard_decisions=[], escalated=False,
    )
    assigner.force_assign(trace, graph, x="q", y_star="42", node_id=node_ids[1])
    # The ablation call went to aux, not the main model.
    assert aux.calls == 1
    assert main.calls == 0


# ---------------------------------------------------------------------------
# Racing path calibration
# ---------------------------------------------------------------------------

def test_racing_reduces_candidate_paths():
    from ppg.data.fragments import build_graph
    from ppg.eval.path_search import _race_paths
    from ppg.eval.harness import EvalExample
    from ppg.training.reward import ExactMatchMetric

    graph = build_graph("gsm8k", topology="rich")
    paths = [list(p) for p in graph.all_paths()]
    assert len(paths) > 3
    subset = [EvalExample(x="q", y_star="42")]
    survivors = _race_paths(
        graph=graph, paths=paths, subset=subset,
        lm=CountingFakeLM(response="42"), metric=ExactMatchMetric(),
        constraint_checker=None, n_workers=1, path_runner=None, survivors=3,
    )
    assert len(survivors) == 3
    # Survivors are drawn from the original candidate list.
    assert all(s in paths for s in survivors)
