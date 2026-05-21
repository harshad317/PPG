"""
PPG FSM executor.

PPG-fast (default, primary): one LM call per query.
  1. pre-LM features from input
  2. walk graph: guards fire on phi, bandit breaks ties
  3. assemble prompt from path
  4. call LM once -> response

PPG-escalate (optional): second stage if sc_disagreement exceeds threshold.
  After step 4, sample k responses, compute disagreement; if high ->
  extend path with uncertainty_escalation node, call LM again.

PathTrace captures everything the trainer needs: path, edges, guard
decisions, features, token count, response, reward (filled later).
"""

from __future__ import annotations

import collections
import dataclasses
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ppg.core.features import FeatureExtractor, RuntimeFeatures
from ppg.core.graph import FragmentType, PPGraph


# ---------------------------------------------------------------------------
# PathTrace
# ---------------------------------------------------------------------------

@dataclass
class GuardDecision:
    src: str
    dst: str
    fired: bool
    phi: np.ndarray   # feature vector at decision point


@dataclass
class PathTrace:
    """Full record of one PPG execution episode."""
    node_ids:         list[str]                  # ordered visited nodes
    edges_traversed:  list[tuple[str, str]]      # (src, dst) pairs in order
    assembled_prompt: str
    token_count:      int
    lm_response:      str
    pre_lm_features:  RuntimeFeatures
    post_lm_features: Optional[RuntimeFeatures]  # None in PPG-fast
    guard_decisions:  list[GuardDecision]
    escalated:        bool
    reward:           Optional[float] = None     # filled by RewardComputer

    @property
    def path_length(self) -> int:
        return len(self.node_ids)

    @property
    def is_trivial(self) -> bool:
        """True if only source + terminal visited (no optional nodes selected)."""
        return self.path_length <= 2

    def node_set(self) -> frozenset[str]:
        return frozenset(self.node_ids)

    def with_reward(self, r: float) -> "PathTrace":
        return dataclasses.replace(self, reward=r)


# ---------------------------------------------------------------------------
# Protocol: NodeSelector (bandit policy interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class NodeSelector(Protocol):
    """
    Selects one node id from a list of candidates given feature vector phi.
    LinUCBPolicy will implement this. RandomSelector used in tests.
    """
    def select(
        self,
        current: Optional[str],
        candidates: list[str],
        phi: np.ndarray,
        train_mode: bool = True,
    ) -> str: ...

    def update(
        self,
        edge: tuple[str, str],
        phi: np.ndarray,
        reward: float,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Protocol: LMClient
# ---------------------------------------------------------------------------

@runtime_checkable
class LMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Built-in selectors (no bandit dependency)
# ---------------------------------------------------------------------------

class RandomSelector:
    """Uniform random node selection. For tests and ablations."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        current: Optional[str],
        candidates: list[str],
        phi: np.ndarray,
        train_mode: bool = True,
    ) -> str:
        return candidates[int(self._rng.integers(len(candidates)))]

    def update(self, edge: tuple[str, str], phi: np.ndarray, reward: float) -> None:
        pass  # stateless


class HighestUtilitySelector:
    """
    Selects node with highest utility score (PromptFragment.utility).
    Pre-bandit baseline; also used in PPG-fast eval mode (no exploration).
    """

    def __init__(self, graph: PPGraph):
        self._graph = graph

    def select(
        self,
        current: Optional[str],
        candidates: list[str],
        phi: np.ndarray,
        train_mode: bool = True,
    ) -> str:
        return max(candidates, key=lambda nid: self._graph.nodes[nid].utility)

    def update(self, edge: tuple[str, str], phi: np.ndarray, reward: float) -> None:
        pass  # utility updates handled by CreditAssigner


# ---------------------------------------------------------------------------
# PromptAssembler
# ---------------------------------------------------------------------------

class PromptAssembler:
    """
    Renders each fragment in path order and joins with separator.
    Skips blank rendered fragments so optional nodes with empty templates
    don't add empty lines.

    When structured=True, each fragment is prefixed with a markdown section
    header derived from its FragmentType. This helps the LM parse distinct
    roles (system context, task, reasoning strategy, output format) instead
    of treating the prompt as an undifferentiated blob.
    """

    _SECTION_HEADERS: dict[FragmentType, str] = {
        FragmentType.DOMAIN_PRIMER:          "## Role",
        FragmentType.TASK_FRAMING:           "## Task",
        FragmentType.FEW_SHOT:               "## Example",
        FragmentType.REASONING_STYLE:        "## Approach",
        FragmentType.COMPRESSION:            "## Brevity",
        FragmentType.OUTPUT_CONTRACT:        "## Output Requirements",
        FragmentType.VERIFICATION:           "## Verification",
        FragmentType.UNCERTAINTY_ESCALATION: "## Uncertainty Check",
        FragmentType.TOOL_USE:               "## Tool Use",
    }

    def __init__(self, graph: PPGraph, separator: str = "\n\n",
                 structured: bool = False):
        self.graph = graph
        self.separator = separator
        self.structured = structured

    def assemble(self, node_ids: list[str], context: dict) -> str:
        parts = []
        for nid in node_ids:
            node = self.graph.nodes[nid]
            rendered = node.render(context)
            if rendered.strip():
                if self.structured:
                    header = self._SECTION_HEADERS.get(node.type, "")
                    parts.append(f"{header}\n{rendered}" if header else rendered)
                else:
                    parts.append(rendered)
        return self.separator.join(parts)


# ---------------------------------------------------------------------------
# PPGExecutor
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    escalation_enabled:   bool  = False   # PPG-fast by default; use .production() for escalation
    k_samples:            int   = 1       # samples for consistency (>1 enables escalation)
    escalation_threshold: float = 0.4     # sc_disagreement above this -> escalate
    max_path_length:      int   = 10      # safety cap to prevent runaway paths
    prompt_separator:     str   = "\n\n"
    structured_prompts:   bool  = False   # add markdown section headers per fragment type
    sample_aggregation:   str   = "first" # "first" or "majority" over normalized answers
    escalation_template:  str   = ""      # fallback escalation text when graph has no node

    @classmethod
    def production(cls) -> "ExecutorConfig":
        """Tuned config for maximum benchmark performance.

        Enables self-consistency escalation with k=3 samples and a lower
        threshold to catch more reasoning errors.
        """
        return cls(
            escalation_enabled=True,
            k_samples=3,
            escalation_threshold=0.3,
            structured_prompts=True,
            sample_aggregation="majority",
            escalation_template=(
                "The sampled answers disagreed:\n{candidate_answers}\n\n"
                "Re-solve the task carefully, check the most likely failure point, "
                "and provide one final answer in the requested format."
            ),
        )


class PPGExecutor:
    """
    FSM runtime for PPG.

    Steps (PPG-fast, default):
      1. pre_lm_features from input
      2. _walk: guards fire -> bandit selects among active successors -> path
      3. PromptAssembler renders path -> prompt string
      4. LMClient.complete(prompt) -> response
      5. Return PathTrace (reward=None, filled by trainer)

    Steps (PPG-escalate, when config.escalation_enabled=True and k_samples>1):
      After step 4: sample k-1 more responses, compute sc_disagreement;
      if > threshold, extend path with UNCERTAINTY_ESCALATION node and
      call LM a second time.
    """

    def __init__(
        self,
        graph:             PPGraph,
        selector:          NodeSelector,
        lm:                LMClient,
        feature_extractor: FeatureExtractor,
        config:            Optional[ExecutorConfig] = None,
    ):
        self.graph     = graph
        self.selector  = selector
        self.lm        = lm
        self.fx        = feature_extractor
        self.cfg       = config or ExecutorConfig()
        self.assembler = PromptAssembler(
            graph, self.cfg.prompt_separator,
            structured=self.cfg.structured_prompts,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(
        self,
        x:          str,
        context:    Optional[dict] = None,
        train_mode: bool = True,
    ) -> PathTrace:
        """
        Run one episode on input x.

        context : template variables passed to fragment.render().
                  Defaults to {"input": x}.
        train_mode : True -> selector may explore (UCB bonus active).
                     False -> greedy (eval/inference mode).
        """
        ctx = context if context is not None else {"input": x}

        # Stage 1: pre-LM features
        pre_phi = self.fx.pre_lm(x)
        phi_vec = pre_phi.as_vector()

        # Stage 2: walk graph
        node_ids, edges, guard_decisions = self._walk(phi_vec, train_mode)

        # Stage 3: assemble prompt
        return self._complete_trace(
            x=x,
            ctx=ctx,
            node_ids=node_ids,
            edges=edges,
            guard_decisions=guard_decisions,
            pre_phi=pre_phi,
        )

    def execute_path(
        self,
        x:          str,
        node_ids:   list[str],
        context:    Optional[dict] = None,
    ) -> PathTrace:
        """
        Run one episode through a fixed path, while preserving the same LM
        sampling, aggregation, token counting, and escalation behavior used by
        dynamic PPG execution.
        """
        missing = [nid for nid in node_ids if nid not in self.graph.nodes]
        if missing:
            raise ValueError(f"fixed path contains unknown node ids: {missing}")

        ctx = context if context is not None else {"input": x}
        pre_phi = self.fx.pre_lm(x)
        edges = list(zip(node_ids, node_ids[1:]))

        return self._complete_trace(
            x=x,
            ctx=ctx,
            node_ids=list(node_ids),
            edges=edges,
            guard_decisions=[],
            pre_phi=pre_phi,
        )

    def _complete_trace(
        self,
        *,
        x:               str,
        ctx:             dict,
        node_ids:        list[str],
        edges:           list[tuple[str, str]],
        guard_decisions: list[GuardDecision],
        pre_phi:         RuntimeFeatures,
    ) -> PathTrace:
        """Assemble a selected path, call the LM, and optionally escalate."""
        prompt = self.assembler.assemble(node_ids, ctx)
        token_count = self._count_tokens(prompt)

        n_samples = self.cfg.k_samples if self.cfg.escalation_enabled and self.cfg.k_samples > 1 else 1
        samples = self._complete_many(prompt, n_samples)
        response = self._select_response(samples)

        post_phi: Optional[RuntimeFeatures] = None
        escalated = False

        if self.cfg.escalation_enabled and self.cfg.k_samples > 1:
            post_phi = self.fx.post_lm(x, samples)

            if post_phi.sc_disagreement > self.cfg.escalation_threshold:
                node_ids, edges, prompt, response, token_count, escalated = (
                    self._escalate(
                        node_ids,
                        edges,
                        ctx,
                        post_phi,
                        original_response=response,
                        samples=samples,
                        original_prompt=prompt,
                    )
                )

        return PathTrace(
            node_ids=node_ids,
            edges_traversed=edges,
            assembled_prompt=prompt,
            token_count=token_count,
            lm_response=response,
            pre_lm_features=pre_phi,
            post_lm_features=post_phi,
            guard_decisions=guard_decisions,
            escalated=escalated,
        )

    # ------------------------------------------------------------------
    # Graph walk
    # ------------------------------------------------------------------

    def _walk(
        self,
        phi_vec:    np.ndarray,
        train_mode: bool,
    ) -> tuple[list[str], list[tuple[str, str]], list[GuardDecision]]:
        """
        DFS walk from source to terminal, driven by guards + selector.
        Returns (node_ids, edges_traversed, guard_decisions).
        """
        guard_decisions: list[GuardDecision] = []
        edges: list[tuple[str, str]] = []

        # Pick starting node (usually one source)
        sources = list(self.graph.source_ids)
        if len(sources) == 1:
            current = sources[0]
        else:
            current = self.selector.select(None, sources, phi_vec, train_mode)

        node_ids = [current]
        steps = 0

        while steps < self.cfg.max_path_length:
            steps += 1

            all_successors = self.graph.successors(current)
            if not all_successors:
                break   # terminal node — no outgoing edges

            # Record guard fire/no-fire for every candidate edge
            active: list[str] = []
            for dst in all_successors:
                guard = self.graph.guard(current, dst)
                fired = guard.evaluate(phi_vec)
                guard_decisions.append(
                    GuardDecision(src=current, dst=dst, fired=fired, phi=phi_vec.copy())
                )
                if fired:
                    active.append(dst)

            if not active:
                # Fallback when all guards block (cold start or
                # over-restrictive thresholds) to avoid dead-ends.
                active = list(all_successors)

            # Selector breaks ties (or handles single active successor)
            next_node = (
                active[0]
                if len(active) == 1
                else self.selector.select(current, active, phi_vec, train_mode)
            )

            edges.append((current, next_node))
            node_ids.append(next_node)
            current = next_node

            if current in self.graph.terminal_ids:
                break

        return node_ids, edges, guard_decisions

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    def _escalate(
        self,
        node_ids:          list[str],
        edges:             list[tuple[str, str]],
        ctx:               dict,
        post_phi:          RuntimeFeatures,
        original_response: str = "",
        samples:           Optional[list[str]] = None,
        original_prompt:   str = "",
    ) -> tuple[list[str], list[tuple[str, str]], str, str, int, bool]:
        """
        Extend path with an UNCERTAINTY_ESCALATION node (if one exists in graph)
        and call LM again. Returns updated (node_ids, edges, prompt, response,
        token_count, escalated).
        """
        ctx = self._with_candidate_answers(ctx, samples or [original_response])
        esc_nodes = self.graph.nodes_by_type(FragmentType.UNCERTAINTY_ESCALATION)
        if not esc_nodes:
            if self.cfg.escalation_template.strip():
                base_prompt = original_prompt or self.assembler.assemble(node_ids, ctx)
                rendered = self.cfg.escalation_template.format_map(
                    collections.defaultdict(str, ctx)
                )
                prompt = self.cfg.prompt_separator.join(
                    p for p in (base_prompt, rendered) if p.strip()
                )
                response = self.lm.complete(prompt)
                return node_ids, edges, prompt, response, self._count_tokens(prompt), True

            prompt = original_prompt or self.assembler.assemble(node_ids, ctx)
            return node_ids, edges, prompt, original_response, self._count_tokens(prompt), False

        # Pick first escalation node not already in path
        esc_candidates = [n for n in esc_nodes if n.id not in node_ids]
        if not esc_candidates:
            prompt = self.assembler.assemble(node_ids, ctx)
            return node_ids, edges, prompt, original_response, self._count_tokens(prompt), False

        esc_node = esc_candidates[0]
        prev = node_ids[-1]
        node_ids = node_ids + [esc_node.id]
        edges = edges + [(prev, esc_node.id)]

        prompt   = self.assembler.assemble(node_ids, ctx)
        response = self.lm.complete(prompt)
        return node_ids, edges, prompt, response, self._count_tokens(prompt), True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _complete_many(self, prompt: str, n: int) -> list[str]:
        """Return n completions, using LM-native sampling when available."""
        if n <= 1:
            return [self.lm.complete(prompt)]

        sampler = getattr(self.lm, "sample", None)
        if callable(sampler):
            samples = list(sampler(prompt, n))
            if len(samples) >= n:
                return samples[:n]
            if samples:
                samples.extend(self.lm.complete(prompt) for _ in range(n - len(samples)))
                return samples

        return [self.lm.complete(prompt) for _ in range(n)]

    def _select_response(self, samples: list[str]) -> str:
        """Pick the response returned to callers from one or more samples."""
        if not samples:
            return ""
        if self.cfg.sample_aggregation != "majority" or len(samples) == 1:
            return samples[0]

        normalizer = getattr(self.fx, "normalizer", None)
        if normalizer is None:
            return samples[0]

        normalized = [normalizer(sample) for sample in samples]
        counts = Counter(normalized)
        best_answer, _ = counts.most_common(1)[0]
        for answer, sample in zip(normalized, samples):
            if answer == best_answer:
                return sample
        return samples[0]

    @staticmethod
    def _with_candidate_answers(ctx: dict, samples: list[str]) -> dict:
        candidate_answers = "\n".join(
            f"{i + 1}. {sample.strip()}" for i, sample in enumerate(samples)
        )
        return {**ctx, "candidate_answers": candidate_answers}

    def _count_tokens(self, text: str) -> int:
        from ppg.core.tokenizer import count_tokens
        return count_tokens(text)
