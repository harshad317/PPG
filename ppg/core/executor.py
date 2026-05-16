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

import dataclasses
from dataclasses import dataclass, field
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
    """

    def __init__(self, graph: PPGraph, separator: str = "\n\n"):
        self.graph = graph
        self.separator = separator

    def assemble(self, node_ids: list[str], context: dict) -> str:
        parts = []
        for nid in node_ids:
            rendered = self.graph.nodes[nid].render(context)
            if rendered.strip():
                parts.append(rendered)
        return self.separator.join(parts)


# ---------------------------------------------------------------------------
# PPGExecutor
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    escalation_enabled:   bool  = False   # PPG-fast by default
    k_samples:            int   = 1       # samples for consistency (>1 enables escalation)
    escalation_threshold: float = 0.4     # sc_disagreement above this -> escalate
    max_path_length:      int   = 10      # safety cap to prevent runaway paths
    prompt_separator:     str   = "\n\n"


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
        self.assembler = PromptAssembler(graph, self.cfg.prompt_separator)

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
        prompt = self.assembler.assemble(node_ids, ctx)
        token_count = self._count_tokens(prompt)

        # Stage 4: primary LM call (PPG-fast)
        response = self.lm.complete(prompt)

        # Stage 5: optional escalation
        post_phi: Optional[RuntimeFeatures] = None
        escalated = False

        if self.cfg.escalation_enabled and self.cfg.k_samples > 1:
            extra_samples = [self.lm.complete(prompt)
                             for _ in range(self.cfg.k_samples - 1)]
            all_samples = [response] + extra_samples
            post_phi = self.fx.post_lm(x, all_samples)

            if post_phi.sc_disagreement > self.cfg.escalation_threshold:
                node_ids, edges, prompt, response, token_count, escalated = (
                    self._escalate(node_ids, edges, ctx, post_phi,
                                   original_response=response)
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

            if not active or not train_mode:
                # At eval time, bypass guard filtering so the bandit selects
                # from the same full successor set it trained against (guards
                # were all-pass during training; filtering post-sync would
                # block edges the bandit never learned to route around).
                # Also fallback when guards block all successors to avoid
                # structural dead-ends.
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
    ) -> tuple[list[str], list[tuple[str, str]], str, str, int, bool]:
        """
        Extend path with an UNCERTAINTY_ESCALATION node (if one exists in graph)
        and call LM again. Returns updated (node_ids, edges, prompt, response,
        token_count, escalated).
        """
        esc_nodes = self.graph.nodes_by_type(FragmentType.UNCERTAINTY_ESCALATION)
        if not esc_nodes:
            # No escalation node in graph; return unchanged without extra LM call
            prompt = self.assembler.assemble(node_ids, ctx)
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

    def _count_tokens(self, text: str) -> int:
        """Whitespace-split proxy. Replaced by tiktoken when available."""
        return len(text.split())
