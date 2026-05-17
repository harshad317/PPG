"""
Fragment evolution for PPG — EvoPrompt/PromptBreeder-inspired mutation.

Instead of a fixed fragment library, evolve fragment text during training.
Low-utility fragments are mutated using:
  1. Failure annotations from the ReflectionLoop
  2. LM-guided rewriting (the LM proposes improved fragment text)
  3. Crossover between high-utility and low-utility fragments of same type

The graph topology stays frozen (edges unchanged) but node TEMPLATES evolve.
New variant nodes can be inserted at the same graph level, giving the bandit
more arms to explore.

Inspired by:
  - EvoPrompt (2023): LM-driven mutation + crossover operators
  - PromptBreeder (2023): self-referential prompt evolution
  - GEPA (2026): reflection-guided mutation using error traces
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ppg.core.executor import LMClient
from ppg.core.graph import FragmentType, Guard, PPGraph, PromptFragment
from ppg.training.reflection import ReflectionLoop


@dataclass
class EvolutionConfig:
    enabled:               bool  = True
    evolve_every:          int   = 500
    min_utility_samples:   int   = 5
    utility_threshold:     float = 0.0
    max_variants_per_type: int   = 5
    mutation_temperature:  float = 0.7
    crossover_prob:        float = 0.3
    prune_threshold:       float = -0.1
    min_prune_samples:     int   = 10


_MUTATE_PROMPT = """\
You are improving a prompt fragment used in an LLM pipeline.

FRAGMENT TYPE: {fragment_type}
CURRENT TEMPLATE:
{template}

FAILURE ANNOTATIONS (from recent episodes where this fragment was in the path):
{annotations}

TASK CONTEXT: This fragment is part of a prompt graph for {benchmark} tasks.

Write an improved version of this template that addresses the failure patterns.
Keep the same {input} placeholder if present. Keep it concise.
Output ONLY the improved template text, nothing else."""

_CROSSOVER_PROMPT = """\
You are combining two prompt fragments into a better one.

FRAGMENT TYPE: {fragment_type}

HIGH-PERFORMING TEMPLATE (utility={high_utility:.3f}):
{high_template}

LOW-PERFORMING TEMPLATE (utility={low_utility:.3f}):
{low_template}

Combine the best aspects of both into a single improved template.
Keep any {input} placeholder. Keep it concise.
Output ONLY the combined template text, nothing else."""


class FragmentEvolver:
    """
    Evolves fragment text during training using LM-guided mutation.

    Called periodically by PPGTrainer (every evolve_every episodes).
    Mutates low-utility fragments, optionally inserts new variant nodes,
    and prunes consistently harmful fragments.
    """

    def __init__(
        self,
        lm:          LMClient,
        config:      Optional[EvolutionConfig] = None,
        reflection:  Optional[ReflectionLoop]  = None,
        benchmark:   str = "unknown",
    ):
        self.lm         = lm
        self.cfg        = config or EvolutionConfig()
        self.reflection = reflection
        self.benchmark  = benchmark

        self._n_mutations:  int = 0
        self._n_crossovers: int = 0
        self._n_pruned:     int = 0
        self._generation:   int = 0
        self._lineage: dict[str, list[str]] = {}

    def maybe_evolve(
        self,
        graph:   PPGraph,
        episode: int,
        rng:     np.random.Generator,
    ) -> list[str]:
        """
        If episode is a multiple of evolve_every, run one evolution cycle.
        Returns list of actions taken (for logging).
        """
        if not self.cfg.enabled:
            return []
        if episode % self.cfg.evolve_every != 0 or episode == 0:
            return []

        self._generation += 1
        actions = []

        actions.extend(self._mutate_weak_fragments(graph, rng))
        actions.extend(self._crossover_fragments(graph, rng))
        actions.extend(self._prune_harmful_fragments(graph))

        return actions

    def _mutate_weak_fragments(
        self,
        graph: PPGraph,
        rng:   np.random.Generator,
    ) -> list[str]:
        """Mutate fragments with low utility using reflection annotations."""
        actions = []
        for nid, frag in list(graph.nodes.items()):
            if frag.utility_n < self.cfg.min_utility_samples:
                continue
            if frag.utility >= self.cfg.utility_threshold:
                continue
            if frag.type in (FragmentType.TASK_FRAMING, FragmentType.OUTPUT_CONTRACT):
                if self._count_variants(graph, frag.type) >= 2:
                    continue

            annotations = self._get_annotations(nid)
            if not annotations:
                annotations = [f"Low utility ({frag.utility:.3f}) after {frag.utility_n} samples"]

            new_template = self._lm_mutate(frag, annotations)
            if new_template and new_template != frag.template:
                frag.template = new_template
                frag.utility = 0.0
                frag.utility_n = 0
                self._n_mutations += 1
                self._lineage.setdefault(nid, []).append(f"mutated_gen{self._generation}")
                actions.append(f"mutated {frag.type.value} node {nid[:8]}")

        return actions

    def _crossover_fragments(
        self,
        graph: PPGraph,
        rng:   np.random.Generator,
    ) -> list[str]:
        """Cross high-utility and low-utility fragments of same type."""
        actions = []
        by_type: dict[FragmentType, list[PromptFragment]] = defaultdict(list)
        for frag in graph.nodes.values():
            if frag.utility_n >= self.cfg.min_utility_samples:
                by_type[frag.type].append(frag)

        for ftype, frags in by_type.items():
            if len(frags) < 2:
                continue
            if rng.uniform() >= self.cfg.crossover_prob:
                continue
            if self._count_variants(graph, ftype) >= self.cfg.max_variants_per_type:
                continue

            sorted_frags = sorted(frags, key=lambda f: f.utility, reverse=True)
            high = sorted_frags[0]
            low = sorted_frags[-1]

            if high.utility <= low.utility:
                continue

            new_template = self._lm_crossover(high, low)
            if new_template:
                new_frag = PromptFragment(
                    id=str(uuid.uuid4()),
                    type=ftype,
                    template=new_template,
                    metadata={"origin": "crossover", "generation": self._generation,
                              "parents": [high.id, low.id]},
                )
                self._insert_variant(graph, new_frag, high)
                self._n_crossovers += 1
                self._lineage[new_frag.id] = [
                    f"crossover({high.id[:8]},{low.id[:8]})_gen{self._generation}"
                ]
                actions.append(f"crossover {ftype.value} -> {new_frag.id[:8]}")

        return actions

    def _prune_harmful_fragments(self, graph: PPGraph) -> list[str]:
        """Remove consistently harmful fragments (very negative utility)."""
        actions = []
        to_remove = []
        for nid, frag in graph.nodes.items():
            if frag.utility_n < self.cfg.min_prune_samples:
                continue
            if frag.utility > self.cfg.prune_threshold:
                continue
            if frag.type in (FragmentType.TASK_FRAMING, FragmentType.OUTPUT_CONTRACT):
                if self._count_variants(graph, frag.type) <= 1:
                    continue
            if nid in graph.source_ids or nid in graph.terminal_ids:
                continue
            to_remove.append(nid)

        for nid in to_remove:
            self._remove_node(graph, nid)
            self._n_pruned += 1
            actions.append(f"pruned node {nid[:8]}")

        return actions

    def _lm_mutate(self, frag: PromptFragment, annotations: list[str]) -> Optional[str]:
        prompt = _MUTATE_PROMPT.format(
            fragment_type=frag.type.value,
            template=frag.template,
            annotations="\n".join(f"- {a}" for a in annotations[:5]),
            benchmark=self.benchmark,
            input="{input}",
        )
        try:
            result = self.lm.complete(prompt)
            result = result.strip().strip('"').strip("'")
            if len(result) < 10 or len(result) > 2000:
                return None
            return result
        except Exception:
            return None

    def _lm_crossover(self, high: PromptFragment, low: PromptFragment) -> Optional[str]:
        prompt = _CROSSOVER_PROMPT.format(
            fragment_type=high.type.value,
            high_utility=high.utility,
            high_template=high.template,
            low_utility=low.utility,
            low_template=low.template,
            input="{input}",
        )
        try:
            result = self.lm.complete(prompt)
            result = result.strip().strip('"').strip("'")
            if len(result) < 10 or len(result) > 2000:
                return None
            return result
        except Exception:
            return None

    def _get_annotations(self, nid: str) -> list[str]:
        if self.reflection is None:
            return []
        return self.reflection.annotations_for_node(nid)[:5]

    def _count_variants(self, graph: PPGraph, ftype: FragmentType) -> int:
        return sum(1 for f in graph.nodes.values() if f.type == ftype)

    def _insert_variant(self, graph: PPGraph, new_frag: PromptFragment,
                        sibling: PromptFragment):
        """Insert new_frag into graph with same connectivity as sibling."""
        graph.nodes[new_frag.id] = new_frag

        for (src, dst), guard in list(graph.edges.items()):
            if dst == sibling.id:
                graph.edges[(src, new_frag.id)] = Guard.all_pass()
            if src == sibling.id:
                graph.edges[(new_frag.id, dst)] = Guard.all_pass()

    def _remove_node(self, graph: PPGraph, nid: str):
        """Remove node and its edges from graph."""
        graph.nodes.pop(nid, None)
        edges_to_remove = [(s, d) for s, d in graph.edges if s == nid or d == nid]
        for edge in edges_to_remove:
            graph.edges.pop(edge, None)

    @property
    def stats(self) -> dict:
        return {
            "generation": self._generation,
            "n_mutations": self._n_mutations,
            "n_crossovers": self._n_crossovers,
            "n_pruned": self._n_pruned,
        }
