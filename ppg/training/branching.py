"""
AMPO-inspired failure-mode branching for PPG.

After accumulating enough reflection results, clusters failures by type and
creates specialized prompt branches for each failure cluster.

For example, if the reflection loop detects that 40% of failures are
constraint violations and 30% are reasoning errors, this module creates:
  - A specialized REASONING_STYLE fragment targeting reasoning failures
  - A specialized OUTPUT_CONTRACT fragment targeting constraint failures

The new branches are wired into the existing graph, giving the bandit
more specialized arms to route inputs to the right fragment variant.

Inspired by:
  - AMPO (EMNLP 2024): Pattern Recognition → Branch Adjustment → Branch Pruning
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ppg.core.executor import LMClient
from ppg.core.graph import FragmentType, Guard, PPGraph, PromptFragment
from ppg.training.reflection import ReflectionLoop, ReflectionResult


@dataclass
class BranchingConfig:
    enabled:              bool  = True
    min_reflections:      int   = 30
    branch_every:         int   = 1000
    min_failure_fraction: float = 0.2
    max_branches_per_mode: int  = 2
    max_total_branches:   int   = 6


_SPECIALISE_PROMPT = """\
You are creating a specialized prompt fragment to handle a specific failure mode.

FRAGMENT TYPE: {fragment_type}
FAILURE MODE: {failure_mode}
FAILURE EXAMPLES:
{examples}

CURRENT BEST TEMPLATE FOR THIS TYPE:
{current_template}

Write a new template specifically designed to prevent "{failure_mode}" failures.
Keep the {input} placeholder if present. Be precise and targeted.
Output ONLY the template text, nothing else."""


class FailureModeBrancher:
    """
    Detects dominant failure modes and creates specialized branches.

    Uses the ReflectionLoop's failure catalog to identify clusters,
    then generates targeted fragment variants for each cluster.
    """

    def __init__(
        self,
        lm:         LMClient,
        reflection: ReflectionLoop,
        config:     Optional[BranchingConfig] = None,
    ):
        self.lm         = lm
        self.reflection = reflection
        self.cfg        = config or BranchingConfig()
        self._branches_created: dict[str, list[str]] = {}
        self._n_branches: int = 0

    def maybe_branch(
        self,
        graph:   PPGraph,
        episode: int,
    ) -> list[str]:
        """Create specialized branches if failure patterns warrant it."""
        if not self.cfg.enabled:
            return []
        if episode % self.cfg.branch_every != 0 or episode == 0:
            return []
        if len(self.reflection.history) < self.cfg.min_reflections:
            return []
        if self._n_branches >= self.cfg.max_total_branches:
            return []

        actions = []
        top_modes = self.reflection.top_failure_modes(n=3)
        total_failures = sum(c for _, c in top_modes)

        for mode, count in top_modes:
            if count / max(1, total_failures) < self.cfg.min_failure_fraction:
                continue
            if mode in self._branches_created:
                if len(self._branches_created[mode]) >= self.cfg.max_branches_per_mode:
                    continue

            ftype = self._mode_to_fragment_type(mode)
            existing = graph.nodes_by_type(ftype)
            if not existing:
                continue

            best = max(existing, key=lambda f: f.utility)
            examples = self._get_failure_examples(mode)

            new_template = self._generate_specialised(ftype, mode, examples, best)
            if new_template:
                new_frag = PromptFragment(
                    id=str(uuid.uuid4()),
                    type=ftype,
                    template=new_template,
                    metadata={
                        "origin": "failure_branch",
                        "failure_mode": mode,
                        "episode_created": episode,
                    },
                )
                self._insert_branch(graph, new_frag, best)
                self._branches_created.setdefault(mode, []).append(new_frag.id)
                self._n_branches += 1
                actions.append(f"branch({mode}) -> {ftype.value} {new_frag.id[:8]}")

        return actions

    def _mode_to_fragment_type(self, mode: str) -> FragmentType:
        mapping = {
            "reasoning":  FragmentType.REASONING_STYLE,
            "format":     FragmentType.OUTPUT_CONTRACT,
            "constraint": FragmentType.OUTPUT_CONTRACT,
            "incomplete": FragmentType.REASONING_STYLE,
            "factual":    FragmentType.DOMAIN_PRIMER,
        }
        return mapping.get(mode, FragmentType.REASONING_STYLE)

    def _get_failure_examples(self, mode: str) -> list[str]:
        examples = []
        for r in reversed(self.reflection.history):
            if mode in r.failure_categories:
                summary = r.diagnosis[:200] if r.diagnosis else "No diagnosis"
                examples.append(summary)
                if len(examples) >= 3:
                    break
        return examples

    def _generate_specialised(
        self,
        ftype:    FragmentType,
        mode:     str,
        examples: list[str],
        best:     PromptFragment,
    ) -> Optional[str]:
        prompt = _SPECIALISE_PROMPT.format(
            fragment_type=ftype.value,
            failure_mode=mode,
            examples="\n".join(f"- {e}" for e in examples),
            current_template=best.template,
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

    def _insert_branch(self, graph: PPGraph, new_frag: PromptFragment,
                       sibling: PromptFragment):
        graph.nodes[new_frag.id] = new_frag
        for (src, dst), guard in list(graph.edges.items()):
            if dst == sibling.id:
                graph.edges[(src, new_frag.id)] = Guard.all_pass()
            if src == sibling.id:
                graph.edges[(new_frag.id, dst)] = Guard.all_pass()

    @property
    def stats(self) -> dict:
        return {
            "n_branches": self._n_branches,
            "branches_by_mode": {k: len(v) for k, v in self._branches_created.items()},
        }
