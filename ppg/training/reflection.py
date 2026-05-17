"""
GEPA-inspired textual reflection loop for PPG training.

After each episode where r_task < threshold, the ReflectionLoop:
  1. Diagnoses WHY the response failed (constraint violations, format errors,
     reasoning mistakes) using the LM itself as a reflector.
  2. Generates a structured error trace.
  3. Annotates the fragments in the path with failure reasons.
  4. Optionally proposes fragment mutations (used by Phase 3 evolution).

The key insight from GEPA (ICLR 2026): using textual error traces as rich
feedback signals — not just scalar rewards — dramatically improves prompt
mutation quality.  PPG adapts this by feeding reflection output into both
the fragment utility system and (later) the fragment evolution pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ppg.core.executor import LMClient, PathTrace
from ppg.core.graph import PPGraph
from ppg.training.reward import ConstraintChecker, RewardComponents


@dataclass
class ReflectionResult:
    """Structured output from one reflection step."""
    episode_idx:        int
    path_node_ids:      list[str]
    diagnosis:          str
    failure_categories: list[str]
    constraint_violations: list[str]
    fragment_annotations: dict[str, str]
    mutation_suggestions: list[str]

    @property
    def has_constraint_failure(self) -> bool:
        return len(self.constraint_violations) > 0

    @property
    def has_reasoning_failure(self) -> bool:
        return "reasoning" in self.failure_categories

    @property
    def has_format_failure(self) -> bool:
        return "format" in self.failure_categories


@dataclass
class ReflectionConfig:
    enabled:           bool  = True
    score_threshold:   float = 0.5
    max_diagnosis_tokens: int = 256
    reflect_fraction:  float = 0.3
    use_lm_reflection: bool  = True


_REFLECT_PROMPT = """\
You are analyzing why an LLM response failed to meet requirements.

TASK INPUT:
{input}

EXPECTED OUTPUT (reference):
{reference}

ACTUAL OUTPUT:
{response}

CONSTRAINTS: {constraints}

PROMPT USED:
{prompt}

Analyze the failure concisely:
1. CATEGORIES: List failure types (reasoning, format, constraint, factual, incomplete)
2. CONSTRAINT_VIOLATIONS: List each violated constraint
3. ROOT_CAUSE: One sentence explaining the primary failure reason
4. FRAGMENT_ISSUES: For each prompt section, note if it caused or failed to prevent the error
5. SUGGESTIONS: 1-2 concrete prompt modifications that would fix this failure

Format your response exactly as:
CATEGORIES: [comma-separated list]
VIOLATIONS: [comma-separated list, or "none"]
ROOT_CAUSE: [one sentence]
FRAGMENTS: [section_name: issue | section_name: issue]
SUGGESTIONS: [numbered list]
"""


class ReflectionLoop:
    """
    Runs textual reflection on failed episodes to extract rich error signals.

    Used by PPGTrainer to:
    - Annotate fragment utilities with qualitative failure reasons
    - Build a failure-mode catalog for Phase 3 branching
    - Generate mutation suggestions for Phase 3 evolution
    """

    def __init__(
        self,
        lm:                 LMClient,
        config:             Optional[ReflectionConfig] = None,
        constraint_checker: Optional[ConstraintChecker] = None,
    ):
        self.lm     = lm
        self.cfg    = config or ReflectionConfig()
        self.checker = constraint_checker
        self._history: list[ReflectionResult] = []
        self._failure_catalog: dict[str, int] = {}

    def maybe_reflect(
        self,
        episode_idx:  int,
        trace:        PathTrace,
        reward:       RewardComponents,
        x:            str,
        y_star:       str,
        graph:        PPGraph,
        constraints:  Optional[list[str]] = None,
        rng=None,
    ) -> Optional[ReflectionResult]:
        """Reflect on failed episode if score < threshold and within fraction budget."""
        if not self.cfg.enabled:
            return None
        if reward.task >= self.cfg.score_threshold:
            return None
        if rng is not None and rng.uniform() >= self.cfg.reflect_fraction:
            return None

        if self.cfg.use_lm_reflection:
            return self._lm_reflect(episode_idx, trace, x, y_star, graph, constraints)
        return self._rule_reflect(episode_idx, trace, reward, x, y_star, graph, constraints)

    def _lm_reflect(
        self,
        episode_idx: int,
        trace:       PathTrace,
        x:           str,
        y_star:      str,
        graph:       PPGraph,
        constraints: Optional[list[str]],
    ) -> ReflectionResult:
        """Use the LM itself to diagnose the failure."""
        prompt = _REFLECT_PROMPT.format(
            input=x[:500],
            reference=y_star[:200],
            response=trace.lm_response[:500],
            constraints=", ".join(constraints) if constraints else "none",
            prompt=trace.assembled_prompt[:300],
        )
        diagnosis = self.lm.complete(prompt)
        return self._parse_reflection(episode_idx, trace, diagnosis, graph)

    def _rule_reflect(
        self,
        episode_idx: int,
        trace:       PathTrace,
        reward:      RewardComponents,
        x:           str,
        y_star:      str,
        graph:       PPGraph,
        constraints: Optional[list[str]],
    ) -> ReflectionResult:
        """Fast rule-based reflection without LM call."""
        categories = []
        violations = []
        annotations = {}

        if reward.constraint > 0 and reward.constraint < 1.0:
            categories.append("constraint")
            if constraints and self.checker:
                resp_lower = trace.lm_response.lower()
                for c in constraints:
                    if c.lower() not in resp_lower:
                        violations.append(c)

        if reward.task < 0.5:
            categories.append("reasoning")

        response = trace.lm_response.strip()
        if not response or len(response) < 10:
            categories.append("incomplete")
        if y_star.strip().isdigit() and not any(c.isdigit() for c in response[-20:]):
            categories.append("format")

        for nid in trace.node_ids:
            frag = graph.nodes.get(nid)
            if frag:
                ftype = frag.type.value
                if ftype == "output_contract" and "format" in categories:
                    annotations[nid] = "output contract failed to enforce format"
                elif ftype == "reasoning_style" and "reasoning" in categories:
                    annotations[nid] = "reasoning style insufficient for this input"

        result = ReflectionResult(
            episode_idx=episode_idx,
            path_node_ids=trace.node_ids,
            diagnosis=f"Rule-based: {', '.join(categories) or 'unknown'}",
            failure_categories=categories,
            constraint_violations=violations,
            fragment_annotations=annotations,
            mutation_suggestions=[],
        )
        self._record(result)
        return result

    def _parse_reflection(
        self,
        episode_idx: int,
        trace:       PathTrace,
        diagnosis:   str,
        graph:       PPGraph,
    ) -> ReflectionResult:
        """Parse structured LM reflection output."""
        categories = _extract_field(diagnosis, "CATEGORIES")
        violations = _extract_field(diagnosis, "VIOLATIONS")
        suggestions = _extract_numbered(diagnosis, "SUGGESTIONS")

        cat_list = [c.strip().lower() for c in categories.split(",") if c.strip()]
        viol_list = [v.strip() for v in violations.split(",")
                     if v.strip() and v.strip().lower() != "none"]

        annotations = {}
        frag_section = _extract_field(diagnosis, "FRAGMENTS")
        for part in frag_section.split("|"):
            part = part.strip()
            if ":" in part:
                name, issue = part.split(":", 1)
                for nid in trace.node_ids:
                    frag = graph.nodes.get(nid)
                    if frag and frag.type.value.replace("_", " ") in name.lower():
                        annotations[nid] = issue.strip()

        result = ReflectionResult(
            episode_idx=episode_idx,
            path_node_ids=trace.node_ids,
            diagnosis=diagnosis,
            failure_categories=cat_list,
            constraint_violations=viol_list,
            fragment_annotations=annotations,
            mutation_suggestions=suggestions,
        )
        self._record(result)
        return result

    def _record(self, result: ReflectionResult):
        self._history.append(result)
        for cat in result.failure_categories:
            self._failure_catalog[cat] = self._failure_catalog.get(cat, 0) + 1

    @property
    def history(self) -> list[ReflectionResult]:
        return list(self._history)

    @property
    def failure_catalog(self) -> dict[str, int]:
        return dict(self._failure_catalog)

    def top_failure_modes(self, n: int = 5) -> list[tuple[str, int]]:
        """Most common failure categories across all reflections."""
        return sorted(self._failure_catalog.items(), key=lambda x: -x[1])[:n]

    def annotations_for_node(self, node_id: str) -> list[str]:
        """All failure annotations accumulated for a specific fragment."""
        return [
            r.fragment_annotations[node_id]
            for r in self._history
            if node_id in r.fragment_annotations
        ]


def _extract_field(text: str, field_name: str) -> str:
    pattern = rf"{field_name}:\s*(.+?)(?:\n[A-Z_]+:|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_numbered(text: str, field_name: str) -> list[str]:
    raw = _extract_field(text, field_name)
    items = re.findall(r"\d+[.)]\s*(.+?)(?=\d+[.)]|$)", raw, re.DOTALL)
    return [item.strip() for item in items if item.strip()]
