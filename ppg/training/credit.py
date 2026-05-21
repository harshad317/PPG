"""
LOO (Leave-One-Out) credit assignment for PPG fragment utility scoring.

For a fraction p_ablate of episodes, drops one non-boundary node from the
executed path, re-calls the LM on the shorter prompt, and computes the
marginal task-score improvement:

    marginal_v = r_task(full_path) - r_task(path_without_v)

Positive marginal -> node v improves accuracy -> high utility.
Negative marginal -> node v hurts accuracy -> low/negative utility.
Zero marginal     -> node v is inert (candidate for pruning).

The marginal is fed into PromptFragment.update_utility() which runs an
online mean (converges to E[marginal_v] across the training distribution).

Why LOO not Shapley
-------------------
  LOO  : O(1) LM call per episode, O(1) per node credit update
  Shapley: O(2^n) exact, O(n^2) approximate — prohibitive for n>4 nodes
  For n=5 optional nodes, Shapley is 32× more expensive. Not worth it for
  the first paper; Shapley in appendix on small graphs only.

Usage
-----
    assigner = CreditAssigner(lm, assembler, task_metric, p_ablate=0.15)
    marginals = assigner.maybe_assign(trace, graph, x, y_star, rng)
    # marginals: {node_id: float} or {} if this episode was skipped
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ppg.core.executor import LMClient, PathTrace, PromptAssembler
from ppg.core.graph import PPGraph
from ppg.training.reward import ConstraintChecker, TaskMetric


# ---------------------------------------------------------------------------
# CreditAssignmentResult
# ---------------------------------------------------------------------------

@dataclass
class CreditAssignmentResult:
    """
    Record of one LOO credit-assignment step.

    node_id       : the node that was ablated
    full_score    : r_task of the original full path
    ablated_score : r_task of the path without node_id
    marginal      : full_score - ablated_score
    ablated_path  : node_ids used for the ablated call (for logging)
    """
    node_id:       str
    full_score:    float
    ablated_score: float
    marginal:      float
    ablated_path:  list[str]

    @property
    def node_was_helpful(self) -> bool:
        return self.marginal > 0.0

    @property
    def node_was_harmful(self) -> bool:
        return self.marginal < 0.0


# ---------------------------------------------------------------------------
# CreditAssigner
# ---------------------------------------------------------------------------

@dataclass
class CreditAssignerConfig:
    p_ablate:         float = 0.15    # fraction of episodes that trigger LOO
    skip_source:      bool  = True    # never ablate the first node in path
    skip_terminal:    bool  = True    # never ablate the last node in path
    min_path_length:  int   = 3       # minimum path length for LOO to make sense


class CreditAssigner:
    """
    Runs leave-one-out ablations to assign per-fragment utility scores.

    Parameters
    ----------
    lm            : LMClient — called once per LOO ablation
    assembler     : PromptAssembler — builds ablated prompts
    task_metric   : TaskMetric — scores the ablated response
    credit_metric : optional TaskMetric used instead of task_metric for LOO
                    scoring.  Useful when the eval metric is noisy (e.g. F1)
                    but a cleaner signal (e.g. ExactMatch) exists for credit.
    config        : CreditAssignerConfig
    """

    def __init__(
        self,
        lm:                 LMClient,
        assembler:          PromptAssembler,
        task_metric:        TaskMetric,
        config:             Optional[CreditAssignerConfig] = None,
        constraint_checker: Optional[ConstraintChecker]   = None,
        constraint_as_task: bool                          = False,
        credit_metric:      Optional[TaskMetric]          = None,
    ):
        self.lm                 = lm
        self.asm                = assembler
        self.metric             = task_metric
        self._credit_metric     = credit_metric or task_metric
        self.cfg                = config or CreditAssignerConfig()
        self.checker            = constraint_checker
        self.constraint_as_task = constraint_as_task

        # Cumulative stats for diagnostics
        self._n_assignments: int = 0
        self._n_skipped:     int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_assign(
        self,
        trace:       PathTrace,
        graph:       PPGraph,
        x:           str,
        y_star:      str,
        rng:         np.random.Generator,
        constraints: Optional[list[str]] = None,
        metadata:    Optional[dict]      = None,
    ) -> Optional[CreditAssignmentResult]:
        """
        With probability p_ablate, run one LOO step and update the ablated
        node's utility score in-place on the graph.

        Returns CreditAssignmentResult if assignment ran, else None.

        Parameters
        ----------
        trace       : PathTrace from the current episode
        graph       : PPGraph (nodes mutated in-place via update_utility)
        x           : original input string
        y_star      : ground-truth reference answer
        rng         : numpy Generator for reproducible sampling
        constraints : constraint strings (used when constraint_as_task=True)
        metadata    : example metadata dict passed to constraint checker
        """
        ablatable = self._ablatable_nodes(trace)
        if not ablatable or rng.uniform() >= self.cfg.p_ablate:
            self._n_skipped += 1
            return None

        node_id = ablatable[int(rng.integers(len(ablatable)))]
        result  = self._run_loo(trace, graph, x, y_star, node_id, constraints, metadata)

        # Update fragment utility (online mean)
        graph.nodes[node_id].update_utility(result.marginal)
        self._n_assignments += 1

        return result

    def force_assign(
        self,
        trace:       PathTrace,
        graph:       PPGraph,
        x:           str,
        y_star:      str,
        node_id:     str,
        constraints: Optional[list[str]] = None,
        metadata:    Optional[dict]      = None,
    ) -> CreditAssignmentResult:
        """
        Force LOO for a specific node regardless of p_ablate.
        Used in tests and for targeted analysis of specific fragments.
        """
        result = self._run_loo(trace, graph, x, y_star, node_id, constraints, metadata)
        graph.nodes[node_id].update_utility(result.marginal)
        self._n_assignments += 1
        return result

    def assign_all(
        self,
        trace:       PathTrace,
        graph:       PPGraph,
        x:           str,
        y_star:      str,
        constraints: Optional[list[str]] = None,
        metadata:    Optional[dict]      = None,
    ) -> list[CreditAssignmentResult]:
        """
        Run LOO for every ablatable node in the path.
        Expensive — use only for analysis, not every episode.
        Returns list of results in path order.
        """
        results = []
        for node_id in self._ablatable_nodes(trace):
            result = self._run_loo(trace, graph, x, y_star, node_id, constraints, metadata)
            graph.nodes[node_id].update_utility(result.marginal)
            self._n_assignments += 1
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_assignments(self) -> int:
        return self._n_assignments

    @property
    def n_skipped(self) -> int:
        return self._n_skipped

    @property
    def assignment_rate(self) -> float:
        total = self._n_assignments + self._n_skipped
        return self._n_assignments / total if total > 0 else 0.0

    def fragment_utility_report(self, graph: PPGraph) -> dict[str, dict]:
        """
        Returns per-node utility summary for all nodes that have received
        at least one credit assignment.
        """
        report = {}
        for nid, frag in graph.nodes.items():
            if frag.utility_n > 0:
                report[nid] = {
                    "type":      frag.type.value,
                    "utility":   frag.utility,
                    "n_samples": frag.utility_n,
                }
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ablatable_nodes(self, trace: PathTrace) -> list[str]:
        """
        Returns the subset of path nodes eligible for LOO ablation.
        Excludes source (index 0) and terminal (last index) by default.
        """
        ids = trace.node_ids
        if len(ids) < self.cfg.min_path_length:
            return []

        start = 1 if self.cfg.skip_source    else 0
        end   = len(ids) - 1 if self.cfg.skip_terminal else len(ids)
        return ids[start:end]

    def _run_loo(
        self,
        trace:       PathTrace,
        graph:       PPGraph,
        x:           str,
        y_star:      str,
        node_id:     str,
        constraints: Optional[list[str]] = None,
        metadata:    Optional[dict]      = None,
    ) -> CreditAssignmentResult:
        """
        Compute marginal = r_task(full) - r_task(path_without_node_id).

        Uses credit_metric (defaults to task_metric) for cleaner LOO signal.
        When constraint_as_task=True and a constraint_checker is present,
        scores are computed via the checker instead.
        """
        # Full path score (re-score from cached response — no extra LM call)
        if self.constraint_as_task and self.checker is not None:
            full_score = self.checker.check(trace.lm_response, constraints or [], metadata)
        else:
            full_score = self._credit_metric.score(trace.lm_response, y_star)

        # Ablated path: remove node_id, re-assemble, call LM
        ablated_ids      = [n for n in trace.node_ids if n != node_id]
        ablated_prompt   = self.asm.assemble(ablated_ids, {"input": x})
        ablated_response = self.lm.complete(ablated_prompt)

        if self.constraint_as_task and self.checker is not None:
            ablated_score = self.checker.check(ablated_response, constraints or [], metadata)
        else:
            ablated_score = self._credit_metric.score(ablated_response, y_star)

        marginal = full_score - ablated_score

        return CreditAssignmentResult(
            node_id=node_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal=marginal,
            ablated_path=ablated_ids,
        )
