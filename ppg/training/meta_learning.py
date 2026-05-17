"""
Cross-benchmark meta-learning for PPG.

Bilevel optimization: outer loop learns shared fragment utilities and guard
weight priors across multiple benchmarks; inner loop adapts routing for a
specific benchmark using few episodes.

Meta-learning enables:
  1. Fast adaptation to new benchmarks (few-shot routing)
  2. Transfer of routing intuitions across task types
  3. Shared understanding of which fragment types help which input patterns

Inspired by:
  - Choi & Baek (2025): Bilevel system prompt optimization
  - MAML-style meta-learning adapted for contextual bandits
  - mmGRPO (2025): composing policy gradients across modules
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ppg.bandits.linucb import LinUCBArm, LinUCBPolicy
from ppg.core.features import FEATURE_DIM
from ppg.core.graph import PPGraph


@dataclass
class MetaLearningConfig:
    n_meta_iterations:    int   = 10
    n_inner_episodes:     int   = 100
    inner_alpha:          float = 0.5
    outer_learning_rate:  float = 0.1
    n_eval_episodes:      int   = 50
    meta_reg:             float = 0.01


@dataclass
class TaskResult:
    benchmark:   str
    inner_reward: float
    eval_reward:  float
    n_episodes:  int


class MetaLearner:
    """
    MAML-inspired meta-learner for LinUCB policies across benchmarks.

    The key insight: guard weights (mu_hat) learned on one benchmark partially
    transfer to others. Meta-learning finds a shared initialization (meta-prior)
    that enables fast adaptation.

    Algorithm:
      1. For each meta-iteration:
         a. For each benchmark task:
            - Clone the meta-policy
            - Train inner loop (n_inner_episodes) on this benchmark
            - Evaluate on held-out examples
         b. Compute meta-gradient: average (adapted - meta) weighted by eval reward
         c. Update meta-policy: meta += lr * meta_gradient

    The meta-policy serves as initialization for any new benchmark, requiring
    only a short inner-loop adaptation (~100 episodes) instead of full training.
    """

    def __init__(
        self,
        meta_graph: PPGraph,
        config:     Optional[MetaLearningConfig] = None,
    ):
        self.graph  = meta_graph
        self.cfg    = config or MetaLearningConfig()
        self._meta_policy = LinUCBPolicy(meta_graph, alpha=self.cfg.inner_alpha)
        self._history: list[dict] = []

    @property
    def meta_policy(self) -> LinUCBPolicy:
        return self._meta_policy

    def meta_train(
        self,
        task_train_fn,
        task_eval_fn,
        benchmarks: list[str],
    ) -> list[dict]:
        """
        Run meta-training across benchmarks.

        task_train_fn(policy, benchmark, n_episodes) -> trained_policy
        task_eval_fn(policy, benchmark, n_episodes) -> mean_reward

        Returns per-iteration metrics.
        """
        iteration_results = []

        for meta_iter in range(self.cfg.n_meta_iterations):
            task_results = []
            adapted_policies = []

            for benchmark in benchmarks:
                adapted = self._clone_policy()
                adapted = task_train_fn(adapted, benchmark, self.cfg.n_inner_episodes)
                eval_reward = task_eval_fn(adapted, benchmark, self.cfg.n_eval_episodes)

                task_results.append(TaskResult(
                    benchmark=benchmark,
                    inner_reward=0.0,
                    eval_reward=eval_reward,
                    n_episodes=self.cfg.n_inner_episodes,
                ))
                adapted_policies.append(adapted)

            self._meta_update(adapted_policies, task_results)

            iter_summary = {
                "meta_iteration": meta_iter,
                "task_results": [
                    {"benchmark": t.benchmark, "eval_reward": t.eval_reward}
                    for t in task_results
                ],
                "mean_eval_reward": float(np.mean([t.eval_reward for t in task_results])),
            }
            iteration_results.append(iter_summary)
            self._history.append(iter_summary)

        return iteration_results

    def adapt(self, train_fn, benchmark: str) -> LinUCBPolicy:
        """
        Adapt meta-policy to a new benchmark using inner loop training.
        Returns adapted policy ready for evaluation.
        """
        adapted = self._clone_policy()
        return train_fn(adapted, benchmark, self.cfg.n_inner_episodes)

    def _meta_update(
        self,
        adapted_policies: list[LinUCBPolicy],
        task_results:     list[TaskResult],
    ) -> None:
        """Reptile-style meta-update: move meta toward average of adapted policies."""
        if not adapted_policies:
            return

        total_reward = sum(t.eval_reward for t in task_results)
        if total_reward <= 0:
            weights = [1.0 / len(task_results)] * len(task_results)
        else:
            weights = [max(0, t.eval_reward) / total_reward for t in task_results]

        lr = self.cfg.outer_learning_rate

        for edge, meta_arm in self._meta_policy._arms.items():
            deltas_A = np.zeros_like(meta_arm.A)
            deltas_b = np.zeros_like(meta_arm.b)

            for policy, w in zip(adapted_policies, weights):
                if edge in policy._arms:
                    adapted_arm = policy._arms[edge]
                    deltas_A += w * (adapted_arm.A - meta_arm.A)
                    deltas_b += w * (adapted_arm.b - meta_arm.b)

            meta_arm.A += lr * deltas_A
            meta_arm.b += lr * deltas_b

            # Regularize toward identity to prevent drift
            reg_target = meta_arm.lambda_reg * np.eye(meta_arm.feature_dim)
            meta_arm.A += self.cfg.meta_reg * (reg_target - meta_arm.A)

    def _clone_policy(self) -> LinUCBPolicy:
        """Deep-copy the meta-policy for inner-loop training."""
        cloned = LinUCBPolicy(
            self.graph,
            feature_dim=self._meta_policy.feature_dim,
            alpha=self.cfg.inner_alpha,
            lambda_reg=self._meta_policy.lambda_reg,
        )
        for edge, arm in self._meta_policy._arms.items():
            if edge in cloned._arms:
                cloned._arms[edge].A = arm.A.copy()
                cloned._arms[edge].b = arm.b.copy()
                cloned._arms[edge].n_updates = arm.n_updates
        return cloned

    def save(self, path: str) -> None:
        self._meta_policy.save(path)

    def load(self, path: str) -> None:
        self._meta_policy.load(path)

    @property
    def history(self) -> list[dict]:
        return list(self._history)
