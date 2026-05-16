"""
3-phase PPGTrainer for the Prompt Policy Graph.

Phase 1 — Warm-up (n_warmup_episodes)
    Random routing (RandomSelector) so all edges get explored.
    LinUCBPolicy receives offline updates from random trajectories.
    CreditAssigner runs at p_ablate_warmup to seed fragment utilities.

Phase 2 — Bandit training (n_train_episodes)
    LinUCBPolicy drives routing (train_mode=True, alpha=alpha_train).
    Full reward + credit assignment at p_ablate_train.

Phase 3 — Fine-tuning (n_finetune_episodes)
    LinUCBPolicy in low-exploration mode (alpha=alpha_finetune).
    Credit assignment at p_ablate_finetune for final utility calibration.

Dataset cycling
    If total episodes > len(dataset), examples cycle with a shuffle each
    epoch so no fixed ordering bias.

Checkpointing
    Policy saved every checkpoint_every episodes when checkpoint_dir is set.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


def _make_bar(iterable, *, desc: str, enabled: bool, total: int, **kwargs):
    """Wrap iterable with tqdm if enabled. Raises ImportError if tqdm missing."""
    try:
        from tqdm import tqdm
    except ImportError:
        if enabled:
            raise ImportError("tqdm required for progress display: pip install tqdm") from None
        return iterable
    return tqdm(iterable, desc=desc, total=total, disable=not enabled,
                unit="ep", ncols=100, leave=True, **kwargs)

from ppg.bandits.linucb import LinUCBPolicy
from ppg.core.executor import PPGExecutor, RandomSelector
from ppg.training.credit import CreditAssigner, CreditAssignmentResult
from ppg.training.reward import RewardComponents, RewardComputer


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    x:           str
    y_star:      str
    constraints: list[str] = field(default_factory=list)


@dataclass
class EpisodeResult:
    phase:       str              # "warmup" | "train" | "finetune"
    episode:     int              # global episode index
    reward:      RewardComponents
    credit:      Optional[CreditAssignmentResult]
    path:        list[str]        # node_ids visited
    token_count: int


@dataclass
class TrainerConfig:
    # Phase lengths
    n_warmup_episodes:   int   = 200
    n_train_episodes:    int   = 1000
    n_finetune_episodes: int   = 200

    # Exploration alphas
    alpha_train:    float = 0.5
    alpha_finetune: float = 0.1

    # Credit assignment probabilities per phase
    p_ablate_warmup:    float = 0.10
    p_ablate_train:     float = 0.15
    p_ablate_finetune:  float = 0.05

    # Checkpointing (disabled when None)
    checkpoint_dir:   Optional[str] = None
    checkpoint_every: int           = 100

    # Reproducibility
    seed: int = 0

    # Progress display
    show_progress: bool = True


# ---------------------------------------------------------------------------
# TrainingStats — lightweight result container
# ---------------------------------------------------------------------------

class TrainingStats:
    """Collects per-episode results; provides aggregate accessors."""

    def __init__(self) -> None:
        self._results: list[EpisodeResult] = []

    def record(self, result: EpisodeResult) -> None:
        self._results.append(result)

    @property
    def results(self) -> list[EpisodeResult]:
        return list(self._results)

    def reward_history(self, phase: Optional[str] = None) -> list[float]:
        """Total reward per episode, optionally filtered by phase."""
        return [
            r.reward.total for r in self._results
            if phase is None or r.phase == phase
        ]

    def mean_reward(self, phase: Optional[str] = None) -> float:
        h = self.reward_history(phase)
        return float(np.mean(h)) if h else 0.0

    def task_accuracy(self, phase: Optional[str] = None) -> float:
        tasks = [
            r.reward.task for r in self._results
            if phase is None or r.phase == phase
        ]
        return float(np.mean(tasks)) if tasks else 0.0

    def n_episodes(self, phase: Optional[str] = None) -> int:
        return sum(1 for r in self._results if phase is None or r.phase == phase)

    def summary(self) -> dict[str, dict]:
        out = {}
        for ph in ("warmup", "train", "finetune"):
            n = self.n_episodes(ph)
            if n == 0:
                continue
            out[ph] = {
                "n_episodes":    n,
                "mean_reward":   round(self.mean_reward(ph), 4),
                "task_accuracy": round(self.task_accuracy(ph), 4),
            }
        return out


# ---------------------------------------------------------------------------
# PPGTrainer
# ---------------------------------------------------------------------------

class PPGTrainer:
    """
    Orchestrates 3-phase PPG training.

    Parameters
    ----------
    executor        : PPGExecutor configured with LinUCBPolicy as selector
    policy          : LinUCBPolicy — same object as executor.selector
    reward_computer : RewardComputer
    credit_assigner : CreditAssigner
    config          : TrainerConfig
    on_episode      : optional callback called after each episode with
                      (episode_index, EpisodeResult); useful for logging
    """

    def __init__(
        self,
        executor:        PPGExecutor,
        policy:          LinUCBPolicy,
        reward_computer: RewardComputer,
        credit_assigner: CreditAssigner,
        config:          Optional[TrainerConfig] = None,
        on_episode:      Optional[Callable[[int, EpisodeResult], None]] = None,
    ):
        self.executor  = executor
        self.policy    = policy
        self.reward    = reward_computer
        self.credit    = credit_assigner
        self.cfg       = config or TrainerConfig()
        self.on_episode = on_episode

        self._rng      = np.random.default_rng(self.cfg.seed)
        self._py_rng   = random.Random(self.cfg.seed)
        self._stats    = TrainingStats()
        self._episode  = 0  # global counter
        # False when executor.selector is a permanent RandomSelector (no_bandit ablation).
        # Prevents policy updates and guard sync from contaminating the ablation baseline.
        self._train_policy: bool = (self.executor.selector is self.policy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, dataset: list[TrainingExample]) -> TrainingStats:
        """
        Run all three training phases over the dataset.
        Dataset is cycled (with shuffle) if episodes > len(dataset).
        Returns accumulated TrainingStats.
        """
        if not dataset:
            raise ValueError("dataset must be non-empty")

        self._run_warmup(dataset)
        self._run_train(dataset)
        self._run_finetune(dataset)

        # Sync guards only when policy drives selection (not no_bandit ablation)
        if self._train_policy:
            self.policy.sync_guards(self.executor.graph)

        return self._stats

    @property
    def stats(self) -> TrainingStats:
        return self._stats

    # ------------------------------------------------------------------
    # Phase runners
    # ------------------------------------------------------------------

    def _run_warmup(self, dataset: list[TrainingExample]) -> None:
        """Phase 1: random routing; offline policy updates."""
        random_selector = RandomSelector(seed=self.cfg.seed)
        original_selector = self.executor.selector
        self.executor.selector = random_selector

        original_p = self.credit.cfg.p_ablate
        self.credit.cfg.p_ablate = self.cfg.p_ablate_warmup

        cycle = self._make_cycle(dataset, self.cfg.n_warmup_episodes)
        bar = _make_bar(cycle, desc="warmup  ", enabled=self.cfg.show_progress,
                        total=self.cfg.n_warmup_episodes)
        run_r = run_t = 0.0
        for i, example in enumerate(bar):
            result = self._run_episode(example, phase="warmup", train_mode=False)
            run_r  = (run_r * i + result.reward.total) / (i + 1)
            run_t  = (run_t * i + result.reward.task)  / (i + 1)
            if self.cfg.show_progress and hasattr(bar, "set_postfix"):
                bar.set_postfix(reward=f"{run_r:.3f}", task=f"{run_t:.2f}")

        self.executor.selector = original_selector
        self.credit.cfg.p_ablate = original_p

    def _run_train(self, dataset: list[TrainingExample]) -> None:
        """Phase 2: LinUCB routing with full exploration."""
        original_alpha = self.policy.alpha
        self.policy.alpha = self.cfg.alpha_train

        original_p = self.credit.cfg.p_ablate
        self.credit.cfg.p_ablate = self.cfg.p_ablate_train

        cycle = self._make_cycle(dataset, self.cfg.n_train_episodes)
        bar = _make_bar(cycle, desc="train   ", enabled=self.cfg.show_progress,
                        total=self.cfg.n_train_episodes)
        run_r = run_t = 0.0
        for i, example in enumerate(bar):
            result = self._run_episode(example, phase="train", train_mode=True)
            run_r  = (run_r * i + result.reward.total) / (i + 1)
            run_t  = (run_t * i + result.reward.task)  / (i + 1)
            if self.cfg.show_progress and hasattr(bar, "set_postfix"):
                bar.set_postfix(reward=f"{run_r:.3f}", task=f"{run_t:.2f}")

        self.policy.alpha = original_alpha
        self.credit.cfg.p_ablate = original_p

    def _run_finetune(self, dataset: list[TrainingExample]) -> None:
        """Phase 3: low-exploration exploitation."""
        original_alpha = self.policy.alpha
        self.policy.alpha = self.cfg.alpha_finetune

        original_p = self.credit.cfg.p_ablate
        self.credit.cfg.p_ablate = self.cfg.p_ablate_finetune

        cycle = self._make_cycle(dataset, self.cfg.n_finetune_episodes)
        bar = _make_bar(cycle, desc="finetune", enabled=self.cfg.show_progress,
                        total=self.cfg.n_finetune_episodes)
        run_r = run_t = 0.0
        for i, example in enumerate(bar):
            result = self._run_episode(example, phase="finetune", train_mode=True)
            run_r  = (run_r * i + result.reward.total) / (i + 1)
            run_t  = (run_t * i + result.reward.task)  / (i + 1)
            if self.cfg.show_progress and hasattr(bar, "set_postfix"):
                bar.set_postfix(reward=f"{run_r:.3f}", task=f"{run_t:.2f}")

        self.policy.alpha = original_alpha
        self.credit.cfg.p_ablate = original_p

    # ------------------------------------------------------------------
    # Core episode loop
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        example:    TrainingExample,
        phase:      str,
        train_mode: bool,
    ) -> EpisodeResult:
        # Execute path through graph
        trace = self.executor.execute(example.x, train_mode=train_mode)

        # Compute reward
        reward_components = self.reward.compute(
            trace=trace,
            x=example.x,
            y_star=example.y_star,
            constraints=example.constraints or None,
        )

        # Update policy with total reward signal (skip for no_bandit ablation)
        phi = trace.pre_lm_features.as_vector()
        if self._train_policy:
            self.policy.update_path(
                trace.edges_traversed,
                phi,
                reward_components.total,
            )

        # LOO credit assignment (probabilistic)
        credit_result = self.credit.maybe_assign(
            trace=trace,
            graph=self.executor.graph,
            x=example.x,
            y_star=example.y_star,
            rng=self._rng,
        )

        result = EpisodeResult(
            phase=phase,
            episode=self._episode,
            reward=reward_components,
            credit=credit_result,
            path=trace.node_ids,
            token_count=trace.token_count,
        )

        self._stats.record(result)

        if self.on_episode is not None:
            self.on_episode(self._episode, result)

        self._maybe_checkpoint(phase)
        self._episode += 1

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_cycle(
        self,
        dataset: list[TrainingExample],
        n: int,
    ) -> list[TrainingExample]:
        """
        Return exactly n examples by cycling through dataset with
        per-epoch shuffling.
        """
        if n == 0:
            return []
        shuffled = list(dataset)
        self._py_rng.shuffle(shuffled)
        result: list[TrainingExample] = []
        while len(result) < n:
            result.extend(shuffled)
            self._py_rng.shuffle(shuffled)
        return result[:n]

    def _maybe_checkpoint(self, phase: str) -> None:
        if self.cfg.checkpoint_dir is None:
            return
        if self._episode % self.cfg.checkpoint_every != 0:
            return
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.cfg.checkpoint_dir,
            f"policy_{phase}_ep{self._episode:06d}.npz",
        )
        self.policy.save(path)
