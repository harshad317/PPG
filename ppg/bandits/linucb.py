"""
Edge-factored LinUCB bandit policy for PPG guard learning.

One LinUCBArm per (src, dst) edge in the frozen graph. Arms are pre-created
for all known edges at init time; unknown edges (e.g. escalation edges added
at runtime) are created lazily.

Regret bound
------------
Per arm, under the following assumptions:
  1. Fixed graph topology — no edges added or removed after __init__.
  2. Linear reward model — E[r | phi] = theta* . phi for unknown theta*.
  3. Rewards bounded in [0, 1] and conditionally independent across episodes.
  4. Feature vectors phi are bounded: ||phi||_2 <= 1 (enforced by FeatureExtractor).

Cumulative regret after T episodes:
  R(T) = O(d * sqrt(T * log T))

where d = FEATURE_DIM. This matches the standard LinUCB result of
Chu et al. (2011, Theorem 3) with lambda_reg = 1.

The bound applies per (src, dst) arm independently. When the policy selects
among K edges at each node, the path-level regret accumulates across the
K arm-selection steps that constitute one episode.

Training mode  : UCB score = mean + alpha * uncertainty  (exploration)
Eval mode      : UCB score = mean only                   (greedy)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ppg.core.features import FEATURE_DIM
from ppg.core.graph import PPGraph

if TYPE_CHECKING:
    pass  # PPGraph already imported above; kept for clarity

# Sentinel used as src key when selecting among source nodes (current=None).
_START = "__start__"


# ---------------------------------------------------------------------------
# LinUCBArm
# ---------------------------------------------------------------------------

@dataclass
class LinUCBArm:
    """
    Single arm for one directed edge (src -> dst).

    State:
      A  : (d x d) regularised feature covariance  — init = lambda_reg * I
      b  : (d,) reward-weighted feature accumulator — init = 0

    Estimates:
      mu_hat = A^{-1} b   (not stored explicitly; computed on demand)

    UCB score for feature vector phi:
      A_inv_phi  = solve(A, phi)          # avoids explicit inversion
      mean       = b @ A_inv_phi          # = mu_hat . phi
      uncertainty= sqrt(phi @ A_inv_phi)  # = sqrt(phi^T A^{-1} phi)
      score      = mean + alpha * uncertainty
    """
    feature_dim: int
    lambda_reg:  float = 1.0

    A:         np.ndarray = field(init=False, repr=False)
    b:         np.ndarray = field(init=False, repr=False)
    n_updates: int        = field(init=False, default=0)

    def __post_init__(self):
        self.A = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64)
        self.b = np.zeros(self.feature_dim, dtype=np.float64)

    # ------------------------------------------------------------------

    def score(self, phi: np.ndarray, alpha: float, train_mode: bool) -> float:
        """
        Returns UCB score (train_mode=True) or greedy mean (train_mode=False).

        Uses solve() not inv() for numerical stability.
        Clips uncertainty to 0 before sqrt to handle floating-point negatives.
        """
        A_inv_phi = np.linalg.solve(self.A, phi)
        mean = float(self.b @ A_inv_phi)
        if train_mode and alpha > 0.0:
            uncertainty = float(np.sqrt(max(0.0, float(phi @ A_inv_phi))))
            return mean + alpha * uncertainty
        return mean

    def update(self, phi: np.ndarray, reward: float) -> None:
        """Sherman-Morrison rank-1 update of A, linear update of b."""
        self.A += np.outer(phi, phi)
        self.b += reward * phi
        self.n_updates += 1

    @property
    def mu_hat(self) -> np.ndarray:
        """Current weight estimate A^{-1} b. Computed on demand (diagnostic use)."""
        return np.linalg.solve(self.A, self.b)

    def mean_score(self, phi: np.ndarray) -> float:
        """Greedy mean score (no UCB bonus). Used in eval mode."""
        return self.score(phi, alpha=0.0, train_mode=False)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "A":        self.A,
            "b":        self.b,
            "meta":     np.array([self.n_updates, self.feature_dim, self.lambda_reg]),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "LinUCBArm":
        meta = arrays["meta"]
        arm = cls(feature_dim=int(meta[1]), lambda_reg=float(meta[2]))
        arm.A = arrays["A"].copy()
        arm.b = arrays["b"].copy()
        arm.n_updates = int(meta[0])
        return arm


# ---------------------------------------------------------------------------
# LinUCBPolicy
# ---------------------------------------------------------------------------

class LinUCBPolicy:
    """
    Edge-factored LinUCB policy — implements the NodeSelector protocol.

    One arm per edge. select() scores all active successors of the current
    node and returns the highest-scoring one. update() pushes the received
    reward back to the arm for the edge that was actually traversed.

    Parameters
    ----------
    graph       : frozen PPGraph (topology must not change after init)
    feature_dim : must equal FEATURE_DIM from ppg.core.features
    alpha       : UCB exploration coefficient (0.0 = greedy)
    lambda_reg  : L2 regularisation for A initialisation
    """

    def __init__(
        self,
        graph:       PPGraph,
        feature_dim: int   = FEATURE_DIM,
        alpha:       float = 0.5,
        lambda_reg:  float = 1.0,
    ):
        self.graph       = graph
        self.feature_dim = feature_dim
        self.alpha       = alpha
        self.lambda_reg  = lambda_reg

        # Pre-create one arm for each edge in the graph
        self._arms: dict[tuple[str, str], LinUCBArm] = {
            edge: LinUCBArm(feature_dim, lambda_reg)
            for edge in graph.edges
        }

    # ------------------------------------------------------------------
    # NodeSelector protocol
    # ------------------------------------------------------------------

    def select(
        self,
        current:    Optional[str],
        candidates: list[str],
        phi:        np.ndarray,
        train_mode: bool = True,
    ) -> str:
        """
        Select one node from candidates given feature vector phi.

        current=None means we are choosing among source nodes (start of
        episode); a sentinel key _START is used for those arms.
        """
        if len(candidates) == 1:
            return candidates[0]

        src = current if current is not None else _START
        scores = {
            dst: self._get_or_create(src, dst).score(phi, self.alpha, train_mode)
            for dst in candidates
        }
        return max(candidates, key=lambda d: scores[d])

    def update(
        self,
        edge:   tuple[str, str],
        phi:    np.ndarray,
        reward: float,
    ) -> None:
        """Push reward to the arm for the traversed edge."""
        self._get_or_create(*edge).update(phi, reward)

    # ------------------------------------------------------------------
    # Bulk update (called by trainer after episode ends)
    # ------------------------------------------------------------------

    def update_path(
        self,
        edges_traversed: list[tuple[str, str]],
        phi:             np.ndarray,
        reward:          float,
    ) -> None:
        """
        Update all arms along the path with the same episode reward.
        Assumes the linear reward decomposition: R(path) ≈ sum_e r_e.
        Each edge arm receives the full episode reward as a proxy.
        """
        for edge in edges_traversed:
            self.update(edge, phi, reward)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def total_updates(self) -> int:
        return sum(arm.n_updates for arm in self._arms.values())

    def arm_stats(self) -> dict[str, dict]:
        """
        Returns per-edge stats keyed by human-readable 'src_type->dst_type'.
        Used for logging and path-heatmap visualisation.
        """
        stats: dict[str, dict] = {}
        for (src, dst), arm in self._arms.items():
            src_label = (
                self.graph.nodes[src].type.value
                if src in self.graph.nodes else src
            )
            dst_label = (
                self.graph.nodes[dst].type.value
                if dst in self.graph.nodes else dst
            )
            label = f"{src_label}->{dst_label}"
            stats[label] = {
                "n_updates":   arm.n_updates,
                "mu_hat_norm": float(np.linalg.norm(arm.mu_hat)),
                "A_det_log":   float(np.linalg.slogdet(arm.A)[1]),
            }
        return stats

    def guard_weights_for_edge(self, src: str, dst: str) -> np.ndarray:
        """Returns mu_hat for an edge arm — useful for interpretability."""
        return self._get_or_create(src, dst).mu_hat

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save all arms to a single .npz file.
        Keys: '{src}|{dst}__{A|b|meta}'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for (src, dst), arm in self._arms.items():
            prefix = f"{src}|{dst}"
            for k, v in arm.to_arrays().items():
                arrays[f"{prefix}__{k}"] = v
        np.savez(str(path), **arrays)

    def load(self, path: str | Path) -> None:
        """
        Load arms from a .npz file, merging into existing arms dict.
        Edges not in the file are left at their current (init) state.
        """
        data = np.load(str(path))
        edges_in_file: set[str] = {
            k.rsplit("__", 1)[0] for k in data.files
        }
        for edge_str in edges_in_file:
            src, dst = edge_str.split("|", 1)
            edge = (src, dst)
            arm_arrays = {
                suffix: data[f"{edge_str}__{suffix}"]
                for suffix in ("A", "b", "meta")
                if f"{edge_str}__{suffix}" in data.files
            }
            self._arms[edge] = LinUCBArm.from_arrays(arm_arrays)

    # ------------------------------------------------------------------
    # Guard synchronisation
    # ------------------------------------------------------------------

    def sync_guards(self, graph: PPGraph, threshold: float = 0.0) -> None:
        """
        Write learned arm weights into graph guard predicates.

        After sync, guard(src, dst).evaluate(phi) iff (mu_hat_{src,dst} @ phi >= threshold).
        Call at end of training so eval-time guards are data-driven, not all-pass.
        """
        for (src, dst), arm in self._arms.items():
            if (src, dst) not in graph.edges:
                continue
            guard = graph.edges[(src, dst)]
            guard.weights = arm.mu_hat.copy()
            guard.bias    = threshold

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, src: str, dst: str) -> LinUCBArm:
        key = (src, dst)
        if key not in self._arms:
            self._arms[key] = LinUCBArm(self.feature_dim, self.lambda_reg)
        return self._arms[key]
