"""
Runtime feature extraction for PPG guard evaluation.

Two-stage pipeline:
  Stage 1 (pre-LM): input text -> input_length_norm, embed_cluster_*
  Stage 2 (post-LM): k samples  -> sc_disagreement, entropy_approx

All features land in a fixed-length float vector whose layout is defined
by FEATURE_NAMES — the single source of truth shared with Guard in graph.py.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Feature schema — single source of truth
# ---------------------------------------------------------------------------

N_CLUSTERS: int = 4   # number of input-embedding clusters (one-hot width)

FEATURE_NAMES: list[str] = [
    "input_length_norm",    # tokenized len / max_tokens, clipped to [0, 1]
    "sc_disagreement",      # 1 - freq(most_common_answer) across k samples
    "entropy_approx",       # normalised entropy of k-sample answer distribution
    "verifier_score",       # 0.0 = fail, 1.0 = pass, 0.5 = unknown
    "tool_success",         # 1.0 if last tool call succeeded, else 0.0
    "tool_failure",         # 1.0 if last tool call failed, else 0.0
    "embed_cluster_0",      # one-hot: input cluster id
    "embed_cluster_1",
    "embed_cluster_2",
    "embed_cluster_3",
]
FEATURE_DIM: int = len(FEATURE_NAMES)   # 10

assert FEATURE_DIM == 6 + N_CLUSTERS, "Update FEATURE_NAMES or N_CLUSTERS together"

_FEAT_IDX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# RuntimeFeatures
# ---------------------------------------------------------------------------

@dataclass
class RuntimeFeatures:
    """
    All runtime signals available to Guards during FSM execution.

    Defaults represent a neutral, pre-LM state:
      - no consistency data (sc_disagreement=0, entropy=0)
      - verifier unknown (0.5)
      - no tool call
      - cluster unknown (-1 -> all-zero one-hot)
    """
    input_length_norm: float = 0.0
    sc_disagreement:   float = 0.0
    entropy_approx:    float = 0.0
    verifier_score:    float = 0.5
    tool_success:      float = 0.0
    tool_failure:      float = 0.0
    embed_cluster:     int   = -1      # -1 = unknown; 0..N_CLUSTERS-1 = valid

    def as_vector(self) -> np.ndarray:
        """Returns a float64 array of shape (FEATURE_DIM,) aligned to FEATURE_NAMES."""
        cluster_onehot = np.zeros(N_CLUSTERS, dtype=np.float64)
        if 0 <= self.embed_cluster < N_CLUSTERS:
            cluster_onehot[self.embed_cluster] = 1.0

        return np.array([
            self.input_length_norm,
            self.sc_disagreement,
            self.entropy_approx,
            self.verifier_score,
            self.tool_success,
            self.tool_failure,
            *cluster_onehot,
        ], dtype=np.float64)

    def as_vector_subset(self, feature_names: list[str]) -> np.ndarray:
        """Return only the requested features (used by Guard.evaluate)."""
        full = self.as_vector()
        return np.array([full[_FEAT_IDX[n]] for n in feature_names], dtype=np.float64)

    def with_tool_outcome(self, success: bool) -> "RuntimeFeatures":
        """Return a copy with tool fields set."""
        import dataclasses
        return dataclasses.replace(
            self,
            tool_success=1.0 if success else 0.0,
            tool_failure=0.0 if success else 1.0,
        )

    def with_verifier(self, score: float) -> "RuntimeFeatures":
        import dataclasses
        return dataclasses.replace(self, verifier_score=float(np.clip(score, 0.0, 1.0)))

    def to_dict(self) -> dict:
        return {
            "input_length_norm": self.input_length_norm,
            "sc_disagreement":   self.sc_disagreement,
            "entropy_approx":    self.entropy_approx,
            "verifier_score":    self.verifier_score,
            "tool_success":      self.tool_success,
            "tool_failure":      self.tool_failure,
            "embed_cluster":     self.embed_cluster,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeFeatures":
        return cls(**d)


# ---------------------------------------------------------------------------
# Answer normalisation (pluggable)
# ---------------------------------------------------------------------------

class AnswerNormalizer(Protocol):
    def __call__(self, text: str) -> str: ...


def default_normalizer(text: str) -> str:
    """
    Best-effort normaliser for short-answer tasks (math, MC, QA).
    Priority: extract last number > strip punctuation/case.
    """
    text = text.strip().lower()
    # extract last standalone number (handles "The answer is 42.")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    # fallback: strip non-alphanumeric boundaries
    return re.sub(r"[^a-z0-9\s]", "", text).strip()


def verbatim_normalizer(text: str) -> str:
    return text.strip().lower()


# ---------------------------------------------------------------------------
# Consistency / entropy helpers
# ---------------------------------------------------------------------------

def _consistency_features(
    samples: list[str],
    normalizer: AnswerNormalizer = default_normalizer,
) -> tuple[float, float]:
    """
    Returns (sc_disagreement, entropy_approx) from k answer samples.
    Both values are in [0, 1].
    """
    if len(samples) <= 1:
        return 0.0, 0.0

    normalized = [normalizer(s) for s in samples]
    counter = Counter(normalized)
    total = len(normalized)

    max_count = counter.most_common(1)[0][1]
    sc_disagreement = 1.0 - max_count / total

    probs = np.array([c / total for c in counter.values()], dtype=np.float64)
    raw_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    # normalise by log(k) so entropy is in [0, 1]
    max_entropy = float(np.log(total))
    entropy_norm = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    return sc_disagreement, float(np.clip(entropy_norm, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Cluster model protocol
# ---------------------------------------------------------------------------

class ClusterModel(Protocol):
    def predict(self, embeddings: np.ndarray) -> np.ndarray: ...


class _HashCluster:
    """
    Deterministic pseudo-cluster that requires no embeddings.
    Assigns cluster by hash(text) % N_CLUSTERS.
    Used when no real cluster model is provided.
    """
    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters

    def predict(self, texts: list[str]) -> list[int]:
        return [
            int(hashlib.sha256(t.encode()).hexdigest(), 16) % self.n_clusters
            for t in texts
        ]


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

@dataclass
class FeatureExtractor:
    """
    Converts raw (input, samples) into RuntimeFeatures.

    Parameters
    ----------
    max_input_tokens : int
        Denominator for input_length_norm. Typically the model's context window.
    tokenizer : callable, optional
        text -> int (token count). If None, uses whitespace split as proxy.
    cluster_model : ClusterModel, optional
        If None, uses hash-based pseudo-clustering (no embeddings needed).
    embedder : callable, optional
        text -> np.ndarray. Required if cluster_model is a real sklearn model.
    normalizer : AnswerNormalizer
        Used to compute sc_disagreement and entropy.
    """
    max_input_tokens: int = 4096
    tokenizer:        Optional[Callable[[str], int]] = None
    cluster_model:    Optional[ClusterModel]         = None
    embedder:         Optional[Callable[[str], np.ndarray]] = None
    normalizer:       AnswerNormalizer = field(default=default_normalizer)

    def __post_init__(self):
        if self.cluster_model is None:
            self._clusterer = _HashCluster()
        else:
            self._clusterer = self.cluster_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pre_lm(self, x: str) -> RuntimeFeatures:
        """
        Features computable before calling the LM.
        Fills: input_length_norm, embed_cluster.
        All post-LM fields remain at their neutral defaults.
        """
        return RuntimeFeatures(
            input_length_norm=self._length_norm(x),
            embed_cluster=self._cluster(x),
        )

    def post_lm(
        self,
        x: str,
        samples: list[str],
        verifier_score: Optional[float] = None,
        tool_success:   Optional[bool]  = None,
    ) -> RuntimeFeatures:
        """
        Full feature vector after receiving k LM samples.

        Parameters
        ----------
        x              : original input text
        samples        : list of k LM response strings
        verifier_score : float in [0,1] from external verifier, or None
        tool_success   : bool from tool execution, or None
        """
        sc_dis, entropy = _consistency_features(samples, self.normalizer)

        feat = RuntimeFeatures(
            input_length_norm=self._length_norm(x),
            sc_disagreement=sc_dis,
            entropy_approx=entropy,
            verifier_score=float(np.clip(verifier_score, 0.0, 1.0))
                           if verifier_score is not None else 0.5,
            tool_success=1.0 if tool_success is True  else 0.0,
            tool_failure=1.0 if tool_success is False else 0.0,
            embed_cluster=self._cluster(x),
        )
        return feat

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _length_norm(self, text: str) -> float:
        if self.tokenizer is not None:
            tokens = self.tokenizer(text)
        else:
            tokens = len(text.split())     # whitespace proxy
        return float(np.clip(tokens / self.max_input_tokens, 0.0, 1.0))

    def _cluster(self, text: str) -> int:
        if isinstance(self._clusterer, _HashCluster):
            return self._clusterer.predict([text])[0]
        # Real cluster model: needs embedding
        if self.embedder is None:
            return 0
        emb = self.embedder(text)
        if emb.ndim == 1:
            emb = emb[None, :]
        return int(self._clusterer.predict(emb)[0])
