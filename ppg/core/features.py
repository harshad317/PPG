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
import threading
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional, Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Feature schema — single source of truth
# ---------------------------------------------------------------------------

N_CLUSTERS: int = 4   # number of input-embedding clusters (one-hot width)

FEATURE_NAMES: list[str] = [
    "input_length_norm",        # tokenized len / max_tokens, clipped to [0, 1]
    "sc_disagreement",          # 1 - freq(most_common_answer) across k samples
    "entropy_approx",           # normalised entropy of k-sample answer distribution
    "verifier_score",           # 0.0 = fail, 1.0 = pass, 0.5 = unknown
    "tool_success",             # 1.0 if last tool call succeeded, else 0.0
    "tool_failure",             # 1.0 if last tool call failed, else 0.0
    # Constraint-type indicators (pre-LM, keyword-based)
    # Gives LinUCB routing signal for instruction-following benchmarks.
    "has_length_constraint",    # 1.0 if input mentions word/sentence/char count or brevity
    "has_format_constraint",    # 1.0 if input mentions bullet/list/json/markdown/header
    "has_keyword_constraint",   # 1.0 if input asks to include/exclude specific words
    "n_constraints_norm",       # normalised count of constraint signals in input
    # Math/reasoning indicators (pre-LM, keyword-based)
    # Gives LinUCB routing signal for math and multi-step reasoning benchmarks.
    "has_numeric_input",        # 1.0 if input contains numbers
    "n_arithmetic_ops_norm",    # normalised count of arithmetic operators/keywords
    "n_steps_heuristic_norm",   # sentence count / 10, proxy for reasoning depth
    "input_word_count_norm",    # word count / 500, proxy for input complexity
    # Domain indicators (pre-LM, pattern-based)
    # Gives LinUCB signal for MCQ, code-generation, and adversarial benchmarks.
    "is_multiple_choice",       # 1.0 if input has A./B./C./D. answer options
    "is_code_task",             # 1.0 if input mentions def/function/assert/python
    "has_adversarial_framing",  # 1.0 if input has trick-question/misconception cues
    "embed_cluster_0",          # one-hot: input cluster id
    "embed_cluster_1",
    "embed_cluster_2",
    "embed_cluster_3",
]
FEATURE_DIM: int = len(FEATURE_NAMES)   # 18

assert FEATURE_DIM == 17 + N_CLUSTERS, "Update FEATURE_NAMES or N_CLUSTERS together"

# ---------------------------------------------------------------------------
# Constraint-type detection (keyword-based, used in extract_pre_lm)
# ---------------------------------------------------------------------------

_LENGTH_PATTERNS = re.compile(
    r'\b(word[s]?\s+count|word[s]?|sentence[s]?|character[s]?|char[s]?|'
    r'brief|concise|short|succinct|limit|at\s+least|at\s+most|no\s+more\s+than|'
    r'fewer\s+than|more\s+than|exactly\s+\d+|within\s+\d+)\b',
    re.IGNORECASE,
)
_FORMAT_PATTERNS = re.compile(
    r'\b(bullet[s]?|numbered\s+list|json|markdown|header[s]?|bold|italic|'
    r'table|paragraph[s]?|section[s]?|format|structure|indent|newline|'
    r'html|xml|csv|code\s+block)\b',
    re.IGNORECASE,
)
_KEYWORD_PATTERNS = re.compile(
    r'\b(include|mention|use\s+the\s+(word|phrase)|contain|must\s+have|'
    r'do\s+not\s+(use|mention|include)|avoid|exclude|without\s+using|'
    r'keyword[s]?|phrase[s]?)\b',
    re.IGNORECASE,
)
_CONSTRAINT_SPLIT = re.compile(r'[;,\.\n]+')

_NUMERIC_PATTERN = re.compile(r'\d+')
_ARITHMETIC_OPS = re.compile(
    r'\b(plus|minus|times|divided|multiply|add|subtract|sum|difference|product|'
    r'quotient|percent|percentage|fraction|ratio|half|double|triple|twice|'
    r'square|cube|total|each|per|every|cost|price|pay|earn|spend|save|'
    r'more than|less than|how many|how much|remainder|average|mean)\b|'
    r'[+\-*/÷×%$]',
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r'[.!?]+\s+')

_MCQ_PATTERN = re.compile(
    r'(?:^|\n)\s*[A-Da-d][.)]\s',
    re.MULTILINE,
)
_CODE_PATTERN = re.compile(
    r'\b(def |function|assert |return |import |class |print\s*\(|'
    r'write a (?:python\s+)?function|implement a?\s*function|algorithm|'
    r'python|program(?:ming)?|code|test case|pass(?:es)? (?:the |all )?\s*test)\b',
    re.IGNORECASE,
)
_ADVERSARIAL_PATTERN = re.compile(
    r'\b(common(?:ly)?\s+(?:believe|thought|misconception|myth|assumption)|'
    r'actually|in fact|contrary to|false premise|trick question|'
    r'popular(?:ly)?\s+believed|many people (?:think|believe)|'
    r'is it true|true or false|fact or (?:fiction|myth)|'
    r'do you believe|does\s+\w+\s+really|really true)\b',
    re.IGNORECASE,
)

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
    input_length_norm:      float = 0.0
    sc_disagreement:        float = 0.0
    entropy_approx:         float = 0.0
    verifier_score:         float = 0.5
    tool_success:           float = 0.0
    tool_failure:           float = 0.0
    has_length_constraint:  float = 0.0
    has_format_constraint:  float = 0.0
    has_keyword_constraint: float = 0.0
    n_constraints_norm:     float = 0.0
    has_numeric_input:      float = 0.0
    n_arithmetic_ops_norm:  float = 0.0
    n_steps_heuristic_norm: float = 0.0
    input_word_count_norm:  float = 0.0
    is_multiple_choice:     float = 0.0
    is_code_task:           float = 0.0
    has_adversarial_framing:float = 0.0
    embed_cluster:          int   = -1      # -1 = unknown; 0..N_CLUSTERS-1 = valid

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
            self.has_length_constraint,
            self.has_format_constraint,
            self.has_keyword_constraint,
            self.n_constraints_norm,
            self.has_numeric_input,
            self.n_arithmetic_ops_norm,
            self.n_steps_heuristic_norm,
            self.input_word_count_norm,
            self.is_multiple_choice,
            self.is_code_task,
            self.has_adversarial_framing,
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
            "has_length_constraint": self.has_length_constraint,
            "has_format_constraint": self.has_format_constraint,
            "has_keyword_constraint": self.has_keyword_constraint,
            "n_constraints_norm": self.n_constraints_norm,
            "has_numeric_input": self.has_numeric_input,
            "n_arithmetic_ops_norm": self.n_arithmetic_ops_norm,
            "n_steps_heuristic_norm": self.n_steps_heuristic_norm,
            "input_word_count_norm": self.input_word_count_norm,
            "is_multiple_choice": self.is_multiple_choice,
            "is_code_task": self.is_code_task,
            "has_adversarial_framing": self.has_adversarial_framing,
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
    numeric = _extract_numeric_answer(text)
    if numeric is not None:
        return numeric
    # fallback: strip non-alphanumeric boundaries
    return re.sub(r"[^a-z0-9\s]", "", text).strip()


def _extract_numeric_answer(text: str) -> str | None:
    """Canonicalize common final-answer forms for self-consistency voting."""
    text = text.replace(",", "").replace("−", "-")
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        boxed_value = _extract_numeric_answer(boxed[-1])
        if boxed_value is not None:
            return boxed_value

    spans = re.findall(
        r"(?:####|final answer(?: is|:)?|answer(?: is|:)?)\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    search_text = spans[-1] if spans else text
    nums = re.findall(r"-?\d+\s*/\s*-?\d+|-?\d+(?:\.\d+)?", search_text)
    if not nums and spans:
        nums = re.findall(r"-?\d+\s*/\s*-?\d+|-?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    try:
        value = Fraction(nums[-1].replace(" ", ""))
        if value.denominator == 1:
            return str(value.numerator)
        return f"{float(value):.12g}"
    except (ValueError, ZeroDivisionError):
        return None


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


class SentenceTransformerCluster:
    """
    Embedding-based clustering using sentence-transformers + online KMeans.

    Lazy-loads the sentence-transformer model on first predict() call.
    Falls back to _HashCluster if sentence-transformers is not installed.

    After accumulating >= min_fit_samples embeddings, fits a MiniBatchKMeans
    and switches from hash-based to real cluster assignments.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        n_clusters: int = N_CLUSTERS,
        min_fit_samples: int = 50,
    ):
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.min_fit_samples = min_fit_samples
        self._model = None
        self._kmeans = None
        self._fallback = _HashCluster(n_clusters)
        self._embedding_buffer: list[np.ndarray] = []
        self._fitted = False
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                self._model = None

    def predict(self, texts: list[str]) -> list[int]:
        self._ensure_model()
        if self._model is None:
            return self._fallback.predict(texts)

        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]

        with self._lock:
            if not self._fitted:
                for emb in embeddings:
                    self._embedding_buffer.append(emb)
                if len(self._embedding_buffer) >= self.min_fit_samples:
                    self._fit()

            if self._fitted:
                return [int(c) for c in self._kmeans.predict(embeddings)]
        return self._fallback.predict(texts)

    def _fit(self):
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            return
        X = np.stack(self._embedding_buffer)
        n = min(self.n_clusters, len(X))
        kmeans = MiniBatchKMeans(n_clusters=n, random_state=0, n_init=3)
        kmeans.fit(X)
        self._kmeans = kmeans
        self._fitted = True
        self._embedding_buffer.clear()


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

    use_semantic_clusters: bool = False

    def __post_init__(self):
        if self.cluster_model is not None:
            self._clusterer = self.cluster_model
        elif self.use_semantic_clusters:
            self._clusterer = SentenceTransformerCluster()
        else:
            self._clusterer = _HashCluster()

    @classmethod
    def production(cls, **overrides) -> "FeatureExtractor":
        """Production config with real sentence-transformer embeddings."""
        defaults = dict(use_semantic_clusters=True)
        defaults.update(overrides)
        return cls(**defaults)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pre_lm(self, x: str) -> RuntimeFeatures:
        """
        Features computable before calling the LM.
        Fills: input_length_norm, constraint-type indicators, math/reasoning
        indicators, embed_cluster. All post-LM fields remain at neutral defaults.
        """
        n_arith = len(_ARITHMETIC_OPS.findall(x))
        n_sentences = len([s for s in _SENTENCE_SPLIT.split(x) if s.strip()])
        n_words = len(x.split())
        return RuntimeFeatures(
            input_length_norm=self._length_norm(x),
            has_length_constraint=float(bool(_LENGTH_PATTERNS.search(x))),
            has_format_constraint=float(bool(_FORMAT_PATTERNS.search(x))),
            has_keyword_constraint=float(bool(_KEYWORD_PATTERNS.search(x))),
            n_constraints_norm=min(1.0, len(_CONSTRAINT_SPLIT.split(x)) / 20.0),
            has_numeric_input=float(bool(_NUMERIC_PATTERN.search(x))),
            n_arithmetic_ops_norm=min(1.0, n_arith / 10.0),
            n_steps_heuristic_norm=min(1.0, n_sentences / 10.0),
            input_word_count_norm=min(1.0, n_words / 500.0),
            is_multiple_choice=float(bool(_MCQ_PATTERN.search(x))),
            is_code_task=float(bool(_CODE_PATTERN.search(x))),
            has_adversarial_framing=float(bool(_ADVERSARIAL_PATTERN.search(x))),
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

        n_arith = len(_ARITHMETIC_OPS.findall(x))
        n_sentences = len([s for s in _SENTENCE_SPLIT.split(x) if s.strip()])
        n_words = len(x.split())

        feat = RuntimeFeatures(
            input_length_norm=self._length_norm(x),
            sc_disagreement=sc_dis,
            entropy_approx=entropy,
            verifier_score=float(np.clip(verifier_score, 0.0, 1.0))
                           if verifier_score is not None else 0.5,
            tool_success=1.0 if tool_success is True  else 0.0,
            tool_failure=1.0 if tool_success is False else 0.0,
            has_length_constraint=float(bool(_LENGTH_PATTERNS.search(x))),
            has_format_constraint=float(bool(_FORMAT_PATTERNS.search(x))),
            has_keyword_constraint=float(bool(_KEYWORD_PATTERNS.search(x))),
            n_constraints_norm=min(1.0, len(_CONSTRAINT_SPLIT.split(x)) / 20.0),
            has_numeric_input=float(bool(_NUMERIC_PATTERN.search(x))),
            n_arithmetic_ops_norm=min(1.0, n_arith / 10.0),
            n_steps_heuristic_norm=min(1.0, n_sentences / 10.0),
            input_word_count_norm=min(1.0, n_words / 500.0),
            is_multiple_choice=float(bool(_MCQ_PATTERN.search(x))),
            is_code_task=float(bool(_CODE_PATTERN.search(x))),
            has_adversarial_framing=float(bool(_ADVERSARIAL_PATTERN.search(x))),
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
            from ppg.core.tokenizer import count_tokens
            tokens = count_tokens(text)
        return float(np.clip(tokens / self.max_input_tokens, 0.0, 1.0))

    def _cluster(self, text: str) -> int:
        if isinstance(self._clusterer, (_HashCluster, SentenceTransformerCluster)):
            return self._clusterer.predict([text])[0]
        # External cluster model: needs embedding
        if self.embedder is None:
            return 0
        emb = self.embedder(text)
        if emb.ndim == 1:
            emb = emb[None, :]
        return int(self._clusterer.predict(emb)[0])
