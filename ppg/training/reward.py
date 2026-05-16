"""
Multi-objective reward computation for PPG training.

Formula:
    R = r_task + λ_if·r_constraint − λ_c·tokens − λ_v·Var[r_task | perturb(x)]

Components
----------
r_task       : task accuracy metric score in [0, 1]
r_constraint : instruction-following constraint satisfaction in [0, 1]
               (0.0 when no ConstraintChecker is provided)
-λ_c·tokens  : token cost penalty (tokens normalised by max_tokens_ref)
-λ_v·Var     : perturbation variance penalty over m perturbed inputs

Variance is estimated cheaply: m=2 perturbations per episode, with results
cached in PerturbationBuffer so repeated inputs share perturbation work.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ppg.core.executor import LMClient, PathTrace, PromptAssembler


# ---------------------------------------------------------------------------
# TaskMetric protocol + built-ins
# ---------------------------------------------------------------------------

@runtime_checkable
class TaskMetric(Protocol):
    def score(self, prediction: str, reference: str) -> float: ...


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _extract_number(text: str) -> str:
    """Return last number found in text, or full normalised text if none."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else _normalize(text)


class ExactMatchMetric:
    """1.0 if normalised strings match, else 0.0. Default for most benchmarks."""

    def score(self, prediction: str, reference: str) -> float:
        return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


class NumericExactMatchMetric:
    """
    Extracts the last number from both prediction and reference before comparing.
    Best for GSM8K and math benchmarks.
    """

    def score(self, prediction: str, reference: str) -> float:
        return 1.0 if _extract_number(prediction) == _extract_number(reference) else 0.0


class F1Metric:
    """
    Token-overlap F1. Used for HotpotQA, DROP (span extraction), TruthfulQA.
    Returns float in [0, 1].
    """

    def score(self, prediction: str, reference: str) -> float:
        pred_tokens = _normalize(prediction).split()
        ref_tokens  = _normalize(reference).split()
        if not pred_tokens or not ref_tokens:
            return 1.0 if pred_tokens == ref_tokens else 0.0
        pred_count = Counter(pred_tokens)
        ref_count  = Counter(ref_tokens)
        common = sum((pred_count & ref_count).values())
        if common == 0:
            return 0.0
        precision = common / len(pred_tokens)
        recall    = common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)


class SubstringMatchMetric:
    """1.0 if normalised reference is a substring of normalised prediction."""

    def score(self, prediction: str, reference: str) -> float:
        return 1.0 if _normalize(reference) in _normalize(prediction) else 0.0


# Registry for easy lookup by name
METRIC_REGISTRY: dict[str, TaskMetric] = {
    "exact_match":         ExactMatchMetric(),
    "numeric_exact_match": NumericExactMatchMetric(),
    "f1":                  F1Metric(),
    "substring":           SubstringMatchMetric(),
}


# ---------------------------------------------------------------------------
# ConstraintChecker protocol + built-in
# ---------------------------------------------------------------------------

@runtime_checkable
class ConstraintChecker(Protocol):
    def check(self, response: str, constraints: list[str], metadata: dict | None = None) -> float: ...


class KeywordConstraintChecker:
    """
    Checks that each constraint keyword appears in the response.
    Returns fraction of constraints satisfied in [0, 1].

    Used as a lightweight proxy for IFBench/IFEval instruction following.
    For training: pass as constraint_checker= to AblationStudy/RewardComputer.
    For eval:     the harness dispatches to check() when example.constraints
                  is non-empty, regardless of metric type.

    Do NOT use as a TaskMetric (metric= arg). Use ExactMatchMetric() or
    F1Metric() for r_task; use this class for r_constraint via constraint_checker=.
    """

    def check(self, response: str, constraints: list[str], metadata: dict | None = None) -> float:
        if not constraints:
            return 1.0
        resp_lower = response.lower()
        satisfied = sum(1 for c in constraints if c.lower() in resp_lower)
        return satisfied / len(constraints)

    def score(self, prediction: str, reference: str) -> float:
        raise TypeError(
            "KeywordConstraintChecker cannot be used as a TaskMetric. "
            "Pass it as constraint_checker= to AblationStudy or RewardComputer. "
            "Use ExactMatchMetric() or F1Metric() for the metric= argument."
        )


class IFEvalOfficialChecker:
    """
    Constraint checker for IFEval using the official Google verifier suite.

    When the EvalExample metadata contains 'instruction_id_list' and 'kwargs'
    (stored by IFEvalLoader), uses the official instruction_following_eval
    library for format/length/regex/JSON/keyword checks.

    Falls back to KeywordConstraintChecker when the library is not installed
    or metadata is absent.

    Install: pip install instruction-following-eval
    Source : https://github.com/google-research/google-research/tree/master/instruction_following_eval
    """

    def check(self, response: str, constraints: list[str], metadata: dict | None = None) -> float:
        if metadata and metadata.get("instruction_id_list"):
            result = self._official_check(response, metadata)
            if result is not None:
                return result
        return KeywordConstraintChecker().check(response, constraints)

    def _official_check(self, response: str, metadata: dict) -> float | None:
        try:
            from instruction_following_eval import instructions_registry  # type: ignore[import]
        except ImportError:
            return None

        instruction_id_list = metadata["instruction_id_list"]
        kwargs_list = metadata.get("kwargs") or [{}] * len(instruction_id_list)

        satisfied = 0
        total = 0
        for instr_id, kwargs in zip(instruction_id_list, kwargs_list):
            if instr_id not in instructions_registry.INSTRUCTION_DICT:
                continue
            try:
                instr = instructions_registry.INSTRUCTION_DICT[instr_id](instr_id)
                instr.build_description(**(kwargs or {}))
                if instr.check_following(response):
                    satisfied += 1
                total += 1
            except Exception:
                pass
        return satisfied / total if total > 0 else 1.0


class IFBenchConstraintChecker:
    """
    Constraint checker for IFBench with type-dispatched rule-based verification.

    When EvalExample metadata contains 'constraint_objects' (stored by
    IFBenchLoader), dispatches each constraint to a type-specific checker:
      - Length   : word/sentence count parsing and comparison
      - Format   : regex patterns for bullets, headers, JSON, numbered lists
      - Keywords : exact case-insensitive keyword matching
      - Style    : heuristic checks; fallback to keyword matching
      - (other)  : keyword substring match on constraint text

    Falls back to KeywordConstraintChecker when metadata is absent.
    """

    _BULLET_RE   = re.compile(r"^\s*[-*•]\s", re.MULTILINE)
    _NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
    _HEADER_RE   = re.compile(r"^#{1,6}\s", re.MULTILINE)
    _JSON_RE     = re.compile(r"^\s*[\[{]", re.DOTALL)

    def check(self, response: str, constraints: list[str], metadata: dict | None = None) -> float:
        objs = (metadata or {}).get("constraint_objects")
        if objs:
            return self._typed_check(response, objs)
        return KeywordConstraintChecker().check(response, constraints)

    def _typed_check(self, response: str, objs: list[dict]) -> float:
        if not objs:
            return 1.0
        satisfied = sum(self._check_one(response, o) for o in objs)
        return satisfied / len(objs)

    def _check_one(self, response: str, obj: dict) -> bool:
        ctype = (obj.get("constraint_type") or "").lower()
        ctext = (obj.get("constraint") or "").lower()

        if ctype == "length":
            return self._check_length(response, ctext)
        if ctype == "format":
            return self._check_format(response, ctext)
        if ctype == "keywords":
            return self._check_keywords(response, ctext)
        # Style and unknown: check if key content words from constraint appear in response
        return self._check_content_words(response, ctext)

    def _check_length(self, response: str, ctext: str) -> bool:
        import re as _re
        words = len(response.split())
        # parse "fewer than N words", "at least N words", "exactly N words", etc.
        m = _re.search(r"(fewer than|less than|under|at most)\s+(\d+)\s+word", ctext)
        if m:
            return words < int(m.group(2))
        m = _re.search(r"(more than|at least|over)\s+(\d+)\s+word", ctext)
        if m:
            return words > int(m.group(2))
        m = _re.search(r"(exactly|around|about)\s+(\d+)\s+word", ctext)
        if m:
            n = int(m.group(2))
            return abs(words - n) <= max(3, int(n * 0.1))
        m = _re.search(r"(\d+)\s*(to|-)\s*(\d+)\s+word", ctext)
        if m:
            return int(m.group(1)) <= words <= int(m.group(3))
        # Sentence count fallback
        sentences = len(_re.split(r"[.!?]+", response.strip()))
        m = _re.search(r"(\d+)\s+sentence", ctext)
        if m:
            return abs(sentences - int(m.group(1))) <= 1
        return ctext in response.lower()

    def _check_format(self, response: str, ctext: str) -> bool:
        if "bullet" in ctext or "bulleted" in ctext:
            return bool(self._BULLET_RE.search(response))
        if "numbered list" in ctext or "ordered list" in ctext:
            return bool(self._NUMBERED_RE.search(response))
        if "header" in ctext or "heading" in ctext:
            return bool(self._HEADER_RE.search(response))
        if "json" in ctext:
            return bool(self._JSON_RE.match(response.strip()))
        if "paragraph" in ctext:
            return len([p for p in response.split("\n\n") if p.strip()]) >= 2
        return ctext in response.lower()

    def _check_keywords(self, response: str, ctext: str) -> bool:
        resp_lower = response.lower()
        # "include the word X" / "must contain X"
        import re as _re
        m = _re.search(r'(?:include|contain|use|mention)\s+(?:the\s+)?(?:word\s+)?["\']?(\w+)["\']?', ctext)
        if m:
            return m.group(1).lower() in resp_lower
        # "do not use X" / "avoid X"
        m = _re.search(r'(?:do not|avoid|without|no)\s+(?:use\s+)?(?:the\s+)?(?:word\s+)?["\']?(\w+)["\']?', ctext)
        if m:
            return m.group(1).lower() not in resp_lower
        return ctext in resp_lower

    def _check_content_words(self, response: str, ctext: str) -> bool:
        """Check if key content words from a constraint appear in the response.

        Extracts words longer than 4 chars, skips common stop words, then checks
        whether at least one content word (via 5-char prefix) appears in the response.
        Covers style/tone constraints like 'tone of excitement and enthusiasm'.
        """
        _STOP = frozenset({
            "about", "after", "along", "also", "although", "another", "because",
            "before", "being", "between", "both", "could", "does", "during",
            "each", "either", "ensure", "every", "follow", "format", "from",
            "give", "have", "into", "just", "keep", "make", "more", "must",
            "need", "only", "other", "over", "part", "please", "provide",
            "response", "should", "since", "some", "such", "than", "that",
            "their", "them", "then", "there", "these", "they", "this", "those",
            "through", "under", "using", "very", "well", "when", "where",
            "which", "while", "will", "with", "within", "without", "write",
            "your",
        })
        words = [w for w in re.findall(r"\w+", ctext.lower())
                 if len(w) > 4 and w not in _STOP]
        if not words:
            return ctext in response.lower()
        resp_lower = response.lower()
        return any(w[:5] in resp_lower for w in words)


# ---------------------------------------------------------------------------
# Perturbators
# ---------------------------------------------------------------------------

class Perturbator(Protocol):
    def perturb(self, text: str, n: int, rng: np.random.Generator) -> list[str]: ...


class WordShufflePerturbator:
    """
    Shuffles non-stop words in the input.
    Simple, local, reproducible via rng. No external dependencies.
    """
    _STOP = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "that", "this", "it", "its", "and", "or", "but", "if", "not",
    })

    def perturb(self, text: str, n: int, rng: np.random.Generator) -> list[str]:
        words = text.split()
        if len(words) <= 2:
            return [text] * n
        non_stop_idx = [i for i, w in enumerate(words)
                        if w.lower().strip(string.punctuation) not in self._STOP]
        results = []
        for _ in range(n):
            words_copy = words.copy()
            if len(non_stop_idx) >= 2:
                i, j = rng.choice(non_stop_idx, size=2, replace=False)
                words_copy[i], words_copy[j] = words_copy[j], words_copy[i]
            results.append(" ".join(words_copy))
        return results


class TruncationPerturbator:
    """
    Truncates input to a random fraction of original length (50-85%).
    Tests robustness to partial inputs.
    """

    def __init__(self, min_frac: float = 0.5, max_frac: float = 0.85):
        self.min_frac = min_frac
        self.max_frac = max_frac

    def perturb(self, text: str, n: int, rng: np.random.Generator) -> list[str]:
        words = text.split()
        results = []
        for _ in range(n):
            frac = rng.uniform(self.min_frac, self.max_frac)
            k = max(1, int(len(words) * frac))
            results.append(" ".join(words[:k]))
        return results


class CompositePerturbator:
    """Applies each child perturbator in round-robin order."""

    def __init__(self, perturbators: list[Perturbator]):
        self._children = perturbators

    def perturb(self, text: str, n: int, rng: np.random.Generator) -> list[str]:
        results = []
        for i in range(n):
            child = self._children[i % len(self._children)]
            results.extend(child.perturb(text, 1, rng))
        return results[:n]


def default_perturbator() -> CompositePerturbator:
    return CompositePerturbator([WordShufflePerturbator(), TruncationPerturbator()])


# ---------------------------------------------------------------------------
# PerturbationBuffer
# ---------------------------------------------------------------------------

class PerturbationBuffer:
    """
    Caches perturbed inputs per original input to avoid regenerating
    perturbations across episodes with the same x.

    Keys: original input string.
    Values: list of perturbed variants.
    """

    def __init__(
        self,
        perturbator: Optional[Perturbator] = None,
        m:           int = 2,
        max_size:    int = 10_000,
        seed:        int = 42,
    ):
        self._perturbator = perturbator or default_perturbator()
        self.m            = m
        self.max_size     = max_size
        self._rng         = np.random.default_rng(seed)
        self._cache:      dict[str, list[str]] = {}
        self._insert_order: list[str] = []   # for LRU-style eviction

    def get(self, x: str, m: Optional[int] = None) -> list[str]:
        """
        Return m perturbed variants of x.
        Generates and caches on first call; extends cache if m > cached count.
        """
        m = m if m is not None else self.m

        if x not in self._cache:
            self._evict_if_full()
            self._cache[x] = self._perturbator.perturb(x, m, self._rng)
            self._insert_order.append(x)
        elif len(self._cache[x]) < m:
            extra = self._perturbator.perturb(x, m - len(self._cache[x]), self._rng)
            self._cache[x].extend(extra)

        return self._cache[x][:m]

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._insert_order.clear()

    def _evict_if_full(self) -> None:
        while len(self._cache) >= self.max_size and self._insert_order:
            oldest = self._insert_order.pop(0)
            self._cache.pop(oldest, None)


# ---------------------------------------------------------------------------
# RewardComponents + RewardConfig
# ---------------------------------------------------------------------------

@dataclass
class RewardComponents:
    task:       float   # r_task ∈ [0, 1]
    constraint: float   # r_constraint ∈ [0, 1]; 0.0 if no checker
    cost:       float   # ≤ 0: -λ_c * normalised_tokens
    variance:   float   # ≤ 0: -λ_v * Var[r_task | perturb(x)]
    total:      float   # task + λ_if*constraint + cost + variance

    def as_dict(self) -> dict[str, float]:
        return {
            "r_task":       self.task,
            "r_constraint": self.constraint,
            "r_cost":       self.cost,
            "r_variance":   self.variance,
            "r_total":      self.total,
        }


@dataclass
class RewardConfig:
    lambda_cost:          float = 0.001   # token cost coefficient
    lambda_variance:      float = 0.1     # perturbation variance coefficient
    lambda_constraint:    float = 0.2     # constraint satisfaction coefficient
    m_perturbation:       int   = 2       # perturbations per reward call
    max_tokens_ref:       int   = 500     # token count upper bound for normalisation
    skip_variance:        bool  = False   # set True for fast eval (no perturb calls)
    constraint_as_task:   bool  = False   # use constraint score as r_task (IFEval/IFBench)
                                          # when True: r_task = checker.check(); r_constraint=0


# ---------------------------------------------------------------------------
# RewardComputer
# ---------------------------------------------------------------------------

class RewardComputer:
    """
    Computes multi-objective reward for one PPG episode.

    Parameters
    ----------
    task_metric         : TaskMetric used for r_task and variance estimation
    lm                  : LMClient — called for each perturbation evaluation
    assembler           : PromptAssembler — builds perturbed prompts from path
    constraint_checker  : optional; if None, r_constraint = 0.0
    perturb_buffer      : optional; if None, a default one is created
    config              : RewardConfig with lambda coefficients
    """

    def __init__(
        self,
        task_metric:        TaskMetric,
        lm:                 LMClient,
        assembler:          PromptAssembler,
        constraint_checker: Optional[ConstraintChecker] = None,
        perturb_buffer:     Optional[PerturbationBuffer] = None,
        config:             Optional[RewardConfig] = None,
    ):
        self.metric     = task_metric
        self.lm         = lm
        self.assembler  = assembler
        self.checker    = constraint_checker
        self.buffer     = perturb_buffer or PerturbationBuffer(m=2)
        self.cfg        = config or RewardConfig()

    def compute(
        self,
        trace:       PathTrace,
        x:           str,
        y_star:      str,
        constraints: Optional[list[str]] = None,
        metadata:    Optional[dict]      = None,
    ) -> RewardComponents:
        """
        Compute all reward components for one episode.

        trace       : PathTrace returned by PPGExecutor.execute()
        x           : original input (used for perturbations)
        y_star      : ground-truth reference answer
        constraints : list of constraint strings for r_constraint (IFBench)
        metadata    : example metadata dict passed to constraint checker
        """
        if self.cfg.constraint_as_task and self.checker is not None:
            # Constraint-only benchmarks (IFEval, IFBench): constraint satisfaction
            # IS the task signal. r_task = checker score; r_constraint suppressed
            # to avoid double-counting in the total reward.
            r_task       = self.checker.check(trace.lm_response, constraints or [], metadata)
            r_constraint = 0.0
        else:
            r_task = self.metric.score(trace.lm_response, y_star)
            r_constraint = (
                self.checker.check(trace.lm_response, constraints or [], metadata)
                if self.checker is not None else 0.0
            )

        r_cost = self._cost(trace.token_count)

        r_var = (
            0.0 if self.cfg.skip_variance
            else self._variance(trace, x, y_star)
        )

        total = (
            r_task
            + self.cfg.lambda_constraint * r_constraint
            + r_cost
            + r_var
        )

        return RewardComponents(
            task=r_task,
            constraint=r_constraint,
            cost=r_cost,
            variance=r_var,
            total=total,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cost(self, token_count: int) -> float:
        normalised = token_count / max(1, self.cfg.max_tokens_ref)
        return -self.cfg.lambda_cost * normalised

    def _variance(self, trace: PathTrace, x: str, y_star: str) -> float:
        """
        Estimate Var[r_task | perturb(x)] using m perturbed inputs.

        Each perturbed input uses the same path (frozen topology) so we only
        vary the prompt context, not the routing. This isolates prompt
        robustness rather than routing robustness.

        Perturbation LM calls are fired concurrently via ThreadPoolExecutor
        (I/O-bound; GIL released during network wait).
        """
        from concurrent.futures import ThreadPoolExecutor

        perturbed_inputs = self.buffer.get(x, m=self.cfg.m_perturbation)
        if len(perturbed_inputs) < 2:
            return 0.0

        node_ids = trace.node_ids
        assembler = self.assembler
        lm = self.lm
        metric = self.metric

        def _score(x_p: str) -> float:
            prompt_p = assembler.assemble(node_ids, {"input": x_p})
            y_p      = lm.complete(prompt_p)
            return metric.score(y_p, y_star)

        with ThreadPoolExecutor(max_workers=len(perturbed_inputs)) as pool:
            scores = list(pool.map(_score, perturbed_inputs))

        var = float(np.var(scores))
        return -self.cfg.lambda_variance * var
