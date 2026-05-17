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


class MultipleChoiceMetric:
    """
    Extracts the final standalone answer option before comparing.

    Intended for ARC/MMLU-style references such as A/B/C/D, while tolerating
    short prose outputs like "The answer is A.".
    """

    _ANSWER_RE = re.compile(
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([A-J]|[1-9])[\)\].:]?",
        re.IGNORECASE,
    )
    _OPTION_RE = re.compile(r"\b([A-J]|[1-9])\b", re.IGNORECASE)

    def score(self, prediction: str, reference: str) -> float:
        return 1.0 if self._extract(prediction) == self._extract(reference) else 0.0

    @classmethod
    def _extract(cls, text: str) -> str:
        answer_matches = cls._ANSWER_RE.findall(text.strip())
        if answer_matches:
            return answer_matches[-1].upper()
        matches = cls._OPTION_RE.findall(text.strip())
        if matches:
            return matches[-1].upper()
        return _normalize(text).upper()


# Registry for easy lookup by name
METRIC_REGISTRY: dict[str, TaskMetric] = {
    "exact_match":         ExactMatchMetric(),
    "numeric_exact_match": NumericExactMatchMetric(),
    "f1":                  F1Metric(),
    "substring":           SubstringMatchMetric(),
    "multiple_choice":     MultipleChoiceMetric(),
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
        chars = len(response)
        sentences = len([s for s in _re.split(r"[.!?]+", response.strip()) if s.strip()])

        # --- word count ---
        m = _re.search(r"(fewer than|less than|under)\s+(\d+)\s+word", ctext)
        if m:
            return words < int(m.group(2))
        m = _re.search(r"(at most|no more than|maximum of)\s+(\d+)\s+word", ctext)
        if m:
            return words <= int(m.group(2))
        m = _re.search(r"(more than|over)\s+(\d+)\s+word", ctext)
        if m:
            return words > int(m.group(2))
        m = _re.search(r"(at least|no fewer than|minimum of)\s+(\d+)\s+word", ctext)
        if m:
            return words >= int(m.group(2))
        m = _re.search(r"(exactly|around|about)\s+(\d+)\s+word", ctext)
        if m:
            n = int(m.group(2))
            return abs(words - n) <= max(3, int(n * 0.1))
        m = _re.search(r"(\d+)\s*(to|-)\s*(\d+)\s+word", ctext)
        if m:
            return int(m.group(1)) <= words <= int(m.group(3))

        # --- character count ---
        m = _re.search(r"(fewer than|less than|under)\s+(\d+)\s+char", ctext)
        if m:
            return chars < int(m.group(2))
        m = _re.search(r"(at most|no more than|maximum of)\s+(\d+)\s+char", ctext)
        if m:
            return chars <= int(m.group(2))
        m = _re.search(r"(more than|over)\s+(\d+)\s+char", ctext)
        if m:
            return chars > int(m.group(2))
        m = _re.search(r"(at least|no fewer than|minimum of)\s+(\d+)\s+char", ctext)
        if m:
            return chars >= int(m.group(2))

        # --- sentence count (comparative first, then bare target) ---
        m = _re.search(r"(fewer than|less than|under)\s+(\d+)\s+sentence", ctext)
        if m:
            return sentences < int(m.group(2))
        m = _re.search(r"(at most|no more than|maximum of)\s+(\d+)\s+sentence", ctext)
        if m:
            return sentences <= int(m.group(2))
        m = _re.search(r"(more than|over)\s+(\d+)\s+sentence", ctext)
        if m:
            return sentences > int(m.group(2))
        m = _re.search(r"(at least|no fewer than|minimum of)\s+(\d+)\s+sentence", ctext)
        if m:
            return sentences >= int(m.group(2))
        m = _re.search(r"(exactly|around|about)\s+(\d+)\s+sentence", ctext)
        if m:
            return abs(sentences - int(m.group(2))) <= 1
        m = _re.search(r"(\d+)\s*(to|-)\s*(\d+)\s+sentence", ctext)
        if m:
            return int(m.group(1)) <= sentences <= int(m.group(3))
        # bare "N sentence(s)" — treat as target with ±1 tolerance
        m = _re.search(r"(\d+)\s+sentence", ctext)
        if m:
            return abs(sentences - int(m.group(1))) <= 1

        return ctext in response.lower()

    def _check_format(self, response: str, ctext: str) -> bool:
        negated = bool(re.search(r"\b(do not|avoid|without|no)\b", ctext))
        if "bullet" in ctext or "bulleted" in ctext:
            has_format = bool(self._BULLET_RE.search(response))
            return not has_format if negated else has_format
        if "numbered list" in ctext or "ordered list" in ctext:
            has_format = bool(self._NUMBERED_RE.search(response))
            return not has_format if negated else has_format
        if "header" in ctext or "heading" in ctext:
            has_format = bool(self._HEADER_RE.search(response))
            return not has_format if negated else has_format
        if "json" in ctext:
            has_format = bool(self._JSON_RE.match(response.strip()))
            return not has_format if negated else has_format
        if "paragraph" in ctext:
            has_format = len([p for p in response.split("\n\n") if p.strip()]) >= 2
            return not has_format if negated else has_format
        return ctext in response.lower()

    def _check_keywords(self, response: str, ctext: str) -> bool:
        resp_lower = response.lower()
        import re as _re

        _KW_STOP = frozenset({
            "the", "a", "an", "any", "this", "that", "with", "and", "or",
            "word", "words", "phrase", "phrases", "keyword", "keywords",
            "include", "contain", "use", "mention", "avoid", "without",
            "not", "do", "no", "must", "should",
        })

        def _extract_kws(tail: str) -> list[str]:
            # Prefer explicitly quoted strings — most precise
            quoted = _re.findall(r"""['"]([^'"]+)['"]""", tail)
            if quoted:
                return [q.strip().lower() for q in quoted if q.strip()]
            # Fall back: content words (3+ chars) not in stop list
            return [
                w.lower() for w in _re.findall(r"[a-zA-Z][\w-]{2,}", tail)
                if w.lower() not in _KW_STOP
            ]

        # Negation: "do not use X and Y", "avoid X, Y", "without X"
        neg = _re.search(
            r'(?:do not|avoid|without|no)\s+'
            r'(?:(?:include|contain|use|mention)\s+)?'
            r'(?:any\s+)?(?:the\s+)?(?:words?\s+|phrases?\s+|keywords?\s+)?',
            ctext, _re.IGNORECASE,
        )
        if neg:
            tail = ctext[neg.end():]
            kws = _extract_kws(tail)
            if kws:
                return all(kw not in resp_lower for kw in kws)

        # Inclusion: "include the words X and Y", "must contain X"
        pos = _re.search(
            r'(?:include|contain|use|mention)\s+'
            r'(?:any\s+)?(?:the\s+)?(?:words?\s+|phrases?\s+|keywords?\s+)?',
            ctext, _re.IGNORECASE,
        )
        if pos:
            tail = ctext[pos.end():]
            kws = _extract_kws(tail)
            if kws:
                return all(kw in resp_lower for kw in kws)

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

    @classmethod
    def production(cls, **overrides) -> "RewardConfig":
        """Tuned config for maximum benchmark performance.

        Higher lambda_constraint prioritizes instruction-following.
        3 perturbations gives tighter variance estimates.
        max_tokens_ref=1024 matches real prompt lengths on rich graphs.
        """
        defaults = dict(
            lambda_cost=0.002,
            lambda_variance=0.15,
            lambda_constraint=0.4,
            m_perturbation=3,
            max_tokens_ref=1024,
        )
        defaults.update(overrides)
        return cls(**defaults)


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
        use_constraint_as_task = self.cfg.constraint_as_task or (
            self.checker is not None and constraints and self._is_constraint_primary(constraints)
        )

        if use_constraint_as_task and self.checker is not None:
            # Constraint-driven benchmarks (IFEval, IFBench): constraint satisfaction
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
            else self._variance(trace, x, y_star, constraints, metadata)
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

    @staticmethod
    def _is_constraint_primary(constraints: list[str]) -> bool:
        """Auto-detect whether constraints should be the primary task signal.

        Heuristic: if there are 2+ constraints, this is likely an IF benchmark
        where constraint satisfaction is more meaningful than string match.
        """
        return len(constraints) >= 2

    def _cost(self, token_count: int) -> float:
        normalised = token_count / max(1, self.cfg.max_tokens_ref)
        return -self.cfg.lambda_cost * normalised

    def _variance(
        self,
        trace: PathTrace,
        x: str,
        y_star: str,
        constraints: list[str],
        metadata: dict | None,
    ) -> float:
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

        use_constraint_as_task = self.cfg.constraint_as_task or (
            self.checker is not None and constraints and self._is_constraint_primary(constraints)
        )

        def _score(x_p: str) -> float:
            prompt_p = assembler.assemble(node_ids, {"input": x_p})
            y_p      = lm.complete(prompt_p)
            if use_constraint_as_task and self.checker is not None:
                return self.checker.check(y_p, constraints or [], metadata)
            return self.metric.score(y_p, y_star)

        with ThreadPoolExecutor(max_workers=len(perturbed_inputs)) as pool:
            scores = list(pool.map(_score, perturbed_inputs))

        var = float(np.var(scores))
        return -self.cfg.lambda_variance * var


# ---------------------------------------------------------------------------
# Pareto-based reward computation (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class ParetoPoint:
    """One point in objective space, linked back to its path."""
    objectives: np.ndarray   # [task, constraint, -cost, -variance] — all maximise
    path_key: str            # "|".join(node_ids) for dedup
    reward_components: RewardComponents

    def dominates(self, other: "ParetoPoint") -> bool:
        """True if self >= other on all objectives and > on at least one."""
        geq = self.objectives >= other.objectives
        gt  = self.objectives > other.objectives
        return bool(geq.all() and gt.any())


class ParetoArchive:
    """
    Maintains a bounded Pareto front of path configurations.

    Used by ParetoRewardComputer to assign dominance-rank rewards instead of
    using linear scalarization. Eliminates lambda hyperparameters.

    Dominance rank reward:
      - Points on Pareto front (rank 0): reward = 1.0
      - Points dominated by k front members: reward = 1.0 / (1 + k)
      - Crowding distance bonus for diversity on the front
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._archive: list[ParetoPoint] = []

    def add(self, point: ParetoPoint) -> float:
        """Add point, update front, return dominance-rank reward in [0, 1]."""
        # Count how many archive members dominate this point
        dominators = sum(1 for p in self._archive if p.dominates(point))

        # Remove points dominated by the new one
        self._archive = [p for p in self._archive if not point.dominates(p)]
        self._archive.append(point)

        # Prune to max_size using crowding distance
        if len(self._archive) > self.max_size:
            self._prune()

        # Reward: non-dominated = 1.0, otherwise decays with dominator count
        rank_reward = 1.0 / (1.0 + dominators)

        # Crowding distance bonus for front members
        if dominators == 0 and len(self._archive) > 2:
            cd = self._crowding_distance(point)
            rank_reward = min(1.0, rank_reward + 0.1 * cd)

        return float(rank_reward)

    @property
    def front_size(self) -> int:
        """Number of non-dominated points."""
        front = [p for p in self._archive
                 if not any(q.dominates(p) for q in self._archive if q is not p)]
        return len(front)

    @property
    def archive_size(self) -> int:
        return len(self._archive)

    def _crowding_distance(self, point: ParetoPoint) -> float:
        """Normalised crowding distance of point within the archive."""
        if len(self._archive) < 3:
            return 1.0
        n_obj = len(point.objectives)
        all_obj = np.array([p.objectives for p in self._archive])
        idx = next(i for i, p in enumerate(self._archive) if p is point)

        cd = 0.0
        for m in range(n_obj):
            sorted_idx = np.argsort(all_obj[:, m])
            pos = int(np.where(sorted_idx == idx)[0][0])
            obj_range = float(all_obj[:, m].max() - all_obj[:, m].min())
            if obj_range < 1e-12 or pos == 0 or pos == len(sorted_idx) - 1:
                cd += 1.0
            else:
                prev_val = all_obj[sorted_idx[pos - 1], m]
                next_val = all_obj[sorted_idx[pos + 1], m]
                cd += (next_val - prev_val) / obj_range

        return cd / n_obj

    def _prune(self):
        """Remove lowest crowding-distance points until at max_size."""
        while len(self._archive) > self.max_size:
            worst_idx = 0
            worst_cd = float("inf")
            for i, p in enumerate(self._archive):
                cd = self._crowding_distance(p)
                if cd < worst_cd:
                    worst_cd = cd
                    worst_idx = i
            self._archive.pop(worst_idx)


class ParetoRewardComputer(RewardComputer):
    """
    Extends RewardComputer with Pareto dominance-based reward assignment.

    Instead of total = task + lambda*constraint + cost + variance,
    maps each episode to a 4D objective vector and computes reward from
    dominance rank in the maintained Pareto archive.

    Falls back to scalarized total when archive is in warmup (< min_archive_size).
    """

    def __init__(self, *args, min_archive_size: int = 30,
                 archive_max_size: int = 200, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.archive = ParetoArchive(max_size=archive_max_size)
        self.min_archive_size = min_archive_size
        self._logger = logger

    def compute(
        self,
        trace,
        x: str,
        y_star: str,
        constraints=None,
        metadata=None,
    ) -> RewardComponents:
        """Compute reward components, then override total with Pareto rank."""
        base = super().compute(trace, x, y_star, constraints, metadata)

        objectives = np.array([
            base.task,
            base.constraint,
            -base.cost,       # negate so higher = better (lower cost)
            -base.variance,   # negate so higher = better (lower variance)
        ], dtype=np.float64)

        path_key = "|".join(trace.node_ids)
        point = ParetoPoint(
            objectives=objectives,
            path_key=path_key,
            reward_components=base,
        )

        if self.archive.archive_size < self.min_archive_size:
            self.archive.add(point)
            return base

        pareto_reward = self.archive.add(point)

        if self._logger is not None:
            self._logger.log_pareto(
                archive_size=self.archive.archive_size,
                front_size=self.archive.front_size,
                dominance_rank=pareto_reward,
            )

        return RewardComponents(
            task=base.task,
            constraint=base.constraint,
            cost=base.cost,
            variance=base.variance,
            total=pareto_reward,
        )
