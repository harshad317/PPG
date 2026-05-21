"""
Benchmark loaders for PPG evaluation.

Each loader pulls from the authoritative source on Hugging Face Hub via the
`datasets` library and converts examples into EvalExample objects.

Sources
-------
GSM8K         : openai/gsm8k                       (config: main)
IFEval        : google/IFEval                      (single "train" split = full eval set)
IFBench       : THU-KEG/IFBench                    (split: train = 444-example eval set)
HotpotQA      : hotpotqa/hotpot_qa                 (config: distractor)
DROP          : ucinlp/drop                        (no config)
MBPP          : google-research-datasets/mbpp      (config: full)
TruthfulQA    : truthfulqa/truthful_qa             (config: generation)
ARCChallenge  : allenai/ai2_arc                    (config: ARC-Challenge)
LiveBenchMath : livebench/math                     (no config)
MMLU          : cais/mmlu                          (config: subject or "all")

Recommended metrics
-------------------
GSM8K         : NumericExactMatchMetric  (extracts last number, ignores reasoning chain)
IFEval        : KeywordConstraintChecker (proxy; full eval needs Google's verifier suite)
IFBench       : KeywordConstraintChecker (proxy; full eval needs constraint verifier)
HotpotQA      : F1Metric                (token-overlap F1, official HotpotQA metric)
DROP          : F1Metric
MBPP          : MBPPPassAtOneMetric      (executes code against test cases in subprocess)
TruthfulQA    : F1Metric                (official eval also uses MC accuracy)
ARCChallenge  : MultipleChoiceMetric    (extracts letter A/B/C/D from prose)
LiveBenchMath : NumericExactMatchMetric
MMLU          : MultipleChoiceMetric    (extracts letter A/B/C/D from prose)

Security note
-------------
MBPPPassAtOneMetric executes LM-generated code in a subprocess. Only run
against trusted/sandboxed environments. Set timeout_seconds to limit runaway.
"""

from __future__ import annotations

import random
import re
import subprocess
import sys
from typing import Optional

from ppg.eval.harness import EvalExample


# ---------------------------------------------------------------------------
# Lazy datasets import
# ---------------------------------------------------------------------------

def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required: pip install datasets"
        ) from None
    return load_dataset(*args, **kwargs)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _sample(rows: list, n: Optional[int], seed: int) -> list:
    """Return up to n rows, shuffled with given seed."""
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

class GSM8KLoader:
    """
    Grade-school math word problems requiring multi-step arithmetic.

    Dataset : openai/gsm8k  (config: main)
    x       : question
    y_star  : numeric final answer (text after "####" in answer field)
    Metric  : NumericExactMatchMetric
    """

    DATASET_ID = "openai/gsm8k"
    CONFIG     = "main"

    def load(
        self,
        split: str = "test",
        n:     Optional[int] = None,
        seed:  int = 0,
    ) -> list[EvalExample]:
        """
        Parameters
        ----------
        split : "train" or "test"  (HF exposes train=7473, test=1319)
        n     : max examples; None = all
        seed  : shuffle seed when n < dataset size
        """
        ds = _load_dataset(self.DATASET_ID, self.CONFIG, split=split,
                           trust_remote_code=False)
        rows = list(ds)
        rows = _sample(rows, n, seed)
        return [
            EvalExample(
                x=row["question"],
                y_star=self._extract_answer(row["answer"]),
            )
            for row in rows
        ]

    @staticmethod
    def _extract_answer(answer_text: str) -> str:
        """Extract numeric answer after '####' separator."""
        parts = answer_text.split("####")
        return parts[-1].strip().replace(",", "") if len(parts) > 1 else answer_text.strip()

    @staticmethod
    def recommended_metric():
        from ppg.training.reward import NumericExactMatchMetric
        return NumericExactMatchMetric()


# ---------------------------------------------------------------------------
# IFEval
# ---------------------------------------------------------------------------

class IFEvalLoader:
    """
    Instruction-following evaluation: prompts with explicit format constraints.

    Dataset : google/IFEval  (single "train" split = the full 541-example eval set)
    x       : prompt
    y_star  : "" (no reference answer; compliance is verified by constraint checker)
    constraints : simplified list of instruction requirement strings
    Metric  : KeywordConstraintChecker (proxy)

    Note: The authoritative IFEval metric uses Google's format-verifier suite
    (regex, word-count, JSON validators). KeywordConstraintChecker is a proxy
    suitable for development; paper results should use the official verifier.
    Official verifier: https://github.com/google-research/google-research/tree/master/instruction_following_eval
    """

    DATASET_ID = "google/IFEval"

    def load(
        self,
        split: str = "train",   # "train" = full eval set (541 examples)
        n:     Optional[int] = None,
        seed:  int = 0,
    ) -> list[EvalExample]:
        ds   = _load_dataset(self.DATASET_ID, split=split,
                             trust_remote_code=False)
        rows = _sample(list(ds), n, seed)
        return [
            EvalExample(
                x=row["prompt"],
                y_star="",
                constraints=self._extract_constraints(row),
                metadata={
                    "instruction_id_list": row.get("instruction_id_list") or [],
                    "kwargs": row.get("kwargs") or [],
                },
            )
            for row in rows
        ]

    @staticmethod
    def _extract_constraints(row: dict) -> list[str]:
        """
        Convert instruction_id_list into human-readable constraint strings.
        E.g. "punctuation:no_comma" → "no comma"
        """
        constraints = []
        for instr_id in row.get("instruction_id_list", []):
            # instruction IDs are like "category:specific_requirement"
            part = instr_id.split(":")[-1].replace("_", " ")
            constraints.append(part)
        return constraints

    @staticmethod
    def recommended_metric():
        from ppg.training.reward import ExactMatchMetric
        return ExactMatchMetric()

    @staticmethod
    def recommended_constraint_checker():
        from ppg.training.reward import IFEvalOfficialChecker
        return IFEvalOfficialChecker()


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

class HotpotQALoader:
    """
    Multi-hop question answering requiring reasoning over 2 supporting passages.

    Dataset : hotpotqa/hotpot_qa  (config: distractor — 10-passage distractor setting)
    x       : question + gold context passages (distractors included)
    y_star  : short answer string
    Metric  : F1Metric (official HotpotQA metric is EM + F1)
    """

    DATASET_ID = "hotpotqa/hotpot_qa"
    CONFIG     = "distractor"

    def load(
        self,
        split:           str  = "validation",
        n:               Optional[int] = None,
        seed:            int  = 0,
        include_context: bool = True,
    ) -> list[EvalExample]:
        """
        Parameters
        ----------
        include_context : prepend gold context passages to the question
                          (required for open-book evaluation; default True)
        """
        ds   = _load_dataset(self.DATASET_ID, self.CONFIG, split=split,
                             trust_remote_code=False)
        rows = _sample(list(ds), n, seed)
        return [
            EvalExample(
                x=self._format_input(row, include_context),
                y_star=row["answer"],
                metadata={
                    "question_type": row.get("type", ""),
                    "level": row.get("level", ""),
                },
            )
            for row in rows
        ]

    @staticmethod
    def _format_input(row: dict, include_context: bool) -> str:
        if not include_context:
            return row["question"]
        # context is {"title": [str], "sentences": [[str]]}
        passages = []
        ctx = row.get("context", {})
        titles    = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        for title, sents in zip(titles, sentences):
            passages.append(f"{title}: {''.join(sents)}")
        context_str = "\n".join(passages)
        return f"Context:\n{context_str}\n\nQuestion: {row['question']}"

    @staticmethod
    def recommended_metric():
        from ppg.training.reward import F1Metric
        return F1Metric()


# ---------------------------------------------------------------------------
# DROP
# ---------------------------------------------------------------------------

class DROPLoader:
    """
    Discrete reasoning over paragraphs: arithmetic, counting, sorting.

    Dataset : ucinlp/drop  (no config)
    x       : passage + question
    y_star  : first span from answers_spans (may be a number or entity)
    Metric  : F1Metric (official DROP metric is EM + F1)
    """

    DATASET_ID = "ucinlp/drop"

    def load(
        self,
        split: str = "validation",
        n:     Optional[int] = None,
        seed:  int = 0,
    ) -> list[EvalExample]:
        ds   = _load_dataset(self.DATASET_ID, split=split,
                             trust_remote_code=False)
        rows = _sample(list(ds), n, seed)
        examples = []
        for row in rows:
            y_star = self._extract_answer(row)
            if not y_star:
                continue  # skip unanswerable examples
            examples.append(EvalExample(
                x=self._format_input(row),
                y_star=y_star,
            ))
        return examples

    @staticmethod
    def _format_input(row: dict) -> str:
        return f"Passage: {row['passage']}\nQuestion: {row['question']}"

    @staticmethod
    def _extract_answer(row: dict) -> str:
        spans = row.get("answers_spans", {}).get("spans", [])
        return spans[0].strip() if spans else ""

    @staticmethod
    def recommended_metric():
        from ppg.training.reward import F1Metric
        return F1Metric()


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

class MBPPLoader:
    """
    Mostly Basic Programming Problems: function synthesis from docstrings.

    Dataset : google-research-datasets/mbpp  (config: full)
    x       : problem description (text/prompt field)
    y_star  : reference solution (code field); used only for fallback metrics
    Metric  : MBPPPassAtOneMetric (executes generated code against test_list)

    The test_list and task_id are stored in EvalExample.metadata so that
    MBPPPassAtOneMetric can retrieve them at evaluation time.
    """

    DATASET_ID = "google-research-datasets/mbpp"
    CONFIG     = "full"

    def load(
        self,
        split: str = "test",    # test=500, train=374, validation=90
        n:     Optional[int] = None,
        seed:  int = 0,
    ) -> list[EvalExample]:
        ds   = _load_dataset(self.DATASET_ID, self.CONFIG, split=split,
                             trust_remote_code=False)
        rows = _sample(list(ds), n, seed)
        return [
            EvalExample(
                x=self._format_input(row),
                y_star=row.get("code", ""),
                metadata={"test_list": row.get("test_list", []), "task_id": row.get("task_id")},
            )
            for row in rows
        ]

    @staticmethod
    def _format_input(row: dict) -> str:
        text = row.get("text", row.get("prompt", ""))
        tests = row.get("test_list", [])
        if tests:
            test_preview = tests[0]
            return (
                f"Write a Python function to solve the following problem.\n\n"
                f"{text}\n\n"
                f"Your function must pass this test: {test_preview}"
            )
        return f"Write a Python function to solve the following problem.\n\n{text}"

    @staticmethod
    def recommended_metric():
        return MBPPPassAtOneMetric()


# ---------------------------------------------------------------------------
# MBPPPassAtOneMetric
# ---------------------------------------------------------------------------

class MBPPPassAtOneMetric:
    """
    Executes LM-generated code against MBPP test cases in a subprocess.

    SECURITY: Only use with trusted/sandboxed LM outputs.
    Generated code is executed directly; malicious code can harm the host.

    Scoring:
        1.0  — code executes without error AND all test assertions pass
        0.0  — syntax error, runtime error, or any assertion fails

    Parameters
    ----------
    timeout_seconds : subprocess wall-clock limit per evaluation
    """

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds

    def score(self, prediction: str, reference: str) -> float:
        """
        prediction : LM-generated code (may include markdown fences)
        reference  : ground-truth code from MBPP (used to extract test cases)

        Test cases are extracted from the reference code's assert statements.
        If no asserts found, returns 0.0.
        """
        code = self._extract_code(prediction)
        tests = self._extract_tests(reference)
        if not tests:
            return 0.0
        return self._run(code, tests)

    def score_with_tests(self, prediction: str, test_list: list[str]) -> float:
        """
        Score when test_list is available directly (preferred over score()).
        test_list : list of assert statements from MBPP row["test_list"]
        """
        code = self._extract_code(prediction)
        if not test_list:
            return 0.0
        return self._run(code, test_list)

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(text: str) -> str:
        """Strip markdown fences if present."""
        # ```python ... ``` or ``` ... ```
        fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if fence:
            return fence.group(1)
        return text

    @staticmethod
    def _extract_tests(reference_code: str) -> list[str]:
        """Extract assert lines from reference code."""
        return [
            line.strip()
            for line in reference_code.splitlines()
            if line.strip().startswith("assert")
        ]

    def _run(self, code: str, tests: list[str]) -> float:
        """Execute code + tests in isolated subprocess with resource limits."""
        test_block = "\n".join(tests)
        script = f"{code}\n\n{test_block}\n"

        def _set_limits() -> None:
            try:
                import resource
                mem = 256 * 1024 * 1024  # 256 MB virtual memory
                cpu = max(1, int(self.timeout))
                resource.setrlimit(resource.RLIMIT_AS,  (mem, mem))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            except Exception:
                pass

        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                timeout=self.timeout,
                preexec_fn=_set_limits if sys.platform != "win32" else None,
            )
            return 1.0 if result.returncode == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# IFBench
# ---------------------------------------------------------------------------

class IFBenchLoader:
    """
    Instruction following benchmark requiring constraint satisfaction.

    Dataset : THU-KEG/IFBench  (split: train = 444 eval examples)
    x       : instruction
    y_star  : chosen["content"] (high-quality reference response)
    Metric  : KeywordConstraintChecker (proxy)

    IFBench tests fine-grained instruction following with explicit constraints
    (keyword inclusion, format requirements, length, style, etc.).

    Reference: https://github.com/THU-KEG/IFBench
    Paper: IFBench: Benchmarking Large Language Models on Instruction Following (2024)
    """

    DATASET_ID = "THU-KEG/IFBench"

    def load(
        self,
        split: str = "train",   # train = full 444-example eval set
        n:     Optional[int] = None,
        seed:  int = 0,
    ) -> list[EvalExample]:
        ds   = _load_dataset(self.DATASET_ID, split=split, trust_remote_code=False)
        rows = _sample(list(ds), n, seed)
        return [
            EvalExample(
                x=row["instruction"],
                y_star=row["chosen"]["content"],
                constraints=self._extract_constraints(row),
                metadata={
                    "constraint_objects": self._extract_constraint_objects(row),
                },
            )
            for row in rows
        ]

    @staticmethod
    def _extract_constraints(row: dict) -> list[str]:
        all_constraints = (
            [c for c in row.get("llm_constraints_used", []) or [] if isinstance(c, dict) and "constraint" in c]
            + [c for c in row.get("code_constraints_used", []) or [] if isinstance(c, dict) and "constraint" in c]
        )
        return [c["constraint"] for c in all_constraints]

    @staticmethod
    def _extract_constraint_objects(row: dict) -> list[dict]:
        return (
            [c for c in row.get("llm_constraints_used", []) or [] if isinstance(c, dict) and "constraint" in c]
            + [c for c in row.get("code_constraints_used", []) or [] if isinstance(c, dict) and "constraint" in c]
        )

    @staticmethod
    def recommended_metric():
        from ppg.training.reward import ExactMatchMetric
        return ExactMatchMetric()

    @staticmethod
    def recommended_constraint_checker():
        from ppg.training.reward import IFBenchConstraintChecker
        return IFBenchConstraintChecker()


# ---------------------------------------------------------------------------
# TruthfulQA
# ---------------------------------------------------------------------------

class TruthfulQALoader:
    """
    Truthfulness evaluation: questions designed to elicit common misconceptions.

    Dataset : truthfulqa/truthful_qa  (config: generation)
    x       : question
    y_star  : best_answer (top human-rated truthful answer)
    Metric  : F1Metric

    All 817 examples are in the "validation" split — there is no separate test split.
    The "multiple_choice" config offers MC scoring; "generation" config used here.

    Reference: https://github.com/sylinrl/TruthfulQA
    Paper: TruthfulQA: Measuring How Models Mimic Human Falsehoods (Lin et al. 2022)
    """

    DATASET_ID = "truthfulqa/truthful_qa"
    CONFIG     = "generation"

    def load(
        self,
        split: str = "validation",   # validation = 817 examples (full eval set)
        n:     Optional[int] = None,
        seed:  int = 0,
    ) -> list[EvalExample]:
        ds   = _load_dataset(self.DATASET_ID, self.CONFIG, split=split,
                             trust_remote_code=False)
        rows = _sample(list(ds), n, seed)
        return [
            EvalExample(
                x=row["question"],
                y_star=row["best_answer"],
            )
            for row in rows
        ]

    @staticmethod
    def recommended_metric():
        from ppg.training.reward import F1Metric
        return F1Metric()


# ---------------------------------------------------------------------------
# ARC Challenge
# ---------------------------------------------------------------------------

class ARCChallengeLoader:
    """
    AI2 Reasoning Challenge — Challenge set: hard science exam questions.

    Dataset : allenai/ai2_arc  (config: ARC-Challenge)
    x       : question + labeled choices (A. ... \\nB. ... \\nC. ... \\nD. ...)
    y_star  : answerKey letter (A/B/C/D; rarely 1/2/3/4 in some rows)
    Metric  : MultipleChoiceMetric

    ARC-Challenge contains questions that require scientific knowledge beyond
    simple retrieval — IR and word-co-occurrence methods score below random.

    Reference: https://allenai.org/data/arc
    Paper: Think you have Solved QA? Try ARC (Clark et al. 2018)
    """

    DATASET_ID = "allenai/ai2_arc"
    CONFIG     = "ARC-Challenge"

    def load(
        self,
        split: str = "test",      # test=1172, validation=299, train=1119
        n:     Optional[int] = None,
        seed:  int = 0,
    ) -> list[EvalExample]:
        ds   = _load_dataset(self.DATASET_ID, self.CONFIG, split=split,
                             trust_remote_code=False)
        rows = _sample(list(ds), n, seed)
        return [
            EvalExample(
                x=self._format_input(row),
                y_star=row["answerKey"],
            )
            for row in rows
        ]

    @staticmethod
    def _format_input(row: dict) -> str:
        choices = row["choices"]
        labels  = choices["label"]
        texts   = choices["text"]
        choice_lines = "\n".join(
            f"{label}. {text}"
            for label, text in zip(labels, texts)
        )
        return f"{row['question']}\n\n{choice_lines}"

    @staticmethod
    def recommended_metric():
        from ppg.training.reward import MultipleChoiceMetric
        return MultipleChoiceMetric()


# ---------------------------------------------------------------------------
# LiveBench Math
# ---------------------------------------------------------------------------

class LiveBenchMathLoader:
    """
    LiveBench Math: contamination-free math reasoning benchmark updated monthly.

    Dataset : livebench/math  (no config; split: test)
    x       : question text (from "question" field; falls back to first turn)
    y_star  : ground_truth (exact answer string)
    Metric  : NumericExactMatchMetric

    LiveBench refreshes questions monthly using competition problems and
    recent math olympiad questions — prevents contamination from training data.
    Math categories include: AMPS_Hard, AMC, AIME, Olympiad, and more.

    Reference: https://livebench.ai
    Paper: LiveBench: A Challenging, Contamination-Free LLM Benchmark (White et al. 2024)
    """

    DATASET_ID = "livebench/math"

    def load(
        self,
        split: str = "test",
        n:     Optional[int] = None,
        seed:  int = 0,
    ) -> list[EvalExample]:
        ds   = _load_dataset(self.DATASET_ID, split=split, trust_remote_code=False)
        rows = _sample(list(ds), n, seed)
        return [
            EvalExample(
                x=self._extract_question(row),
                y_star=str(row["ground_truth"]).strip(),
            )
            for row in rows
        ]

    @staticmethod
    def _extract_question(row: dict) -> str:
        # Primary field is "question"; LiveBench also uses "turns" (list of messages)
        if row.get("question"):
            return str(row["question"]).strip()
        turns = row.get("turns", [])
        return str(turns[0]).strip() if turns else ""

    @staticmethod
    def recommended_metric():
        from ppg.training.reward import NumericExactMatchMetric
        return NumericExactMatchMetric()


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

class MMLULoader:
    """
    Massive Multitask Language Understanding: 57-subject multiple-choice benchmark.

    Dataset : cais/mmlu  (config: subject name or "all")
    x       : question + labeled choices (A. ... \\nB. ... \\nC. ... \\nD. ...)
    y_star  : correct answer letter (A/B/C/D)
    Metric  : MultipleChoiceMetric

    Use subject="all" to load all subjects combined (14,042 test examples).
    Use a specific subject name for targeted evaluation. See SUBJECTS for all 57.

    Reference: https://github.com/hendrycks/test
    Paper: Measuring Massive Multitask Language Understanding (Hendrycks et al. 2021)
    """

    DATASET_ID = "cais/mmlu"

    SUBJECTS: list[str] = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ]

    _LETTERS = ["A", "B", "C", "D"]

    def load(
        self,
        subject: str = "all",
        split:   str = "test",      # test, validation, dev, auxiliary_train
        n:       Optional[int] = None,
        seed:    int = 0,
    ) -> list[EvalExample]:
        """
        Parameters
        ----------
        subject : "all" or any name from MMLULoader.SUBJECTS
        split   : "test" (~100 examples per subject), "validation", "dev", "auxiliary_train"
        """
        if subject != "all" and subject not in self.SUBJECTS:
            raise ValueError(
                f"Unknown MMLU subject: {subject!r}. "
                f"Use 'all' or a name from MMLULoader.SUBJECTS."
            )
        ds   = _load_dataset(self.DATASET_ID, subject, split=split,
                             trust_remote_code=False)
        rows = _sample(list(ds), n, seed)
        return [
            EvalExample(
                x=self._format_input(row),
                y_star=self._LETTERS[row["answer"]],
            )
            for row in rows
        ]

    @staticmethod
    def _format_input(row: dict) -> str:
        choices = row["choices"]   # list of 4 strings
        choice_lines = "\n".join(
            f"{letter}. {text}"
            for letter, text in zip(["A", "B", "C", "D"], choices)
        )
        return f"{row['question']}\n\n{choice_lines}"

    @staticmethod
    def recommended_metric():
        from ppg.training.reward import MultipleChoiceMetric
        return MultipleChoiceMetric()
