"""
Tests for ppg/eval/benchmarks/loaders.py.

All tests mock datasets.load_dataset — no network required.
Tests cover:
  - preprocessing / field extraction for each loader
  - EvalExample structure (x, y_star, constraints)
  - Sampling / shuffling
  - MBPPPassAtOneMetric execution
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ppg.eval.benchmarks.loaders import (
    ARCChallengeLoader,
    DROPLoader,
    GSM8KLoader,
    HotpotQALoader,
    IFBenchLoader,
    IFEvalLoader,
    LiveBenchMathLoader,
    MBPPLoader,
    MBPPPassAtOneMetric,
    MMLULoader,
    TruthfulQALoader,
)
from ppg.eval.harness import EvalExample


# ---------------------------------------------------------------------------
# Fixtures — raw dataset rows matching actual HF schema
# ---------------------------------------------------------------------------

GSM8K_ROWS = [
    {
        "question": "Natalia sold clips to 48 friends.",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 24",
    },
    {
        "question": "Weng earns $12 an hour babysitting.",
        "answer": "Weng earns 12/60 = <<12/60=0.2>>0.2 per minute.\n#### 4.8",
    },
    {
        "question": "Betty had 5 apples and ate 2.",
        "answer": "Betty has 5-2=<<5-2=3>>3 apples.\n#### 3",
    },
]

IFEVAL_ROWS = [
    {
        "key": 0,
        "prompt": "Write a poem about autumn. Do not use commas.",
        "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_sentences_5"],
        "kwargs": [{}, {}],
    },
    {
        "key": 1,
        "prompt": "Explain quantum physics. Use exactly 3 paragraphs.",
        "instruction_id_list": ["length_constraints:number_paragraphs_3"],
        "kwargs": [{}],
    },
]

HOTPOTQA_ROWS = [
    {
        "id": "abc123",
        "question": "Who was the president of France in 2020?",
        "answer": "Emmanuel Macron",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": {"title": ["France"], "sent_id": [0]},
        "context": {
            "title": ["France", "Emmanuel Macron"],
            "sentences": [
                ["France is a country in Europe.", "Its capital is Paris."],
                ["Emmanuel Macron became president in 2017.", "He was re-elected in 2022."],
            ],
        },
    },
    {
        "id": "def456",
        "question": "What is the capital of Germany?",
        "answer": "Berlin",
        "type": "comparison",
        "level": "easy",
        "supporting_facts": {"title": ["Germany"], "sent_id": [0]},
        "context": {
            "title": ["Germany"],
            "sentences": [["Germany's capital is Berlin."]],
        },
    },
]

DROP_ROWS = [
    {
        "section_id": "nfl_1",
        "query_id": "q1",
        "passage": "In 2020, the team scored 42 points in the first quarter.",
        "question": "How many points did the team score?",
        "answers_spans": {"spans": ["42"], "types": ["number"]},
    },
    {
        "section_id": "nfl_2",
        "query_id": "q2",
        "passage": "The Super Bowl was played in Tampa Bay.",
        "question": "Where was the Super Bowl played?",
        "answers_spans": {"spans": ["Tampa Bay"], "types": ["span"]},
    },
    {
        "section_id": "nfl_3",
        "query_id": "q3",
        "passage": "Empty example.",
        "question": "Unanswerable?",
        "answers_spans": {"spans": [], "types": []},  # unanswerable
    },
]

MBPP_ROWS = [
    {
        "task_id": 1,
        "text": "Write a function to find the sum of a list.",
        "code": "def sum_list(lst):\n    return sum(lst)\nassert sum_list([1,2,3]) == 6",
        "test_list": ["assert sum_list([1,2,3]) == 6", "assert sum_list([]) == 0"],
        "test_setup_code": "",
    },
    {
        "task_id": 2,
        "text": "Write a function to reverse a string.",
        "code": "def reverse_str(s):\n    return s[::-1]\nassert reverse_str('abc') == 'cba'",
        "test_list": ["assert reverse_str('abc') == 'cba'"],
        "test_setup_code": "",
    },
]


def mock_load_dataset(rows):
    """Return a context manager that patches datasets.load_dataset."""
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(return_value=iter(rows))
    mock_ds.__len__ = MagicMock(return_value=len(rows))
    mock_load = MagicMock(return_value=mock_ds)
    return patch("ppg.eval.benchmarks.loaders._load_dataset", mock_load)


# ---------------------------------------------------------------------------
# GSM8KLoader
# ---------------------------------------------------------------------------

class TestGSM8KLoader:
    def test_dataset_id(self):
        assert GSM8KLoader.DATASET_ID == "openai/gsm8k"

    def test_config_is_main(self):
        assert GSM8KLoader.CONFIG == "main"

    def test_load_returns_eval_examples(self):
        with mock_load_dataset(GSM8K_ROWS):
            examples = GSM8KLoader().load()
        assert all(isinstance(e, EvalExample) for e in examples)

    def test_load_count(self):
        with mock_load_dataset(GSM8K_ROWS):
            examples = GSM8KLoader().load()
        assert len(examples) == len(GSM8K_ROWS)

    def test_x_is_question(self):
        with mock_load_dataset(GSM8K_ROWS):
            examples = GSM8KLoader().load()
        assert examples[0].x == GSM8K_ROWS[0]["question"]

    def test_y_star_extracted_from_hash_separator(self):
        with mock_load_dataset(GSM8K_ROWS):
            examples = GSM8KLoader().load()
        assert examples[0].y_star == "24"
        assert examples[1].y_star == "4.8"
        assert examples[2].y_star == "3"

    def test_y_star_strips_commas(self):
        rows = [{
            "question": "q",
            "answer": "Lots of steps...\n#### 1,000",
        }]
        with mock_load_dataset(rows):
            examples = GSM8KLoader().load()
        assert examples[0].y_star == "1000"

    def test_n_limits_examples(self):
        with mock_load_dataset(GSM8K_ROWS):
            examples = GSM8KLoader().load(n=2)
        assert len(examples) == 2

    def test_recommended_metric_is_numeric(self):
        from ppg.training.reward import NumericExactMatchMetric
        assert isinstance(GSM8KLoader.recommended_metric(), NumericExactMatchMetric)

    def test_extract_answer_no_separator(self):
        result = GSM8KLoader._extract_answer("just a number: 42")
        assert result == "just a number: 42"

    def test_extract_answer_multiple_hashes(self):
        result = GSM8KLoader._extract_answer("step1\n#### 10\n#### 20")
        assert result == "20"


# ---------------------------------------------------------------------------
# IFEvalLoader
# ---------------------------------------------------------------------------

class TestIFEvalLoader:
    def test_dataset_id(self):
        assert IFEvalLoader.DATASET_ID == "google/IFEval"

    def test_default_split_is_train(self):
        """IFEval only has 'train' split (= full 541-example eval set)."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(IFEVAL_ROWS))
        mock_ds.__len__ = MagicMock(return_value=len(IFEVAL_ROWS))
        mock_load = MagicMock(return_value=mock_ds)
        with patch("ppg.eval.benchmarks.loaders._load_dataset", mock_load):
            IFEvalLoader().load()
        call_kwargs = mock_load.call_args
        assert "train" in str(call_kwargs)

    def test_x_is_prompt(self):
        with mock_load_dataset(IFEVAL_ROWS):
            examples = IFEvalLoader().load()
        assert examples[0].x == IFEVAL_ROWS[0]["prompt"]

    def test_y_star_is_empty_string(self):
        with mock_load_dataset(IFEVAL_ROWS):
            examples = IFEvalLoader().load()
        assert examples[0].y_star == ""

    def test_constraints_extracted(self):
        with mock_load_dataset(IFEVAL_ROWS):
            examples = IFEvalLoader().load()
        # instruction_id_list: ["punctuation:no_comma", "length_constraints:number_sentences_5"]
        # → constraints: ["no comma", "number sentences 5"]
        assert len(examples[0].constraints) == 2

    def test_constraint_text_readable(self):
        with mock_load_dataset(IFEVAL_ROWS):
            examples = IFEvalLoader().load()
        for c in examples[0].constraints:
            assert isinstance(c, str)
            assert len(c) > 0

    def test_no_instructions_gives_empty_constraints(self):
        rows = [{"key": 0, "prompt": "hi", "instruction_id_list": [], "kwargs": []}]
        with mock_load_dataset(rows):
            examples = IFEvalLoader().load()
        assert examples[0].constraints == []

    def test_recommended_metric_is_exact_match(self):
        from ppg.training.reward import ExactMatchMetric
        assert isinstance(IFEvalLoader.recommended_metric(), ExactMatchMetric)

    def test_recommended_constraint_checker_is_official(self):
        from ppg.training.reward import IFEvalOfficialChecker
        assert isinstance(IFEvalLoader.recommended_constraint_checker(), IFEvalOfficialChecker)


# ---------------------------------------------------------------------------
# HotpotQALoader
# ---------------------------------------------------------------------------

class TestHotpotQALoader:
    def test_dataset_id(self):
        assert HotpotQALoader.DATASET_ID == "hotpotqa/hotpot_qa"

    def test_config_is_distractor(self):
        assert HotpotQALoader.CONFIG == "distractor"

    def test_y_star_is_answer(self):
        with mock_load_dataset(HOTPOTQA_ROWS):
            examples = HotpotQALoader().load()
        assert examples[0].y_star == "Emmanuel Macron"
        assert examples[1].y_star == "Berlin"

    def test_x_includes_question(self):
        with mock_load_dataset(HOTPOTQA_ROWS):
            examples = HotpotQALoader().load()
        assert "Who was the president" in examples[0].x

    def test_x_includes_context_by_default(self):
        with mock_load_dataset(HOTPOTQA_ROWS):
            examples = HotpotQALoader().load(include_context=True)
        assert "Context:" in examples[0].x
        assert "Emmanuel Macron" in examples[0].x

    def test_x_no_context_when_disabled(self):
        with mock_load_dataset(HOTPOTQA_ROWS):
            examples = HotpotQALoader().load(include_context=False)
        assert "Context:" not in examples[0].x
        assert examples[0].x == HOTPOTQA_ROWS[0]["question"]

    def test_load_count(self):
        with mock_load_dataset(HOTPOTQA_ROWS):
            examples = HotpotQALoader().load()
        assert len(examples) == len(HOTPOTQA_ROWS)

    def test_n_limits_examples(self):
        with mock_load_dataset(HOTPOTQA_ROWS):
            examples = HotpotQALoader().load(n=1)
        assert len(examples) == 1

    def test_recommended_metric_is_f1(self):
        from ppg.training.reward import F1Metric
        assert isinstance(HotpotQALoader.recommended_metric(), F1Metric)

    def test_context_titles_present_in_x(self):
        with mock_load_dataset(HOTPOTQA_ROWS):
            examples = HotpotQALoader().load()
        assert "France" in examples[0].x


# ---------------------------------------------------------------------------
# DROPLoader
# ---------------------------------------------------------------------------

class TestDROPLoader:
    def test_dataset_id(self):
        assert DROPLoader.DATASET_ID == "ucinlp/drop"

    def test_y_star_is_first_span(self):
        with mock_load_dataset(DROP_ROWS):
            examples = DROPLoader().load()
        assert examples[0].y_star == "42"
        assert examples[1].y_star == "Tampa Bay"

    def test_x_includes_passage_and_question(self):
        with mock_load_dataset(DROP_ROWS):
            examples = DROPLoader().load()
        assert "Passage:" in examples[0].x
        assert "Question:" in examples[0].x
        assert DROP_ROWS[0]["passage"] in examples[0].x
        assert DROP_ROWS[0]["question"] in examples[0].x

    def test_unanswerable_rows_skipped(self):
        """Rows with empty spans should be filtered out."""
        with mock_load_dataset(DROP_ROWS):
            examples = DROPLoader().load()
        # DROP_ROWS has 3 rows; 1 unanswerable → 2 examples
        assert len(examples) == 2

    def test_n_limits_examples(self):
        with mock_load_dataset(DROP_ROWS):
            examples = DROPLoader().load(n=1)
        assert len(examples) == 1

    def test_recommended_metric_is_f1(self):
        from ppg.training.reward import F1Metric
        assert isinstance(DROPLoader.recommended_metric(), F1Metric)

    def test_no_empty_y_star(self):
        with mock_load_dataset(DROP_ROWS):
            examples = DROPLoader().load()
        assert all(ex.y_star != "" for ex in examples)


# ---------------------------------------------------------------------------
# MBPPLoader
# ---------------------------------------------------------------------------

class TestMBPPLoader:
    def test_dataset_id(self):
        assert MBPPLoader.DATASET_ID == "google-research-datasets/mbpp"

    def test_config_is_full(self):
        assert MBPPLoader.CONFIG == "full"

    def test_load_returns_examples(self):
        with mock_load_dataset(MBPP_ROWS):
            examples = MBPPLoader().load()
        assert len(examples) == len(MBPP_ROWS)
        assert all(isinstance(e, EvalExample) for e in examples)

    def test_x_contains_problem_description(self):
        with mock_load_dataset(MBPP_ROWS):
            examples = MBPPLoader().load()
        assert "sum" in examples[0].x.lower()

    def test_x_contains_test_preview(self):
        with mock_load_dataset(MBPP_ROWS):
            examples = MBPPLoader().load()
        assert "assert" in examples[0].x

    def test_y_star_is_reference_code(self):
        with mock_load_dataset(MBPP_ROWS):
            examples = MBPPLoader().load()
        assert "def sum_list" in examples[0].y_star

    def test_n_limits_examples(self):
        with mock_load_dataset(MBPP_ROWS):
            examples = MBPPLoader().load(n=1)
        assert len(examples) == 1

    def test_recommended_metric_is_pass_at_one(self):
        assert isinstance(MBPPLoader.recommended_metric(), MBPPPassAtOneMetric)

    def test_metadata_has_test_list(self):
        """MBPP examples should carry test_list in metadata for score_with_tests dispatch."""
        with mock_load_dataset(MBPP_ROWS):
            examples = MBPPLoader().load()
        assert "test_list" in examples[0].metadata
        assert examples[0].metadata["test_list"] == MBPP_ROWS[0]["test_list"]

    def test_metadata_has_task_id(self):
        with mock_load_dataset(MBPP_ROWS):
            examples = MBPPLoader().load()
        assert examples[0].metadata["task_id"] == MBPP_ROWS[0]["task_id"]


# ---------------------------------------------------------------------------
# MBPPPassAtOneMetric
# ---------------------------------------------------------------------------

class TestMBPPPassAtOneMetric:
    def test_correct_code_scores_one(self):
        metric = MBPPPassAtOneMetric()
        code   = "def add(a, b):\n    return a + b"
        tests  = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        assert metric.score_with_tests(code, tests) == pytest.approx(1.0)

    def test_wrong_code_scores_zero(self):
        metric = MBPPPassAtOneMetric()
        code   = "def add(a, b):\n    return a - b"  # wrong operation
        tests  = ["assert add(1, 2) == 3"]
        assert metric.score_with_tests(code, tests) == pytest.approx(0.0)

    def test_syntax_error_scores_zero(self):
        metric = MBPPPassAtOneMetric()
        code   = "def add(a b:\n    return a + b"   # syntax error
        tests  = ["assert add(1, 2) == 3"]
        assert metric.score_with_tests(code, tests) == pytest.approx(0.0)

    def test_runtime_error_scores_zero(self):
        metric = MBPPPassAtOneMetric()
        code   = "def add(a, b):\n    raise ValueError('boom')"
        tests  = ["assert add(1, 2) == 3"]
        assert metric.score_with_tests(code, tests) == pytest.approx(0.0)

    def test_empty_tests_scores_zero(self):
        metric = MBPPPassAtOneMetric()
        code   = "def f(): pass"
        assert metric.score_with_tests(code, []) == pytest.approx(0.0)

    def test_extracts_code_from_markdown_fence(self):
        metric = MBPPPassAtOneMetric()
        code_in_fence = "```python\ndef add(a, b):\n    return a + b\n```"
        tests = ["assert add(2, 3) == 5"]
        assert metric.score_with_tests(code_in_fence, tests) == pytest.approx(1.0)

    def test_extracts_code_from_plain_fence(self):
        metric = MBPPPassAtOneMetric()
        code_in_fence = "```\ndef add(a, b):\n    return a + b\n```"
        tests = ["assert add(2, 3) == 5"]
        assert metric.score_with_tests(code_in_fence, tests) == pytest.approx(1.0)

    def test_score_uses_reference_asserts(self):
        """score() extracts assert lines from reference code."""
        metric = MBPPPassAtOneMetric()
        prediction = "def add(a, b):\n    return a + b"
        reference  = "def add(a, b):\n    return a + b\nassert add(1,2)==3"
        assert metric.score(prediction, reference) == pytest.approx(1.0)

    def test_score_no_asserts_in_reference_scores_zero(self):
        metric = MBPPPassAtOneMetric()
        prediction = "def add(a, b):\n    return a + b"
        reference  = "def add(a, b):\n    return a + b"  # no assert
        assert metric.score(prediction, reference) == pytest.approx(0.0)

    def test_timeout_scores_zero(self):
        """Infinite loop should time out and score 0."""
        metric = MBPPPassAtOneMetric(timeout_seconds=1.0)
        code   = "def f():\n    while True: pass"
        tests  = ["assert f() is None"]
        assert metric.score_with_tests(code, tests) == pytest.approx(0.0)

    def test_all_tests_must_pass(self):
        """Partial test pass = 0.0 (pass@1 semantics: all or nothing)."""
        metric = MBPPPassAtOneMetric()
        code   = "def add(a, b):\n    return a + b"
        tests  = ["assert add(1, 2) == 3", "assert add(1, 2) == 99"]  # second fails
        assert metric.score_with_tests(code, tests) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sampling / shuffling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_n_none_returns_all(self):
        with mock_load_dataset(GSM8K_ROWS):
            examples = GSM8KLoader().load(n=None)
        assert len(examples) == len(GSM8K_ROWS)

    def test_n_larger_than_dataset_returns_all(self):
        with mock_load_dataset(GSM8K_ROWS):
            examples = GSM8KLoader().load(n=1000)
        assert len(examples) == len(GSM8K_ROWS)

    def test_same_seed_same_sample(self):
        with mock_load_dataset(GSM8K_ROWS):
            e1 = GSM8KLoader().load(n=2, seed=42)
        with mock_load_dataset(GSM8K_ROWS):
            e2 = GSM8KLoader().load(n=2, seed=42)
        assert [ex.x for ex in e1] == [ex.x for ex in e2]

    def test_different_seeds_may_differ(self):
        """With n<len and different seeds, order may differ."""
        with mock_load_dataset(GSM8K_ROWS):
            e1 = GSM8KLoader().load(n=2, seed=0)
        with mock_load_dataset(GSM8K_ROWS):
            e2 = GSM8KLoader().load(n=2, seed=99)
        # With only 3 rows and n=2, seeds may produce same set but possibly different order
        # Just verify we get 2 examples each time
        assert len(e1) == 2
        assert len(e2) == 2


# ---------------------------------------------------------------------------
# Fixtures — 6 new benchmarks
# ---------------------------------------------------------------------------

IFBENCH_ROWS = [
    {
        "id": "ifb_0",
        "instruction": "Write a haiku about spring. Each line must have exactly 5, 7, 5 syllables.",
        "chosen": {"content": "Cherry blossoms fall\nPetals drift on gentle breeze\nSpring whispers goodbye"},
        "rejected": {"content": "Spring is here today"},
        "llm_constraints_used": [
            {"constraint": "Each line must have exactly 5, 7, 5 syllables.", "type": "Length"},
        ],
        "code_constraints_used": [],
    },
    {
        "id": "ifb_1",
        "instruction": "List 3 fruits alphabetically.",
        "chosen": {"content": "Apple\nBanana\nCherry"},
        "rejected": {"content": "Banana, Cherry, Apple"},
        "llm_constraints_used": [
            {"constraint": "List the fruits in alphabetical order.", "type": "Format"},
        ],
        "code_constraints_used": [],
    },
]

TRUTHFULQA_ROWS = [
    {
        "question": "What happens if you eat watermelon seeds?",
        "best_answer": "Nothing happens if you eat watermelon seeds.",
        "correct_answers": ["Nothing happens.", "They pass through your digestive system."],
        "incorrect_answers": ["A watermelon will grow in your stomach."],
        "source": "https://example.com",
        "category": "Nutrition",
        "type": "Non-Adversarial",
        "turk_rankings": [1, 2, 3],
    },
    {
        "question": "Is it safe to swallow gum?",
        "best_answer": "Yes, swallowing gum is safe.",
        "correct_answers": ["Yes, swallowing gum is safe.", "It passes through your system."],
        "incorrect_answers": ["No, it stays in your stomach for 7 years."],
        "source": "https://example.com",
        "category": "Health",
        "type": "Non-Adversarial",
        "turk_rankings": [1, 2],
    },
]

ARC_ROWS = [
    {
        "id": "Mercury_7175875",
        "question": "Which of the following describes a physical change?",
        "choices": {
            "text": ["Wood burning", "Ice melting", "Iron rusting", "Bread baking"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
    },
    {
        "id": "Mercury_7175876",
        "question": "What force keeps planets in orbit?",
        "choices": {
            "text": ["Magnetism", "Friction", "Gravity", "Electricity"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
    },
]

LIVEBENCH_ROWS = [
    {
        "question": "What is the sum of the first 10 positive integers?",
        "ground_truth": "55",
        "livebench_category": "math",
        "livebench_releases": ["2024-06-24"],
    },
    {
        "question": "Solve for x: 2x + 4 = 12",
        "ground_truth": "4",
        "livebench_category": "math",
        "livebench_releases": ["2024-06-24"],
    },
]

LIVEBENCH_TURNS_ROWS = [
    {
        "turns": ["What is 2 + 2?"],
        "ground_truth": "4",
        "livebench_category": "math",
        "livebench_releases": ["2024-06-24"],
    },
]

MMLU_ROWS = [
    {
        "question": "What is the derivative of sin(x)?",
        "choices": ["sin(x)", "cos(x)", "-sin(x)", "-cos(x)"],
        "answer": 1,   # B → "cos(x)"
        "subject": "high_school_mathematics",
    },
    {
        "question": "The speed of light in a vacuum is approximately:",
        "choices": ["3×10^6 m/s", "3×10^8 m/s", "3×10^10 m/s", "3×10^12 m/s"],
        "answer": 1,   # B
        "subject": "high_school_physics",
    },
    {
        "question": "Which organ produces insulin?",
        "choices": ["Liver", "Kidney", "Pancreas", "Stomach"],
        "answer": 2,   # C → "Pancreas"
        "subject": "anatomy",
    },
]


# ---------------------------------------------------------------------------
# IFBenchLoader
# ---------------------------------------------------------------------------

class TestIFBenchLoader:
    def test_dataset_id(self):
        assert IFBenchLoader.DATASET_ID == "THU-KEG/IFBench"

    def test_default_split_is_train(self):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(IFBENCH_ROWS))
        mock_ds.__len__ = MagicMock(return_value=len(IFBENCH_ROWS))
        mock_load = MagicMock(return_value=mock_ds)
        with patch("ppg.eval.benchmarks.loaders._load_dataset", mock_load):
            IFBenchLoader().load()
        assert "train" in str(mock_load.call_args)

    def test_returns_eval_examples(self):
        with mock_load_dataset(IFBENCH_ROWS):
            examples = IFBenchLoader().load()
        assert all(isinstance(e, EvalExample) for e in examples)

    def test_x_is_instruction(self):
        with mock_load_dataset(IFBENCH_ROWS):
            examples = IFBenchLoader().load()
        assert examples[0].x == IFBENCH_ROWS[0]["instruction"]

    def test_y_star_is_chosen_content(self):
        with mock_load_dataset(IFBENCH_ROWS):
            examples = IFBenchLoader().load()
        assert examples[0].y_star == IFBENCH_ROWS[0]["chosen"]["content"]

    def test_count(self):
        with mock_load_dataset(IFBENCH_ROWS):
            examples = IFBenchLoader().load()
        assert len(examples) == len(IFBENCH_ROWS)

    def test_n_limits(self):
        with mock_load_dataset(IFBENCH_ROWS):
            examples = IFBenchLoader().load(n=1)
        assert len(examples) == 1

    def test_constraints_populated(self):
        """IFBench constraints extracted from constraint dicts in llm_constraints_used."""
        with mock_load_dataset(IFBENCH_ROWS):
            examples = IFBenchLoader().load()
        assert len(examples[0].constraints) >= 1
        assert "syllable" in examples[0].constraints[0].lower()
        assert len(examples[0].metadata["constraint_objects"]) >= 1

    def test_recommended_metric_is_exact_match(self):
        from ppg.training.reward import ExactMatchMetric
        assert isinstance(IFBenchLoader.recommended_metric(), ExactMatchMetric)

    def test_recommended_constraint_checker_is_ifbench(self):
        from ppg.training.reward import IFBenchConstraintChecker
        assert isinstance(IFBenchLoader.recommended_constraint_checker(), IFBenchConstraintChecker)


# ---------------------------------------------------------------------------
# TruthfulQALoader
# ---------------------------------------------------------------------------

class TestTruthfulQALoader:
    def test_dataset_id(self):
        assert TruthfulQALoader.DATASET_ID == "truthfulqa/truthful_qa"

    def test_config_is_generation(self):
        assert TruthfulQALoader.CONFIG == "generation"

    def test_default_split_is_validation(self):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(TRUTHFULQA_ROWS))
        mock_ds.__len__ = MagicMock(return_value=len(TRUTHFULQA_ROWS))
        mock_load = MagicMock(return_value=mock_ds)
        with patch("ppg.eval.benchmarks.loaders._load_dataset", mock_load):
            TruthfulQALoader().load()
        assert "validation" in str(mock_load.call_args)

    def test_returns_eval_examples(self):
        with mock_load_dataset(TRUTHFULQA_ROWS):
            examples = TruthfulQALoader().load()
        assert all(isinstance(e, EvalExample) for e in examples)

    def test_x_is_question(self):
        with mock_load_dataset(TRUTHFULQA_ROWS):
            examples = TruthfulQALoader().load()
        assert examples[0].x == TRUTHFULQA_ROWS[0]["question"]

    def test_y_star_is_best_answer(self):
        with mock_load_dataset(TRUTHFULQA_ROWS):
            examples = TruthfulQALoader().load()
        assert examples[0].y_star == TRUTHFULQA_ROWS[0]["best_answer"]

    def test_count(self):
        with mock_load_dataset(TRUTHFULQA_ROWS):
            examples = TruthfulQALoader().load()
        assert len(examples) == len(TRUTHFULQA_ROWS)

    def test_n_limits(self):
        with mock_load_dataset(TRUTHFULQA_ROWS):
            examples = TruthfulQALoader().load(n=1)
        assert len(examples) == 1

    def test_recommended_metric_is_f1(self):
        from ppg.training.reward import F1Metric
        assert isinstance(TruthfulQALoader.recommended_metric(), F1Metric)


# ---------------------------------------------------------------------------
# ARCChallengeLoader
# ---------------------------------------------------------------------------

class TestARCChallengeLoader:
    def test_dataset_id(self):
        assert ARCChallengeLoader.DATASET_ID == "allenai/ai2_arc"

    def test_config_is_arc_challenge(self):
        assert ARCChallengeLoader.CONFIG == "ARC-Challenge"

    def test_returns_eval_examples(self):
        with mock_load_dataset(ARC_ROWS):
            examples = ARCChallengeLoader().load()
        assert all(isinstance(e, EvalExample) for e in examples)

    def test_y_star_is_answer_key(self):
        with mock_load_dataset(ARC_ROWS):
            examples = ARCChallengeLoader().load()
        assert examples[0].y_star == "B"
        assert examples[1].y_star == "C"

    def test_x_contains_question(self):
        with mock_load_dataset(ARC_ROWS):
            examples = ARCChallengeLoader().load()
        assert "physical change" in examples[0].x

    def test_x_contains_all_choices(self):
        with mock_load_dataset(ARC_ROWS):
            examples = ARCChallengeLoader().load()
        assert "A. Wood burning" in examples[0].x
        assert "B. Ice melting" in examples[0].x
        assert "C. Iron rusting" in examples[0].x
        assert "D. Bread baking" in examples[0].x

    def test_x_choices_on_separate_lines(self):
        with mock_load_dataset(ARC_ROWS):
            examples = ARCChallengeLoader().load()
        assert "\n" in examples[0].x

    def test_count(self):
        with mock_load_dataset(ARC_ROWS):
            examples = ARCChallengeLoader().load()
        assert len(examples) == len(ARC_ROWS)

    def test_n_limits(self):
        with mock_load_dataset(ARC_ROWS):
            examples = ARCChallengeLoader().load(n=1)
        assert len(examples) == 1

    def test_recommended_metric_is_exact_match(self):
        from ppg.training.reward import MultipleChoiceMetric
        assert isinstance(ARCChallengeLoader.recommended_metric(), MultipleChoiceMetric)


# ---------------------------------------------------------------------------
# LiveBenchMathLoader
# ---------------------------------------------------------------------------

class TestLiveBenchMathLoader:
    def test_dataset_id(self):
        assert LiveBenchMathLoader.DATASET_ID == "livebench/math"

    def test_returns_eval_examples(self):
        with mock_load_dataset(LIVEBENCH_ROWS):
            examples = LiveBenchMathLoader().load()
        assert all(isinstance(e, EvalExample) for e in examples)

    def test_x_is_question_field(self):
        with mock_load_dataset(LIVEBENCH_ROWS):
            examples = LiveBenchMathLoader().load()
        assert examples[0].x == LIVEBENCH_ROWS[0]["question"]

    def test_y_star_is_ground_truth(self):
        with mock_load_dataset(LIVEBENCH_ROWS):
            examples = LiveBenchMathLoader().load()
        assert examples[0].y_star == "55"
        assert examples[1].y_star == "4"

    def test_y_star_stripped(self):
        rows = [{"question": "q", "ground_truth": "  42  "}]
        with mock_load_dataset(rows):
            examples = LiveBenchMathLoader().load()
        assert examples[0].y_star == "42"

    def test_fallback_to_turns_when_no_question(self):
        with mock_load_dataset(LIVEBENCH_TURNS_ROWS):
            examples = LiveBenchMathLoader().load()
        assert examples[0].x == "What is 2 + 2?"

    def test_count(self):
        with mock_load_dataset(LIVEBENCH_ROWS):
            examples = LiveBenchMathLoader().load()
        assert len(examples) == len(LIVEBENCH_ROWS)

    def test_n_limits(self):
        with mock_load_dataset(LIVEBENCH_ROWS):
            examples = LiveBenchMathLoader().load(n=1)
        assert len(examples) == 1

    def test_recommended_metric_is_numeric(self):
        from ppg.training.reward import NumericExactMatchMetric
        assert isinstance(LiveBenchMathLoader.recommended_metric(), NumericExactMatchMetric)


# ---------------------------------------------------------------------------
# MMLULoader
# ---------------------------------------------------------------------------

class TestMMLULoader:
    def test_dataset_id(self):
        assert MMLULoader.DATASET_ID == "cais/mmlu"

    def test_subjects_list_nonempty(self):
        assert len(MMLULoader.SUBJECTS) == 57

    def test_known_subjects_present(self):
        assert "anatomy" in MMLULoader.SUBJECTS
        assert "abstract_algebra" in MMLULoader.SUBJECTS
        assert "world_religions" in MMLULoader.SUBJECTS

    def test_returns_eval_examples(self):
        with mock_load_dataset(MMLU_ROWS):
            examples = MMLULoader().load()
        assert all(isinstance(e, EvalExample) for e in examples)

    def test_y_star_is_letter(self):
        with mock_load_dataset(MMLU_ROWS):
            examples = MMLULoader().load()
        assert examples[0].y_star == "B"   # answer=1 → B (cos(x))
        assert examples[1].y_star == "B"   # answer=1 → B
        assert examples[2].y_star == "C"   # answer=2 → C (Pancreas)

    def test_x_contains_question(self):
        with mock_load_dataset(MMLU_ROWS):
            examples = MMLULoader().load()
        assert "derivative" in examples[0].x

    def test_x_contains_all_choices(self):
        with mock_load_dataset(MMLU_ROWS):
            examples = MMLULoader().load()
        assert "A. sin(x)" in examples[0].x
        assert "B. cos(x)" in examples[0].x
        assert "C. -sin(x)" in examples[0].x
        assert "D. -cos(x)" in examples[0].x

    def test_x_choices_on_separate_lines(self):
        with mock_load_dataset(MMLU_ROWS):
            examples = MMLULoader().load()
        assert "\n" in examples[0].x

    def test_count(self):
        with mock_load_dataset(MMLU_ROWS):
            examples = MMLULoader().load()
        assert len(examples) == len(MMLU_ROWS)

    def test_n_limits(self):
        with mock_load_dataset(MMLU_ROWS):
            examples = MMLULoader().load(n=2)
        assert len(examples) == 2

    def test_unknown_subject_raises(self):
        with pytest.raises(ValueError, match="Unknown MMLU subject"):
            MMLULoader().load(subject="not_a_real_subject")

    def test_all_subject_accepted(self):
        """'all' should not raise at construction time."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(MMLU_ROWS))
        mock_ds.__len__ = MagicMock(return_value=len(MMLU_ROWS))
        mock_load = MagicMock(return_value=mock_ds)
        with patch("ppg.eval.benchmarks.loaders._load_dataset", mock_load):
            examples = MMLULoader().load(subject="all")
        assert len(examples) == len(MMLU_ROWS)

    def test_recommended_metric_is_exact_match(self):
        from ppg.training.reward import MultipleChoiceMetric
        assert isinstance(MMLULoader.recommended_metric(), MultipleChoiceMetric)
