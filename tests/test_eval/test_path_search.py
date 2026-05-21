"""Tests for ppg/eval/path_search.py."""

from __future__ import annotations

import pytest

from ppg.core import FragmentType, PPGraphBuilder
from ppg.eval.harness import EvalExample
from ppg.eval.path_search import ranked_paths, score_path, select_path_by_validation
from ppg.training.reward import ExactMatchMetric


class PromptAwareLM:
    def __init__(self):
        self.n_calls = 0

    def complete(self, prompt: str) -> str:
        self.n_calls += 1
        return "42" if "good route" in prompt else "wrong"


class ComplementaryRouteLM:
    def complete(self, prompt: str) -> str:
        if "route a" in prompt:
            return {
                "q0": "a0",
                "q1": "a1",
                "q2": "wrong-a",
                "q3": "a3",
            }[_question_id(prompt)]
        if "route b" in prompt:
            return {
                "q0": "a0",
                "q1": "wrong-b",
                "q2": "a2",
                "q3": "a3",
            }[_question_id(prompt)]
        if "route c" in prompt:
            return {
                "q0": "wrong-c",
                "q1": "a1",
                "q2": "a2",
                "q3": "a3",
            }[_question_id(prompt)]
        return "wrong"


def _question_id(prompt: str) -> str:
    for qid in ("q0", "q1", "q2", "q3"):
        if qid in prompt:
            return qid
    raise AssertionError(f"missing qid in prompt: {prompt}")


def make_branching_graph():
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING, "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "bad route")
    b.add_fragment(FragmentType.REASONING_STYLE, "good route")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    tf, bad, good, oc = b.node_ids()
    b.connect(tf, bad)
    b.connect(tf, good)
    b.connect(bad, oc)
    b.connect(good, oc)
    graph = b.build()
    graph.nodes[bad].utility = 1.0
    graph.nodes[good].utility = 0.0
    return graph, [tf, bad, good, oc]


def make_complementary_graph():
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING, "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "route a")
    b.add_fragment(FragmentType.REASONING_STYLE, "route b")
    b.add_fragment(FragmentType.REASONING_STYLE, "route c")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    tf, route_a, route_b, route_c, oc = ids
    for route in (route_a, route_b, route_c):
        b.connect(tf, route)
        b.connect(route, oc)
    return b.build(), ids


def make_examples(n=3):
    return [EvalExample(x=f"q{i}", y_star="42") for i in range(n)]


def make_complementary_examples():
    return [
        EvalExample(x="q0", y_star="a0"),
        EvalExample(x="q1", y_star="a1"),
        EvalExample(x="q2", y_star="a2"),
        EvalExample(x="q3", y_star="a3"),
    ]


def test_score_path_scores_fixed_route():
    graph, ids = make_branching_graph()
    tf, _bad, good, oc = ids
    lm = PromptAwareLM()

    score, mean_tokens = score_path(
        graph, [tf, good, oc], make_examples(2), lm, ExactMatchMetric()
    )

    assert score == pytest.approx(1.0)
    assert mean_tokens > 0
    assert lm.n_calls == 2


def test_validation_search_can_override_utility_ranking():
    graph, ids = make_branching_graph()
    tf, _bad, good, oc = ids
    lm = PromptAwareLM()

    result = select_path_by_validation(
        graph, make_examples(3), lm, ExactMatchMetric(), show_progress=False
    )

    assert result.path == [tf, good, oc]
    assert result.val_score == pytest.approx(1.0)
    assert result.n_paths_scored == 2
    assert result.total_paths == 2
    assert result.candidates[0].path == [tf, good, oc]


def test_validation_search_returns_top_candidates():
    graph, ids = make_branching_graph()
    tf, bad, good, oc = ids
    lm = PromptAwareLM()

    result = select_path_by_validation(
        graph,
        make_examples(3),
        lm,
        ExactMatchMetric(),
        show_progress=False,
        return_top_k=2,
    )

    assert [c.path for c in result.candidates] == [
        [tf, good, oc],
        [tf, bad, oc],
    ]


def test_validation_search_scores_path_ensemble():
    graph, _ids = make_complementary_graph()

    result = select_path_by_validation(
        graph,
        make_complementary_examples(),
        ComplementaryRouteLM(),
        ExactMatchMetric(),
        show_progress=False,
        early_stop_patience=0,
        return_top_k=3,
    )

    assert result.val_score == pytest.approx(0.75)
    assert result.ensemble_val_score == pytest.approx(1.0)
    assert len(result.candidates) == 3


def test_ranked_paths_respects_candidate_cap():
    graph, ids = make_branching_graph()
    _tf, bad, _good, _oc = ids

    paths = ranked_paths(graph, max_candidates=1)

    assert len(paths) == 1
    assert bad in paths[0]


def test_validation_search_rejects_empty_examples():
    graph, _ids = make_branching_graph()
    with pytest.raises(ValueError, match="examples"):
        select_path_by_validation(graph, [], PromptAwareLM(), ExactMatchMetric())
