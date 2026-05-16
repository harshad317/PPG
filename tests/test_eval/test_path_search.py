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


def make_examples(n=3):
    return [EvalExample(x=f"q{i}", y_star="42") for i in range(n)]


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
