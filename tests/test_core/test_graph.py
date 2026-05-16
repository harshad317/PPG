"""Tests for ppg/core/graph.py."""

import json
import tempfile

import numpy as np
import pytest

from ppg.core import (
    FragmentType,
    FEATURE_DIM,
    Guard,
    PromptFragment,
    PPGraph,
    GraphValidator,
    PPGraphBuilder,
    REQUIRED_TYPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_linear_graph() -> tuple[PPGraph, list[str]]:
    """task_framing -> reasoning_style -> output_contract (linear chain)."""
    builder = PPGraphBuilder()
    builder.add_fragment(FragmentType.TASK_FRAMING,   "Solve: {input}")
    builder.add_fragment(FragmentType.REASONING_STYLE, "Think step by step.")
    builder.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = builder.node_ids()
    builder.connect_chain(*ids)
    return builder.build(), ids


def make_branching_graph() -> PPGraph:
    """
    task_framing -> reasoning_style  --|
                 -> compression       |--> output_contract
    """
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,   "Solve: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think step by step.")
    b.add_fragment(FragmentType.COMPRESSION,     "Be concise.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    tf, rs, comp, oc = ids
    b.connect(tf, rs)
    b.connect(tf, comp)
    b.connect(rs, oc)
    b.connect(comp, oc)
    return b.build()


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------

class TestGuard:
    def test_all_pass_fires_on_zero_phi(self):
        g = Guard.all_pass()
        phi = np.zeros(FEATURE_DIM)
        assert g.evaluate(phi) is True

    def test_all_pass_fires_on_any_phi(self):
        g = Guard.all_pass()
        rng = np.random.default_rng(0)
        for _ in range(20):
            phi = rng.standard_normal(FEATURE_DIM)
            assert g.evaluate(phi) is True

    def test_threshold_guard_fires_above_threshold(self):
        weights = np.zeros(FEATURE_DIM)
        weights[0] = 1.0  # gate on input_length_norm
        g = Guard(weights=weights, bias=0.5)
        phi_below = np.zeros(FEATURE_DIM)
        phi_below[0] = 0.3
        phi_above = np.zeros(FEATURE_DIM)
        phi_above[0] = 0.7
        assert g.evaluate(phi_below) is False
        assert g.evaluate(phi_above) is True

    def test_serialization_roundtrip(self):
        rng = np.random.default_rng(42)
        g = Guard(weights=rng.standard_normal(FEATURE_DIM), bias=0.3)
        g2 = Guard.from_dict(g.to_dict())
        assert np.allclose(g.weights, g2.weights)
        assert g.bias == g2.bias
        assert g.feature_names == g2.feature_names


# ---------------------------------------------------------------------------
# PromptFragment tests
# ---------------------------------------------------------------------------

class TestPromptFragment:
    def test_render_simple(self):
        f = PromptFragment.create(FragmentType.TASK_FRAMING, "Solve: {input}")
        assert f.render({"input": "2+2"}) == "Solve: 2+2"

    def test_render_missing_key_raises(self):
        f = PromptFragment.create(FragmentType.TASK_FRAMING, "Solve: {input} in {lang}")
        with pytest.raises(ValueError, match="missing key"):
            f.render({"input": "2+2"})

    def test_update_utility_online_mean(self):
        f = PromptFragment.create(FragmentType.REASONING_STYLE, "Think.")
        f.update_utility(1.0)
        f.update_utility(0.0)
        assert f.utility == pytest.approx(0.5)
        assert f.utility_n == 2

    def test_serialization_roundtrip(self):
        f = PromptFragment.create(FragmentType.VERIFICATION, "Check: {input}", author="test")
        f.update_utility(0.8)
        d = f.to_dict()
        f2 = PromptFragment.from_dict(d)
        assert f2.id == f.id
        assert f2.type == f.type
        assert f2.template == f.template
        assert f2.utility == pytest.approx(f.utility)
        assert f2.metadata == f.metadata


# ---------------------------------------------------------------------------
# PPGraphBuilder and PPGraph tests
# ---------------------------------------------------------------------------

class TestPPGraphBuilder:
    def test_builds_linear_graph(self):
        g, ids = make_linear_graph()
        assert len(g.nodes) == 3
        assert len(g.edges) == 2
        assert len(g.source_ids) == 1
        assert len(g.terminal_ids) == 1

    def test_source_and_terminal_correct(self):
        g, ids = make_linear_graph()
        tf_id, _, oc_id = ids
        assert tf_id in g.source_ids
        assert oc_id in g.terminal_ids

    def test_branching_graph_structure(self):
        g = make_branching_graph()
        assert len(g.nodes) == 4
        assert len(g.edges) == 4
        assert len(g.source_ids) == 1
        assert len(g.terminal_ids) == 1

    def test_missing_required_type_raises(self):
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.REASONING_STYLE, "Think.")
        b.add_fragment(FragmentType.VERIFICATION, "Check.")
        ids = b.node_ids()
        b.connect(*ids)
        with pytest.raises(ValueError, match="Required type"):
            b.build()

    def test_cycle_raises(self):
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING, "Solve: {input}")
        b.add_fragment(FragmentType.REASONING_STYLE, "Think.")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
        ids = b.node_ids()
        b.connect_chain(*ids)
        b.connect(ids[2], ids[1])  # create cycle: output_contract -> reasoning_style
        with pytest.raises(ValueError, match="cycle"):
            b.build()

    def test_unreachable_node_raises(self):
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING, "Solve: {input}")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
        b.add_fragment(FragmentType.VERIFICATION, "Orphan.")  # not connected
        ids = b.node_ids()
        b.connect(ids[0], ids[1])
        # ids[2] has no predecessor and is not a source (it has no outgoing either)
        # validator should catch it
        with pytest.raises(ValueError):
            b.build()


# ---------------------------------------------------------------------------
# PPGraph query tests
# ---------------------------------------------------------------------------

class TestPPGraphQueries:
    def test_successors(self):
        g, ids = make_linear_graph()
        tf, rs, oc = ids
        assert g.successors(tf) == [rs]
        assert g.successors(rs) == [oc]
        assert g.successors(oc) == []

    def test_predecessors(self):
        g, ids = make_linear_graph()
        tf, rs, oc = ids
        assert g.predecessors(tf) == []
        assert g.predecessors(rs) == [tf]

    def test_active_successors_all_pass(self):
        g, ids = make_linear_graph()
        tf = ids[0]
        phi = np.zeros(FEATURE_DIM)
        active = g.active_successors(tf, phi)
        assert active == [ids[1]]

    def test_active_successors_guard_blocks(self):
        """Replace default all-pass guard with a blocking one."""
        g, ids = make_linear_graph()
        tf, rs, oc = ids
        # Block the tf->rs edge
        blocking_guard = Guard(weights=np.ones(FEATURE_DIM), bias=1e9)
        g.edges[(tf, rs)] = blocking_guard
        phi = np.zeros(FEATURE_DIM)
        active = g.active_successors(tf, phi)
        assert active == []

    def test_path_enumeration_linear(self):
        g, ids = make_linear_graph()
        paths = list(g.all_paths())
        assert len(paths) == 1
        assert paths[0] == ids

    def test_path_enumeration_branching(self):
        g = make_branching_graph()
        paths = list(g.all_paths())
        assert len(paths) == 2

    def test_path_count(self):
        g = make_branching_graph()
        assert g.path_count() == 2

    def test_nodes_by_type(self):
        g, ids = make_linear_graph()
        tf_nodes = g.nodes_by_type(FragmentType.TASK_FRAMING)
        assert len(tf_nodes) == 1
        assert tf_nodes[0].type == FragmentType.TASK_FRAMING


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_dict_roundtrip_linear(self):
        g, _ = make_linear_graph()
        d = g.to_dict()
        g2 = PPGraph.from_dict(d)
        assert set(g2.nodes) == set(g.nodes)
        assert set(g2.edges) == set(g.edges)
        assert g2.source_ids == g.source_ids
        assert g2.terminal_ids == g.terminal_ids

    def test_json_roundtrip(self):
        g, _ = make_linear_graph()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        g.to_json(path)
        g2 = PPGraph.from_json(path)
        assert set(g2.nodes) == set(g.nodes)

    def test_guard_preserved_in_roundtrip(self):
        b = PPGraphBuilder()
        b.add_fragment(FragmentType.TASK_FRAMING, "Solve: {input}")
        b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
        ids = b.node_ids()
        custom_guard = Guard(weights=np.arange(FEATURE_DIM, dtype=float), bias=0.42)
        b.connect(ids[0], ids[1], guard=custom_guard)
        g = b.build()

        g2 = PPGraph.from_dict(g.to_dict())
        restored_guard = g2.edges[(ids[0], ids[1])]
        assert np.allclose(restored_guard.weights, custom_guard.weights)
        assert restored_guard.bias == pytest.approx(custom_guard.bias)


# ---------------------------------------------------------------------------
# GraphValidator edge cases
# ---------------------------------------------------------------------------

class TestGraphValidator:
    def test_valid_graph_no_errors(self):
        g, _ = make_linear_graph()
        errors = GraphValidator().validate(g)
        assert errors == []

    def test_invalid_edge_endpoint(self):
        g, ids = make_linear_graph()
        g.edges[("nonexistent", ids[1])] = Guard.all_pass()
        errors = GraphValidator().validate(g)
        assert any("nonexistent" in e for e in errors)
