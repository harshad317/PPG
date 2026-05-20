"""Tests for ppg/data/fragments.py — seed fragment library."""

from __future__ import annotations

import pytest

from ppg.core.graph import FragmentType, PPGraph
from ppg.data.fragments import (
    FRAGMENTS,
    available_benchmarks,
    build_graph,
    fragment_count,
    list_fragments,
)


BENCHMARKS = ["gsm8k", "ifbench", "hotpotqa", "mbpp", "truthfulqa", "arc_challenge", "livebench_math"]

REQUIRED_TYPES = {FragmentType.TASK_FRAMING, FragmentType.OUTPUT_CONTRACT}


# ---------------------------------------------------------------------------
# FRAGMENTS dict structure
# ---------------------------------------------------------------------------

class TestFragmentsDict:
    def test_all_benchmarks_present(self):
        for bm in BENCHMARKS:
            assert bm in FRAGMENTS

    def test_each_benchmark_has_task_framing(self):
        for bm in BENCHMARKS:
            assert "task_framing" in FRAGMENTS[bm]
            assert len(FRAGMENTS[bm]["task_framing"]) >= 1

    def test_each_benchmark_has_reasoning_style(self):
        for bm in BENCHMARKS:
            assert "reasoning_style" in FRAGMENTS[bm]

    def test_each_benchmark_has_output_contract(self):
        for bm in BENCHMARKS:
            assert "output_contract" in FRAGMENTS[bm]

    def test_task_framing_contains_input_placeholder(self):
        """Every task_framing template must contain {input}."""
        for bm in BENCHMARKS:
            for template in FRAGMENTS[bm]["task_framing"]:
                assert "{input}" in template, (
                    f"{bm}/task_framing template missing {{input}}: {template!r}"
                )

    def test_non_task_framing_no_input_placeholder(self):
        """Non-task-framing templates should NOT contain {input}."""
        for bm in BENCHMARKS:
            for ftype, templates in FRAGMENTS[bm].items():
                if ftype == "task_framing":
                    continue
                for tmpl in templates:
                    assert "{input}" not in tmpl, (
                        f"{bm}/{ftype} contains unexpected {{input}}: {tmpl!r}"
                    )

    def test_no_empty_templates(self):
        for bm in BENCHMARKS:
            for ftype, templates in FRAGMENTS[bm].items():
                for tmpl in templates:
                    assert tmpl.strip(), f"{bm}/{ftype} has empty template"

    def test_all_template_types_are_valid_fragment_types(self):
        valid_types = {ft.value for ft in FragmentType}
        for bm in BENCHMARKS:
            for ftype in FRAGMENTS[bm]:
                assert ftype in valid_types, (
                    f"{bm} uses unknown fragment type {ftype!r}"
                )


# ---------------------------------------------------------------------------
# available_benchmarks
# ---------------------------------------------------------------------------

class TestAvailableBenchmarks:
    def test_returns_list(self):
        result = available_benchmarks()
        assert isinstance(result, list)

    def test_includes_all_benchmarks(self):
        result = available_benchmarks()
        for bm in BENCHMARKS:
            assert bm in result

    def test_sorted(self):
        result = available_benchmarks()
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# fragment_count
# ---------------------------------------------------------------------------

class TestFragmentCount:
    def test_returns_dict(self):
        assert isinstance(fragment_count("gsm8k"), dict)

    def test_keys_are_fragment_types(self):
        counts = fragment_count("gsm8k")
        valid  = {ft.value for ft in FragmentType}
        for k in counts:
            assert k in valid

    def test_values_are_positive_ints(self):
        for bm in BENCHMARKS:
            counts = fragment_count(bm)
            for k, v in counts.items():
                assert isinstance(v, int) and v >= 1, f"{bm}/{k}: count={v}"

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            fragment_count("nonexistent")

    def test_gsm8k_has_multiple_reasoning_variants(self):
        counts = fragment_count("gsm8k")
        assert counts.get("reasoning_style", 0) >= 2


# ---------------------------------------------------------------------------
# list_fragments
# ---------------------------------------------------------------------------

class TestListFragments:
    def test_returns_list_of_strings(self):
        result = list_fragments("gsm8k", "task_framing")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            list_fragments("bad_bm", "task_framing")

    def test_unknown_type_returns_empty(self):
        result = list_fragments("gsm8k", "nonexistent_type")
        assert result == []

    def test_returns_all_variants(self):
        result = list_fragments("gsm8k", "reasoning_style")
        assert len(result) == fragment_count("gsm8k")["reasoning_style"]


# ---------------------------------------------------------------------------
# build_graph — lean topology
# ---------------------------------------------------------------------------

class TestBuildGraphLean:
    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_returns_ppgraph(self, bm):
        g = build_graph(bm, topology="lean")
        assert isinstance(g, PPGraph)

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_lean_has_exactly_3_nodes(self, bm):
        g = build_graph(bm, topology="lean")
        assert len(g.nodes) == 3

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_lean_has_exactly_2_edges(self, bm):
        g = build_graph(bm, topology="lean")
        assert len(g.edges) == 2

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_lean_has_one_source(self, bm):
        g = build_graph(bm, topology="lean")
        assert len(g.source_ids) == 1

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_lean_has_one_terminal(self, bm):
        g = build_graph(bm, topology="lean")
        assert len(g.terminal_ids) == 1

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_lean_source_is_task_framing(self, bm):
        g   = build_graph(bm, topology="lean")
        src = list(g.source_ids)[0]
        assert g.nodes[src].type == FragmentType.TASK_FRAMING

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_lean_terminal_is_output_contract(self, bm):
        g   = build_graph(bm, topology="lean")
        trm = list(g.terminal_ids)[0]
        assert g.nodes[trm].type == FragmentType.OUTPUT_CONTRACT

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_lean_task_framing_template_has_input_placeholder(self, bm):
        g   = build_graph(bm, topology="lean")
        src = list(g.source_ids)[0]
        assert "{input}" in g.nodes[src].template

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_lean_graph_validates(self, bm):
        """build_graph runs GraphValidator internally; if it passes, graph is valid."""
        g = build_graph(bm, topology="lean")
        assert g is not None

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            build_graph("unknown_bm", topology="lean")

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            build_graph("gsm8k", topology="ultra")  # type: ignore


# ---------------------------------------------------------------------------
# build_graph — rich topology
# ---------------------------------------------------------------------------

class TestBuildGraphRich:
    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_returns_ppgraph(self, bm):
        g = build_graph(bm, topology="rich")
        assert isinstance(g, PPGraph)

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_rich_has_more_nodes_than_lean(self, bm):
        lean = build_graph(bm, topology="lean")
        rich = build_graph(bm, topology="rich")
        assert len(rich.nodes) > len(lean.nodes)

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_rich_has_required_types_on_every_path(self, bm):
        """Every source→terminal path must cover TASK_FRAMING and OUTPUT_CONTRACT."""
        g = build_graph(bm, topology="rich")
        sources   = g.source_ids
        terminals = g.terminal_ids

        def dfs(node, visited, path_types):
            if node in visited:
                return True
            visited = visited | {node}
            path_types = path_types | {g.nodes[node].type}
            if node in terminals:
                return REQUIRED_TYPES.issubset(path_types)
            successors = [dst for (src, dst) in g.edges if src == node]
            if not successors:
                return REQUIRED_TYPES.issubset(path_types)
            return all(dfs(s, visited, path_types) for s in successors)

        for src in sources:
            assert dfs(src, frozenset(), frozenset())

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_rich_has_branching(self, bm):
        """Rich topology should have at least one branching node (out-degree > 1)."""
        g = build_graph(bm, topology="rich")
        out_degrees = {}
        for (src, _) in g.edges:
            out_degrees[src] = out_degrees.get(src, 0) + 1
        assert any(d > 1 for d in out_degrees.values()), (
            f"{bm} rich graph has no branching node"
        )

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_rich_has_compression_node(self, bm):
        g = build_graph(bm, topology="rich")
        types = {n.type for n in g.nodes.values()}
        assert FragmentType.COMPRESSION in types

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_rich_has_domain_primer(self, bm):
        g = build_graph(bm, topology="rich")
        types = {n.type for n in g.nodes.values()}
        assert FragmentType.DOMAIN_PRIMER in types

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_rich_terminals_are_output_contracts(self, bm):
        """All terminal nodes in the rich graph are OUTPUT_CONTRACT variants."""
        g = build_graph(bm, topology="rich")
        assert len(g.terminal_ids) >= 1
        for tid in g.terminal_ids:
            assert g.nodes[tid].type == FragmentType.OUTPUT_CONTRACT


# ---------------------------------------------------------------------------
# build_graph — variant selection
# ---------------------------------------------------------------------------

class TestBuildGraphVariants:
    def test_variant_0_and_1_differ(self):
        g0 = build_graph("gsm8k", topology="lean", variant=0)
        g1 = build_graph("gsm8k", topology="lean", variant=1)
        # Different variant → different templates on at least one node
        templates_0 = {n.template for n in g0.nodes.values()}
        templates_1 = {n.template for n in g1.nodes.values()}
        # gsm8k has multiple reasoning_style variants → should differ
        assert templates_0 != templates_1

    def test_variant_wraps_around(self):
        """variant index wraps: variant=100 same as variant=100%len."""
        g0 = build_graph("gsm8k", topology="lean", variant=0)
        n  = len(FRAGMENTS["gsm8k"]["reasoning_style"])
        gn = build_graph("gsm8k", topology="lean", variant=n)
        # After wrap, reasoning_style should be same as variant=0
        def reasoning_template(g):
            for node in g.nodes.values():
                if node.type == FragmentType.REASONING_STYLE:
                    return node.template
        assert reasoning_template(g0) == reasoning_template(gn)


# ---------------------------------------------------------------------------
# Integration: build_graph + PPGExecutor
# ---------------------------------------------------------------------------

class TestIntegrationWithExecutor:
    def test_lean_graph_runs_executor(self):
        from ppg.bandits.linucb import LinUCBPolicy
        from ppg.core import ExecutorConfig, FeatureExtractor, PPGExecutor

        class FixedLM:
            def complete(self, prompt):
                return "42"

        g       = build_graph("gsm8k", topology="lean")
        policy  = LinUCBPolicy(g)
        executor = PPGExecutor(
            graph=g, selector=policy, lm=FixedLM(),
            feature_extractor=FeatureExtractor(),
            config=ExecutorConfig(escalation_enabled=False),
        )
        trace = executor.execute("What is 2+2?")
        assert trace.lm_response == "42"
        assert len(trace.node_ids) == 3

    def test_rich_graph_runs_executor(self):
        from ppg.bandits.linucb import LinUCBPolicy
        from ppg.core import ExecutorConfig, FeatureExtractor, PPGExecutor

        class FixedLM:
            def complete(self, prompt):
                return "#### 4"

        g       = build_graph("gsm8k", topology="rich")
        policy  = LinUCBPolicy(g)
        executor = PPGExecutor(
            graph=g, selector=policy, lm=FixedLM(),
            feature_extractor=FeatureExtractor(),
            config=ExecutorConfig(escalation_enabled=False),
        )
        trace = executor.execute("2+2=?")
        assert trace.lm_response == "#### 4"
        # Rich graph has >= 4 nodes in traversed path
        assert len(trace.node_ids) >= 3

    @pytest.mark.parametrize("bm", BENCHMARKS)
    def test_assembled_prompt_contains_input(self, bm):
        from ppg.bandits.linucb import LinUCBPolicy
        from ppg.core import ExecutorConfig, FeatureExtractor, PPGExecutor
        from ppg.core.executor import PromptAssembler

        class CaptureLM:
            def __init__(self):
                self.last_prompt = ""
            def complete(self, prompt):
                self.last_prompt = prompt
                return "ok"

        g       = build_graph(bm, topology="lean")
        policy  = LinUCBPolicy(g)
        lm      = CaptureLM()
        executor = PPGExecutor(
            graph=g, selector=policy, lm=lm,
            feature_extractor=FeatureExtractor(),
            config=ExecutorConfig(escalation_enabled=False),
        )
        question = "What is the capital of France?"
        executor.execute(question)
        assert question in lm.last_prompt, (
            f"{bm}: input not found in assembled prompt"
        )
