"""
Matched-budget evaluation harness for PPG.

Runs PPG against 6 baselines on a held-out test set; all share the same
LMClient and TaskMetric to ensure fair comparison.

"Matched budget" means each baseline makes ≤ the same number of LM calls
per example as PPG-fast (= 1). Baselines that would require more calls are
either approximated or marked N/A.

Baselines
---------
flat_all       : Concatenate ALL graph nodes in topological order → 1 LM call.
                 No routing; maximum context length.
static_best    : Fixed path supplied at construction (e.g. found by offline
                 exhaustive scoring on a held-out val set). 1 LM call.
random_gating  : RandomSelector routing. 1 LM call.
highest_utility: HighestUtilitySelector routing (greedy on learned utility
                 scores, no UCB). 1 LM call.
miprov2        : External integration — compile MIPROv2Baseline (ppg/eval/external.py),
                 pass as external_baselines={"miprov2": baseline}. Requires dspy-ai.
gepa           : External integration — compile GEPABaseline (ppg/eval/external.py),
                 pass as external_baselines={"gepa": baseline}. Requires gepa.

Usage
-----
    harness = EvalHarness(
        executor=trained_executor,
        metric=ExactMatchMetric(),
        lm=lm_client,
        config=EvalConfig(baselines=["flat_all", "random_gating"]),
    )
    report = harness.evaluate(test_examples)
    print(report.comparison_table())
"""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ppg.core.executor import (
    HighestUtilitySelector,
    LMClient,
    PathTrace,
    PPGExecutor,
    PromptAssembler,
    RandomSelector,
)
from ppg.core.graph import PPGraph
from ppg.training.reward import ConstraintChecker, TaskMetric


# ---------------------------------------------------------------------------
# Progress bar helpers
# ---------------------------------------------------------------------------

def _make_eval_bar(iterable, *, desc: str, enabled: bool, total: int):
    """Wrap eval loop with tqdm if enabled."""
    try:
        from tqdm import tqdm
    except ImportError:
        if enabled:
            raise ImportError("tqdm required for progress display: pip install tqdm") from None
        return iterable
    return tqdm(iterable, desc=desc, total=total, disable=not enabled,
                unit="ex", ncols=100, leave=True)


def _make_manual_bar(*, desc: str, enabled: bool, total: int):
    if not enabled:
        return None
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="ex", ncols=100, leave=True)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalExample:
    x:           str
    y_star:      str
    constraints: list[str] = field(default_factory=list)
    metadata:    dict      = field(default_factory=dict)


@dataclass
class BaselineMetrics:
    """Per-baseline aggregate metrics over the test set."""
    name:              str
    task_scores:       list[float]
    token_counts:      list[int]
    constraint_scores: list[float]   # empty when no ConstraintChecker used
    lm_calls:          int           # total LM calls made across all examples

    @property
    def task_accuracy(self) -> float:
        return statistics.mean(self.task_scores) if self.task_scores else 0.0

    @property
    def mean_tokens(self) -> float:
        return statistics.mean(self.token_counts) if self.token_counts else 0.0

    @property
    def std_task(self) -> float:
        return statistics.stdev(self.task_scores) if len(self.task_scores) > 1 else 0.0

    @property
    def mean_constraint(self) -> float:
        return statistics.mean(self.constraint_scores) if self.constraint_scores else 0.0

    def as_dict(self) -> dict:
        return {
            "name":            self.name,
            "task_accuracy":   round(self.task_accuracy, 4),
            "std_task":        round(self.std_task, 4),
            "mean_tokens":     round(self.mean_tokens, 1),
            "mean_constraint": round(self.mean_constraint, 4),
            "lm_calls":        self.lm_calls,
            "n_examples":      len(self.task_scores),
        }


@dataclass
class EvalReport:
    """Full evaluation report: PPG vs all baselines."""
    ppg:       BaselineMetrics
    baselines: dict[str, BaselineMetrics]

    def all_metrics(self) -> list[BaselineMetrics]:
        return [self.ppg] + list(self.baselines.values())

    def comparison_table(self) -> list[dict]:
        """
        Returns list of per-system dicts sorted by task_accuracy descending.
        Suitable for tabulate(), pandas.DataFrame(), or direct printing.
        """
        rows = [m.as_dict() for m in self.all_metrics()]
        return sorted(rows, key=lambda r: r["task_accuracy"], reverse=True)

    def winner(self) -> str:
        """Name of system with highest task_accuracy."""
        return max(self.all_metrics(), key=lambda m: m.task_accuracy).name

    def ppg_delta(self, baseline_name: str) -> float:
        """task_accuracy(PPG) - task_accuracy(baseline)."""
        return self.ppg.task_accuracy - self.baselines[baseline_name].task_accuracy


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------

SUPPORTED_BASELINES = frozenset({
    "flat_all",
    "static_best",
    "random_gating",
    "highest_utility",
    "miprov2",
    "gepa",
})


@dataclass
class EvalConfig:
    baselines: list[str] = field(default_factory=lambda: [
        "flat_all",
        "static_best",
        "random_gating",
        "highest_utility",
    ])
    # Fixed path for static_best (topological order of node IDs).
    # If None, static_best uses topological sort of the graph.
    static_best_path: Optional[list[str]] = None
    seed: int = 0

    # Progress display
    show_progress: bool = True

    # Print a markdown summary table after each method finishes evaluating.
    show_method_tables: bool = True

    # Name of the split being evaluated (shown in per-method tables).
    split_name: str = "test"

    # Parallel eval workers — LM calls are I/O-bound so threads scale well.
    n_workers: int = 1

    def __post_init__(self):
        unknown = set(self.baselines) - SUPPORTED_BASELINES
        if unknown:
            raise ValueError(f"Unknown baselines: {unknown}. "
                             f"Supported: {sorted(SUPPORTED_BASELINES)}")


# ---------------------------------------------------------------------------
# Baseline runners (internal helpers)
# ---------------------------------------------------------------------------

class _FlatAllRunner:
    """All fragments concatenated in topological order."""

    def __init__(self, graph: PPGraph, lm: LMClient):
        self._graph   = graph
        self._lm      = lm
        self._all_ids = self._topological_order()
        self._asm     = PromptAssembler(graph)

    def _topological_order(self) -> list[str]:
        """Kahn's algorithm on the frozen graph."""
        in_deg = {n: 0 for n in self._graph.nodes}
        for src, dst in self._graph.edges:
            in_deg[dst] += 1
        queue = sorted([n for n, d in in_deg.items() if d == 0])
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            nbrs = sorted(dst for (s, dst) in self._graph.edges if s == node)
            for nbr in nbrs:
                in_deg[nbr] -= 1
                if in_deg[nbr] == 0:
                    queue.append(nbr)
        return order

    def run(self, example: EvalExample) -> tuple[str, int]:
        from ppg.core.tokenizer import count_tokens
        prompt   = self._asm.assemble(self._all_ids, {"input": example.x})
        response = self._lm.complete(prompt)
        return response, count_tokens(prompt)


class _StaticBestRunner:
    """Fixed path (given at construction)."""

    def __init__(self, graph: PPGraph, lm: LMClient, path: list[str]):
        self._lm     = lm
        self._path   = path
        self._asm    = PromptAssembler(graph)

    def run(self, example: EvalExample) -> tuple[str, int]:
        from ppg.core.tokenizer import count_tokens
        prompt   = self._asm.assemble(self._path, {"input": example.x})
        response = self._lm.complete(prompt)
        return response, count_tokens(prompt)


class _ExecutorRunner:
    """Wraps PPGExecutor with a swapped selector."""

    def __init__(self, executor: PPGExecutor, selector):
        self._executor = executor
        self._selector = selector

    def run(self, example: EvalExample) -> tuple[str, int]:
        original = self._executor.selector
        self._executor.selector = self._selector
        try:
            trace    = self._executor.execute(example.x, train_mode=False)
            response = trace.lm_response
            tokens   = trace.token_count
        finally:
            self._executor.selector = original
        return response, tokens


# ---------------------------------------------------------------------------
# EvalHarness
# ---------------------------------------------------------------------------

class EvalHarness:
    """
    Evaluates PPG and a configurable set of baselines on a test dataset.

    Parameters
    ----------
    executor            : trained PPGExecutor (its policy is the PPG system under eval)
    metric              : TaskMetric used to score responses
    lm                  : LMClient shared by all baselines
    config              : EvalConfig controlling which baselines to run
    constraint_checker  : optional ConstraintChecker for IFEval/IFBench examples;
                          when provided, examples with non-empty constraints are
                          scored via checker.check() instead of metric.score()
    external_baselines  : pre-compiled external baselines keyed by name
                          (e.g. {"miprov2": MIPROv2Baseline(...), "gepa": GEPABaseline(...)})
                          Each value must implement run(EvalExample) -> (str, int).
                          Required to use "miprov2" or "gepa" in EvalConfig.baselines.
    """

    def __init__(
        self,
        executor:            PPGExecutor,
        metric:              TaskMetric,
        lm:                  LMClient,
        config:              Optional[EvalConfig] = None,
        constraint_checker:  Optional[ConstraintChecker] = None,
        external_baselines:  Optional[dict] = None,
    ):
        self._executor  = executor
        self._metric    = metric
        self._lm        = lm
        self._cfg       = config or EvalConfig()
        self._checker   = constraint_checker
        self._external  = external_baselines or {}
        self._rng       = np.random.default_rng(self._cfg.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, examples: list[EvalExample]) -> EvalReport:
        """
        Run PPG and all configured baselines on examples.
        Returns EvalReport with per-system BaselineMetrics.
        """
        if not examples:
            raise ValueError("examples must be non-empty")

        ppg_metrics  = self.evaluate_one("ppg", examples)
        base_metrics = {}
        for name in self._cfg.baselines:
            base_metrics[name] = self.evaluate_one(name, examples)

        return EvalReport(ppg=ppg_metrics, baselines=base_metrics)

    def evaluate_one(self, name: str, examples: list[EvalExample]) -> BaselineMetrics:
        """
        Run and score exactly one method, print its per-method table, return metrics.

        name : "ppg" or any baseline name in SUPPORTED_BASELINES
        """
        if not examples:
            raise ValueError("examples must be non-empty")

        m = self._run_ppg(examples) if name == "ppg" else self._run_baseline(name, examples)
        self._print_method_table(name, m)
        return m

    def register_external(self, name: str, baseline) -> None:
        """Register a pre-compiled external baseline so it can be used in evaluate_one()."""
        self._external[name] = baseline

    def _print_method_table(self, name: str, m: "BaselineMetrics") -> None:
        """Print a rich summary table for one method after its eval run."""
        if not self._cfg.show_method_tables:
            return
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich import box
        except ImportError:
            split = self._cfg.split_name
            print(f"\n  {name}")
            print(f"  | Split   | Score | StdDev  | AvgTok  | API calls |")
            print(f"  |---------|-------|---------|---------|-----------|")
            print(f"  | {split:<7} | {m.task_accuracy:.3f} | {m.std_task:.3f}   "
                  f"| {m.mean_tokens:>7.1f} | {m.lm_calls:>9} |")
            return

        is_ppg = name == "ppg"
        split  = self._cfg.split_name

        table = Table(box=box.ROUNDED, show_lines=False, header_style="bold cyan",
                      padding=(0, 1), show_header=True)
        table.add_column("Split",     style="bold",    no_wrap=True)
        table.add_column("Score",     justify="right", min_width=7)
        table.add_column("StdDev",    justify="right", min_width=7)
        table.add_column("AvgTok",    justify="right", min_width=7)
        table.add_column("API calls", justify="right", min_width=9)

        score_str = (f"[bold green]{m.task_accuracy:.3f}[/bold green]"
                     if is_ppg else f"{m.task_accuracy:.3f}")
        table.add_row(split, score_str, f"{m.std_task:.3f}",
                      f"{m.mean_tokens:.1f}", str(m.lm_calls))

        title_style = "bold cyan" if is_ppg else "bold white"
        Console().print(
            Panel(table,
                  title=f"[{title_style}]{name}[/{title_style}]",
                  expand=False),
            end="\n",
        )

    # ------------------------------------------------------------------
    # Parallel execution core
    # ------------------------------------------------------------------

    def _run_examples(
        self,
        examples: list[EvalExample],
        call_fn:  Callable[[EvalExample], tuple[str, int]],
        desc:     str,
    ) -> tuple[list[float], list[int], list[float]]:
        """
        Evaluate call_fn over examples, sequential or parallel based on n_workers.

        call_fn(example) -> (response, token_count)

        Returns (task_scores, token_counts, constraint_scores).
        Results are in the same order as examples.
        """
        n = len(examples)
        n_workers = self._cfg.n_workers

        if n_workers <= 1:
            return self._run_sequential(examples, call_fn, desc)

        # Parallel: submit all examples, collect in order
        bar = _make_manual_bar(desc=desc, enabled=self._cfg.show_progress, total=n)
        scores, tokens, cscores = [], [], []

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(call_fn, ex) for ex in examples]
            for ex, fut in zip(examples, futures):
                response, t = fut.result()
                scores.append(self._score_example(response, ex))
                tokens.append(t)
                c = self._constraint_score(response, ex)
                if c is not None:
                    cscores.append(c)
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(acc=f"{sum(scores)/len(scores):.3f}")

        if bar is not None:
            bar.close()

        return scores, tokens, cscores

    def _run_sequential(
        self,
        examples: list[EvalExample],
        call_fn:  Callable[[EvalExample], tuple[str, int]],
        desc:     str,
    ) -> tuple[list[float], list[int], list[float]]:
        bar = _make_eval_bar(examples, desc=desc,
                             enabled=self._cfg.show_progress, total=len(examples))
        scores, tokens, cscores = [], [], []
        for ex in bar:
            response, t = call_fn(ex)
            scores.append(self._score_example(response, ex))
            tokens.append(t)
            c = self._constraint_score(response, ex)
            if c is not None:
                cscores.append(c)
            if self._cfg.show_progress and hasattr(bar, "set_postfix"):
                bar.set_postfix(acc=f"{sum(scores)/len(scores):.3f}")
        return scores, tokens, cscores

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_example(self, prediction: str, example: EvalExample) -> float:
        """
        Dispatch to the appropriate scoring method:
        - MBPP: score_with_tests() when metadata has 'test_list'
        - IFEval/IFBench: constraint_checker.check() when checker provided and constraints non-empty
        - Default: metric.score(prediction, y_star)
        """
        test_list = example.metadata.get("test_list") if example.metadata else None
        if test_list and hasattr(self._metric, "score_with_tests"):
            return self._metric.score_with_tests(prediction, test_list)
        if example.constraints and self._checker is not None:
            return self._checker.check(prediction, example.constraints, example.metadata or {})
        return self._metric.score(prediction, example.y_star)

    def _constraint_score(self, prediction: str, example: EvalExample) -> Optional[float]:
        """Returns constraint satisfaction score when checker + constraints present, else None."""
        if example.constraints and self._checker is not None:
            return self._checker.check(prediction, example.constraints, example.metadata or {})
        return None

    # ------------------------------------------------------------------
    # PPG evaluation
    # ------------------------------------------------------------------

    def _run_ppg(self, examples: list[EvalExample]) -> BaselineMetrics:
        executor = self._executor

        def _call(ex: EvalExample) -> tuple[str, int]:
            trace = executor.execute(ex.x, train_mode=False)
            return trace.lm_response, trace.token_count

        scores, tokens, cscores = self._run_examples(
            examples, _call, desc="eval ppg      "
        )
        return BaselineMetrics(
            name="ppg",
            task_scores=scores,
            token_counts=tokens,
            constraint_scores=cscores,
            lm_calls=len(examples),
        )

    # ------------------------------------------------------------------
    # Baseline dispatch
    # ------------------------------------------------------------------

    def _run_baseline(
        self, name: str, examples: list[EvalExample]
    ) -> BaselineMetrics:
        if name == "flat_all":
            return self._run_flat_all(examples)
        if name == "static_best":
            return self._run_static_best(examples)
        if name == "random_gating":
            return self._run_executor_baseline(
                "random_gating",
                RandomSelector(seed=self._cfg.seed),
                examples,
            )
        if name == "highest_utility":
            return self._run_executor_baseline(
                "highest_utility",
                HighestUtilitySelector(self._executor.graph),
                examples,
            )
        if name in self._external:
            return self._run_external(name, self._external[name], examples)
        if name == "miprov2":
            raise NotImplementedError(
                "MIPROv2 baseline requires a pre-compiled MIPROv2Baseline passed as "
                "external_baselines={'miprov2': baseline} to EvalHarness. "
                "See ppg/eval/external.py — requires: pip install dspy-ai"
            )
        if name == "gepa":
            raise NotImplementedError(
                "GEPA baseline requires a pre-compiled GEPABaseline passed as "
                "external_baselines={'gepa': baseline} to EvalHarness. "
                "See ppg/eval/external.py — requires: pip install gepa"
            )
        raise ValueError(f"Unknown baseline: {name}")

    # ------------------------------------------------------------------
    # Baseline runners
    # ------------------------------------------------------------------

    def _run_flat_all(self, examples: list[EvalExample]) -> BaselineMetrics:
        runner = _FlatAllRunner(self._executor.graph, self._lm)

        scores, tokens, cscores = self._run_examples(
            examples, runner.run, desc="eval flat_all "
        )
        return BaselineMetrics(
            name="flat_all",
            task_scores=scores,
            token_counts=tokens,
            constraint_scores=cscores,
            lm_calls=len(examples),
        )

    def _run_static_best(self, examples: list[EvalExample]) -> BaselineMetrics:
        path = self._cfg.static_best_path
        if path is None:
            path = self._best_utility_path()
        runner = _StaticBestRunner(self._executor.graph, self._lm, path)

        scores, tokens, cscores = self._run_examples(
            examples, runner.run, desc="eval static   "
        )
        return BaselineMetrics(
            name="static_best",
            task_scores=scores,
            token_counts=tokens,
            constraint_scores=cscores,
            lm_calls=len(examples),
        )

    def _run_executor_baseline(
        self,
        name:     str,
        selector,
        examples: list[EvalExample],
    ) -> BaselineMetrics:
        executor = self._executor
        original = executor.selector
        executor.selector = selector  # set once before threads start

        def _call(ex: EvalExample) -> tuple[str, int]:
            trace = executor.execute(ex.x, train_mode=False)
            return trace.lm_response, trace.token_count

        try:
            scores, tokens, cscores = self._run_examples(
                examples, _call, desc=f"eval {name[:10]:<10}"
            )
        finally:
            executor.selector = original  # always restore

        return BaselineMetrics(
            name=name,
            task_scores=scores,
            token_counts=tokens,
            constraint_scores=cscores,
            lm_calls=len(examples),
        )

    def _run_external(
        self,
        name:     str,
        baseline,
        examples: list[EvalExample],
    ) -> BaselineMetrics:
        """Run a pre-compiled external baseline (MIPROv2Baseline, GEPABaseline, etc.)."""
        scores, tokens, cscores = self._run_examples(
            examples, baseline.run, desc=f"eval {name[:10]:<10}"
        )
        return BaselineMetrics(
            name=name,
            task_scores=scores,
            token_counts=tokens,
            constraint_scores=cscores,
            lm_calls=len(examples),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_utility_path(self) -> list[str]:
        """
        Greedy highest-utility path: at each step, pick the successor
        with the highest fragment utility score.
        Falls back to first successor when all utilities are equal (init state).
        """
        graph = self._executor.graph
        sources = list(graph.source_ids)
        current = min(sources)  # deterministic tie-break
        path = [current]
        visited = {current}

        while current not in graph.terminal_ids:
            successors = [
                dst for (src, dst) in graph.edges
                if src == current and dst not in visited
            ]
            if not successors:
                break
            current = max(
                successors,
                key=lambda nid: graph.nodes[nid].utility,
            )
            path.append(current)
            visited.add(current)

        return path
