"""
Ablation study framework for PPG.

Each ablation disables exactly one component of the full PPG system so the
paper can measure its individual contribution.

Ablations defined here
----------------------
ppg_full        : Complete PPG (bandit + credit + variance penalty + rich graph)
no_credit       : LOO credit assignment disabled (p_ablate=0)
no_variance     : Perturbation variance penalty disabled (λ_v=0)
no_bandit       : LinUCB replaced by random routing throughout training
lean_topology   : Rich graph replaced by lean 3-node chain (no optional branches)
no_domain_primer: Domain primer node removed; lean TASK_FRAMING→RS→OC only
skip_variance   : Alias for no_variance (used in reward cost analysis)

Paper mapping (Table 3)
-----------------------
Row                       | Ablation name
-----------------------------------------------------------
PPG (full system)         | ppg_full
− LOO credit              | no_credit
− variance penalty        | no_variance
− LinUCB (random routing) | no_bandit
− rich topology           | lean_topology

Usage
-----
    study = AblationStudy(
        lm=lm_client,
        train_dataset=train_examples,
        test_dataset=test_examples,
        benchmark="gsm8k",
        ablations=["ppg_full", "no_credit", "no_variance"],
    )
    report = study.run()
    print(report.table())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _print_ablation_table(results: list) -> None:
    """Print a rich table summarising completed ablations. No-op if rich missing."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        return

    full_acc = None
    for r in results:
        if r.config.name == "ppg_full":
            full_acc = r.metrics.task_accuracy
            break

    table = Table(
        title="Ablation Study — results so far",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("Ablation",   style="bold",  no_wrap=True, min_width=16)
    table.add_column("Task Acc",   justify="right", min_width=9)
    table.add_column("Δ vs Full",  justify="right", min_width=10)
    table.add_column("Avg Tokens", justify="right", min_width=11)
    table.add_column("LM Calls",   justify="right", min_width=9)
    table.add_column("Description", style="dim")

    sorted_results = sorted(results, key=lambda r: r.metrics.task_accuracy, reverse=True)
    for r in sorted_results:
        acc   = r.metrics.task_accuracy
        delta = ""
        if full_acc is not None and r.config.name != "ppg_full":
            d     = full_acc - acc
            sign  = "+" if d >= 0 else ""
            color = "green" if d >= 0 else "red"
            delta = f"[{color}]{sign}{d:.4f}[/{color}]"
        elif r.config.name == "ppg_full":
            delta = "[dim]  —[/dim]"

        acc_str = f"[bold green]{acc:.4f}[/bold green]" if r.config.name == "ppg_full" else f"{acc:.4f}"
        table.add_row(
            r.config.name,
            acc_str,
            delta,
            f"{r.metrics.mean_tokens:.1f}",
            str(r.metrics.lm_calls),
            r.config.description,
        )

    Console().print(table)

from ppg.bandits.linucb import LinUCBPolicy
from ppg.core import ExecutorConfig, FeatureExtractor, PPGExecutor
from ppg.core.executor import LMClient, PromptAssembler, RandomSelector
from ppg.core.graph import PPGraph
from ppg.data.fragments import build_graph
from ppg.eval.harness import BaselineMetrics, EvalExample, EvalHarness, EvalConfig
from ppg.training.credit import CreditAssigner, CreditAssignerConfig
from ppg.training.reward import ConstraintChecker, RewardComputer, RewardConfig, TaskMetric
from ppg.training.trainer import PPGTrainer, TrainerConfig, TrainingExample


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    Specification for one ablation run.

    Each flag disables exactly one PPG component relative to ppg_full.
    Only one flag should be True per config (otherwise it's a compound ablation).
    """
    name:        str
    description: str

    # Training modifications
    p_ablate:       float = 0.15    # set 0.0 to disable LOO credit
    skip_variance:  bool  = False   # set True to disable variance penalty
    use_random:     bool  = False   # replace LinUCB with RandomSelector throughout
    topology:       str   = "rich"  # "lean" removes optional branches

    # Derived helpers
    @property
    def credit_disabled(self) -> bool:
        return self.p_ablate == 0.0

    @property
    def bandit_disabled(self) -> bool:
        return self.use_random


# ---------------------------------------------------------------------------
# Named ablation registry
# ---------------------------------------------------------------------------

ABLATIONS: dict[str, AblationConfig] = {
    "ppg_full": AblationConfig(
        name="ppg_full",
        description="Complete PPG: LinUCB + LOO credit + variance penalty + rich graph",
        p_ablate=0.15,
        skip_variance=False,
        use_random=False,
        topology="rich",
    ),
    "no_credit": AblationConfig(
        name="no_credit",
        description="PPG without LOO credit assignment (p_ablate=0)",
        p_ablate=0.0,
        skip_variance=False,
        use_random=False,
        topology="rich",
    ),
    "no_variance": AblationConfig(
        name="no_variance",
        description="PPG without perturbation variance penalty (λ_v=0)",
        p_ablate=0.15,
        skip_variance=True,
        use_random=False,
        topology="rich",
    ),
    "no_bandit": AblationConfig(
        name="no_bandit",
        description="PPG with random routing instead of LinUCB throughout",
        p_ablate=0.15,
        skip_variance=False,
        use_random=True,
        topology="rich",
    ),
    "lean_topology": AblationConfig(
        name="lean_topology",
        description="PPG with lean graph (TASK_FRAMING → RS → OUTPUT_CONTRACT only)",
        p_ablate=0.15,
        skip_variance=False,
        use_random=False,
        topology="lean",
    ),
}


def available_ablations() -> list[str]:
    return sorted(ABLATIONS.keys())


# ---------------------------------------------------------------------------
# AblationResult
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    config:  AblationConfig
    metrics: BaselineMetrics

    def as_dict(self) -> dict:
        d = self.metrics.as_dict()
        d["description"] = self.config.description
        return d


@dataclass
class AblationReport:
    results: list[AblationResult]

    def table(self) -> list[dict]:
        """Rows sorted by task_accuracy descending."""
        return sorted(
            [r.as_dict() for r in self.results],
            key=lambda d: d["task_accuracy"],
            reverse=True,
        )

    def get(self, ablation_name: str) -> Optional[AblationResult]:
        for r in self.results:
            if r.config.name == ablation_name:
                return r
        return None

    def delta_vs_full(self, ablation_name: str) -> Optional[float]:
        """
        task_accuracy(ppg_full) - task_accuracy(ablation).
        Positive = full PPG is better; negative = ablation is surprisingly better.
        Returns None if ppg_full not in results.
        """
        full = self.get("ppg_full")
        other = self.get(ablation_name)
        if full is None or other is None:
            return None
        return full.metrics.task_accuracy - other.metrics.task_accuracy

    def winner(self) -> str:
        return max(self.results, key=lambda r: r.metrics.task_accuracy).config.name


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------

def build_ablation_components(
    config:             AblationConfig,
    lm:                 LMClient,
    metric:             TaskMetric,
    graph:              Optional[PPGraph] = None,
    benchmark:          str = "gsm8k",
    trainer_cfg:        Optional[TrainerConfig] = None,
    constraint_checker: Optional[ConstraintChecker] = None,
) -> tuple[PPGExecutor, LinUCBPolicy, RewardComputer, CreditAssigner, PPGTrainer]:
    """
    Build all PPG components for one ablation.

    Parameters
    ----------
    config      : AblationConfig specifying which components to disable
    lm          : LMClient shared across all components
    metric      : TaskMetric for reward and evaluation
    graph       : PPGraph to use; if None, built from benchmark + config.topology
    benchmark   : used to build graph when graph is None
    trainer_cfg : TrainerConfig; if None, a short default is used

    Returns
    -------
    (executor, policy, reward_computer, credit_assigner, trainer)
    """
    if graph is None:
        graph = build_graph(benchmark, topology=config.topology)

    assembler = PromptAssembler(graph)
    policy    = LinUCBPolicy(graph)

    # Selector: bandit (default) or random (no_bandit ablation)
    selector = RandomSelector() if config.use_random else policy

    executor = PPGExecutor(
        graph=graph,
        selector=selector,
        lm=lm,
        feature_extractor=FeatureExtractor(),
        config=ExecutorConfig(escalation_enabled=False),
    )

    reward = RewardComputer(
        task_metric=metric,
        lm=lm,
        assembler=assembler,
        constraint_checker=constraint_checker,
        config=RewardConfig(
            skip_variance=config.skip_variance,
            constraint_as_task=benchmark in ("ifeval", "ifbench"),
        ),
    )

    credit = CreditAssigner(
        lm=lm,
        assembler=assembler,
        task_metric=metric,
        constraint_checker=constraint_checker,
        constraint_as_task=benchmark in ("ifeval", "ifbench"),
        config=CreditAssignerConfig(p_ablate=config.p_ablate),
    )

    cfg = trainer_cfg or TrainerConfig(
        n_warmup_episodes=50,
        n_train_episodes=200,
        n_finetune_episodes=50,
    )

    trainer = PPGTrainer(
        executor=executor,
        policy=policy,
        reward_computer=reward,
        credit_assigner=credit,
        config=cfg,
    )

    return executor, policy, reward, credit, trainer


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------

class AblationStudy:
    """
    Trains and evaluates PPG under each ablation configuration.

    Each ablation is trained independently from scratch on train_dataset,
    then evaluated on test_dataset with the same metric and LM.

    Parameters
    ----------
    lm            : LMClient used for all training and evaluation
    metric        : TaskMetric (e.g., NumericExactMatchMetric for GSM8K)
    train_dataset : list of TrainingExample for training each ablation
    test_dataset  : list of EvalExample for evaluation
    benchmark     : passed to build_graph when no graph is provided
    ablations     : names from ABLATIONS registry; None = all ablations
    trainer_cfg   : shared TrainerConfig; None = short default (fast CI)
    graphs        : optional per-ablation pre-built graphs; None = auto-build
    on_ablation   : optional callback(name, AblationResult) after each run
    show_progress : print rich summary table after each ablation completes
    """

    def __init__(
        self,
        lm:                 LMClient,
        metric:             TaskMetric,
        train_dataset:      list[TrainingExample],
        test_dataset:       list[EvalExample],
        benchmark:          str = "gsm8k",
        ablations:          Optional[list[str]] = None,
        trainer_cfg:        Optional[TrainerConfig] = None,
        graphs:             Optional[dict[str, PPGraph]] = None,
        on_ablation:        Optional[object] = None,
        show_progress:      bool = True,
        constraint_checker: Optional[ConstraintChecker] = None,
    ):
        self._lm                 = lm
        self._metric             = metric
        self._train              = train_dataset
        self._test               = test_dataset
        self._benchmark          = benchmark
        self._ablation_names     = ablations or list(ABLATIONS.keys())
        self._trainer_cfg        = trainer_cfg
        self._graphs             = graphs or {}
        self._on_ablation        = on_ablation
        self._show_progress      = show_progress
        self._constraint_checker = constraint_checker

        unknown = set(self._ablation_names) - set(ABLATIONS)
        if unknown:
            raise ValueError(
                f"Unknown ablations: {unknown}. "
                f"Available: {sorted(ABLATIONS.keys())}"
            )

    def run(self) -> AblationReport:
        """
        Run all ablations sequentially. Returns AblationReport.
        Training order: ablation list order (deterministic).
        After each ablation, prints a rich summary table when show_progress=True.
        """
        if not self._train:
            raise ValueError("train_dataset must be non-empty")
        if not self._test:
            raise ValueError("test_dataset must be non-empty")

        results = []
        n = len(self._ablation_names)
        for i, name in enumerate(self._ablation_names, 1):
            if self._show_progress:
                try:
                    from rich.console import Console
                    Console().print(
                        f"\n[bold cyan]▶ Ablation {i}/{n}: {name}[/bold cyan]"
                    )
                except ImportError:
                    print(f"\n[{i}/{n}] Running ablation: {name}")

            config = ABLATIONS[name]
            result = self._run_one(config)
            results.append(result)

            if self._show_progress:
                _print_ablation_table(results)

            if self._on_ablation is not None:
                self._on_ablation(name, result)

        return AblationReport(results=results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_one(self, config: AblationConfig) -> AblationResult:
        from dataclasses import replace as _dc_replace

        graph = self._graphs.get(config.name)

        # Propagate show_progress into trainer_cfg
        trainer_cfg = self._trainer_cfg
        if trainer_cfg is None:
            trainer_cfg = TrainerConfig(
                n_warmup_episodes=50,
                n_train_episodes=200,
                n_finetune_episodes=50,
                show_progress=self._show_progress,
            )
        elif trainer_cfg.show_progress != self._show_progress:
            trainer_cfg = _dc_replace(trainer_cfg, show_progress=self._show_progress)

        executor, policy, reward, credit, trainer = build_ablation_components(
            config=config,
            lm=self._lm,
            metric=self._metric,
            graph=graph,
            benchmark=self._benchmark,
            trainer_cfg=trainer_cfg,
            constraint_checker=self._constraint_checker,
        )

        # Train
        trainer.train(self._train)

        # Evaluate (no baselines — just this system)
        harness = EvalHarness(
            executor=executor,
            metric=self._metric,
            lm=self._lm,
            config=EvalConfig(baselines=[], show_progress=self._show_progress),
            constraint_checker=self._constraint_checker,
        )
        report = harness.evaluate(self._test)

        # Rename PPG metrics to the ablation name for clarity
        metrics = BaselineMetrics(
            name=config.name,
            task_scores=report.ppg.task_scores,
            token_counts=report.ppg.token_counts,
            constraint_scores=report.ppg.constraint_scores,
            lm_calls=report.ppg.lm_calls,
        )

        return AblationResult(config=config, metrics=metrics)
