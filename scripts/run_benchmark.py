#!/usr/bin/env python3
"""
PPG benchmark runner.

Each method runs independently. By default (no flags), trains PPG from scratch
and evaluates it alongside internal baselines. Use --run-mipro or --run-gepa
to run those methods standalone. Use --include-ppg to combine PPG with external
methods in a single run.

Methods
-------
ppg (default)  : PPG + internal baselines (flat_all, static_best, random_gating, highest_utility)
--run-mipro    : MIPROv2 (DSPy, auto='heavy') — standalone, no PPG training
--run-gepa     : GEPA (DSPy) — standalone, no PPG training
--include-ppg  : Add PPG training + eval when using --run-mipro/--run-gepa

Graph map (each benchmark has its own dedicated fragment set)
-------------------------------------------------------------
All 9 benchmarks now have dedicated fragment graphs.

Usage
-----
# PPG only (default):
python scripts/run_benchmark.py gsm8k --model gpt-4.1-mini --production

# MIPROv2 only:
python scripts/run_benchmark.py gsm8k --model gpt-4.1-mini --run-mipro

# GEPA only:
python scripts/run_benchmark.py gsm8k --model gpt-4.1-mini --run-gepa \\
    --reflection-model gpt-4o

# All three together:
python scripts/run_benchmark.py gsm8k --model gpt-4.1-mini --production \\
    --run-mipro --run-gepa --include-ppg --reflection-model gpt-4o

Environment variables
---------------------
OPENAI_API_KEY    : required for --provider openai (default)
ANTHROPIC_API_KEY : required for --provider anthropic
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Graph fallback map for benchmarks without dedicated fragment sets
# ---------------------------------------------------------------------------

_GRAPH_MAP: dict[str, str] = {
    "gsm8k":         "gsm8k",
    "ifbench":       "ifbench",
    "hotpotqa":      "hotpotqa",
    "drop":          "drop",
    "mbpp":          "mbpp",
    "truthfulqa":    "truthfulqa",
    "arc_challenge": "arc_challenge",
    "livebench_math":"livebench_math",
    "mmlu":          "mmlu",
}

SUPPORTED_BENCHMARKS = sorted(_GRAPH_MAP.keys())


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def load_splits(
    benchmark:    str,
    n_train:      int,
    n_val:        int,
    n_test:       int,
    seed:         int,
    mmlu_subject: str,
):
    """
    Returns (train_examples, val_examples, test_examples, metric, constraint_checker, objective).

    For benchmarks with a single HF split, performs a deterministic
    train/val/test partition. For benchmarks with separate HF splits, uses
    them directly and caps to n_train / n_val / n_test.
    """
    from ppg.eval.benchmarks.loaders import (
        ARCChallengeLoader, DROPLoader, GSM8KLoader,
        HotpotQALoader, IFBenchLoader, LiveBenchMathLoader,
        MBPPLoader, MMLULoader, TruthfulQALoader,
    )
    from ppg.eval.harness import EvalExample
    from ppg.training.reward import (
        ExactMatchMetric, F1Metric, NumericExactMatchMetric,
    )

    rng = random.Random(seed)

    def _split_single(examples, n_tr, n_v, n_te):
        """Partition a single-split dataset into train / val / test."""
        data = list(examples)
        rng.shuffle(data)
        tr = data[:n_tr]
        v  = data[n_tr:n_tr + n_v]
        te = data[n_tr + n_v:n_tr + n_v + n_te]
        return tr, v, te

    def _split_two(examples, n_v, n_te):
        """Partition a single held-out split into disjoint val / test slices."""
        data = list(examples)
        rng.shuffle(data)
        v  = data[:n_v]
        te = data[n_v:n_v + n_te]
        return v, te

    def _cap(lst, n):
        out = list(lst)
        rng.shuffle(out)
        return out[:n]

    constraint_checker = None
    objective = "Optimize the prompt to maximize task accuracy."

    # -----------------------------------------------------------------------
    if benchmark == "gsm8k":
        loader = GSM8KLoader()
        train_pool = loader.load("train", seed=seed)
        train, val, _ = _split_single(train_pool, n_train, n_val, 0)
        test   = _cap(loader.load("test",  seed=seed + 2), n_test)
        metric = loader.recommended_metric()
        objective = "Maximize exact-match accuracy on multi-step arithmetic word problems."

    # -----------------------------------------------------------------------
    elif benchmark == "ifbench":
        loader = IFBenchLoader()
        all_ex = loader.load("train", seed=seed)
        train, val, test = _split_single(all_ex, n_train, n_val, n_test)
        metric = loader.recommended_metric()
        constraint_checker = loader.recommended_constraint_checker()
        objective = (
            "Maximize constraint satisfaction — responses must include required "
            "keywords and satisfy all stated format constraints."
        )

    # -----------------------------------------------------------------------
    elif benchmark == "hotpotqa":
        loader = HotpotQALoader()
        train  = _cap(loader.load("train",      seed=seed),     n_train)
        heldout = loader.load("validation", seed=seed + 1)
        val, test = _split_two(heldout, n_val, n_test)
        metric = loader.recommended_metric()
        objective = "Maximize token-F1 on multi-hop reading-comprehension questions."

    # -----------------------------------------------------------------------
    elif benchmark == "drop":
        loader = DROPLoader()
        train  = _cap(loader.load("train",      seed=seed),     n_train)
        heldout = loader.load("validation", seed=seed + 1)
        val, test = _split_two(heldout, n_val, n_test)
        metric = loader.recommended_metric()
        objective = "Maximize F1 on discrete-reasoning passages (arithmetic, counting, sorting)."

    # -----------------------------------------------------------------------
    elif benchmark == "mbpp":
        loader = MBPPLoader()
        train  = _cap(loader.load("train", seed=seed),     n_train)
        val    = _cap(loader.load("validation", seed=seed + 1), n_val)
        test   = _cap(loader.load("test",  seed=seed + 2), n_test)
        metric = loader.recommended_metric()
        objective = "Generate correct Python functions that pass all provided test assertions."

    # -----------------------------------------------------------------------
    elif benchmark == "truthfulqa":
        loader = TruthfulQALoader()
        all_ex = loader.load("validation", seed=seed)
        train, val, test = _split_single(all_ex, n_train, n_val, n_test)
        metric = loader.recommended_metric()
        objective = "Maximize F1 against the best truthful reference answer."

    # -----------------------------------------------------------------------
    elif benchmark == "arc_challenge":
        loader = ARCChallengeLoader()
        train  = _cap(loader.load("train", seed=seed),     n_train)
        val    = _cap(loader.load("validation", seed=seed + 1), n_val)
        test   = _cap(loader.load("test",  seed=seed + 2), n_test)
        metric = loader.recommended_metric()
        objective = "Maximize exact-match accuracy on hard science multiple-choice questions."

    # -----------------------------------------------------------------------
    elif benchmark == "livebench_math":
        loader = LiveBenchMathLoader()
        all_ex = loader.load("test", seed=seed)
        train, val, test = _split_single(all_ex, n_train, n_val, n_test)
        metric = loader.recommended_metric()
        objective = "Maximize numeric exact-match accuracy on competition math problems."

    # -----------------------------------------------------------------------
    elif benchmark == "mmlu":
        loader = MMLULoader()
        # dev split is tiny (5 per subject); supplement with auxiliary_train when available.
        dev_ex  = loader.load(subject=mmlu_subject, split="dev",        seed=seed)
        val_ex  = loader.load(subject=mmlu_subject, split="validation", seed=seed)
        test_ex = loader.load(subject=mmlu_subject, split="test",       seed=seed)
        try:
            aux_ex = loader.load(subject=mmlu_subject, split="auxiliary_train", seed=seed)
        except Exception:
            aux_ex = []
        train_pool = dev_ex + aux_ex
        rng.shuffle(train_pool)
        train = train_pool[:n_train]
        val   = _cap(val_ex, n_val) if val_ex else _cap(test_ex, n_val)
        test  = _cap(test_ex, n_test)
        metric = loader.recommended_metric()
        objective = f"Maximize exact-match accuracy on MMLU subject: {mmlu_subject}."

    else:
        raise ValueError(f"Unsupported benchmark: {benchmark!r}. "
                         f"Choose from: {SUPPORTED_BENCHMARKS}")

    return train, val, test, metric, constraint_checker, objective


# ---------------------------------------------------------------------------
# Convert EvalExample → TrainingExample
# ---------------------------------------------------------------------------

def to_training(examples):
    from ppg.training.trainer import TrainingExample
    return [
        TrainingExample(x=ex.x, y_star=ex.y_star,
                        constraints=ex.constraints, metadata=ex.metadata or {})
        for ex in examples
    ]


# ---------------------------------------------------------------------------
# Build flat seed prompt for external optimizers
# ---------------------------------------------------------------------------

def build_seed_prompt(graph) -> str:
    """Topological-order flat concat of all fragment templates."""
    from ppg.core.executor import PromptAssembler
    in_deg = {n: 0 for n in graph.nodes}
    for src, dst in graph.edges:
        in_deg[dst] += 1
    queue = sorted([n for n, d in in_deg.items() if d == 0])
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for (s, d) in sorted(graph.edges):
            if s == node:
                in_deg[d] -= 1
                if in_deg[d] == 0:
                    queue.append(d)
    asm = PromptAssembler(graph)
    return asm.assemble(order, {"input": "<INPUT>"})


def build_best_path_prompt(graph) -> str:
    """Greedy highest-utility path assembled as a flat prompt template."""
    return build_path_prompt(graph, best_utility_path(graph))


def build_path_prompt(graph, path: list[str]) -> str:
    """Assemble a fixed path as a flat prompt template."""
    from ppg.core.executor import PromptAssembler
    asm = PromptAssembler(graph)
    return asm.assemble(path, {"input": "<INPUT>"})


def best_utility_path(graph) -> list[str]:
    """Greedy highest-utility source-to-terminal path."""
    sources  = list(graph.source_ids)
    current  = min(sources)
    path     = [current]
    visited  = {current}
    while current not in graph.terminal_ids:
        successors = [dst for (src, dst) in graph.edges
                      if src == current and dst not in visited]
        if not successors:
            break
        current = max(successors, key=lambda nid: graph.nodes[nid].utility)
        path.append(current)
        visited.add(current)
    return path


# ---------------------------------------------------------------------------
# DSPy call counter — reads dspy.LM.history (available in DSPy >= 2.4)
# ---------------------------------------------------------------------------

class _DSPyCounter:
    """
    Counts DSPy LM calls by reading dspy.LM.history.

    Satisfies the same reset()/call_count interface as CountingLMClient so it
    can be passed as lm_counter to harness.evaluate_splits().
    """
    def __init__(self, dspy_lm):
        self._lm   = dspy_lm
        self._mark = len(getattr(dspy_lm, "history", []))

    def reset(self) -> int:
        hist = getattr(self._lm, "history", [])
        n    = len(hist) - self._mark
        self._mark = len(hist)
        return n

    @property
    def call_count(self) -> int:
        return len(getattr(self._lm, "history", [])) - self._mark


# ---------------------------------------------------------------------------
# LM client factory
# ---------------------------------------------------------------------------

def make_lm(provider: str, model: str, cache_dir: str | None):
    if provider == "openai":
        from ppg.lm.clients import OpenAIClient, OpenAIConfig, DiskCachedLMClient
        lm = OpenAIClient(OpenAIConfig(model=model, temperature=0.0, max_tokens=2048))
    elif provider == "anthropic":
        from ppg.lm.clients import AnthropicClient, AnthropicConfig, DiskCachedLMClient
        lm = AnthropicClient(AnthropicConfig(model=model, temperature=0.0, max_tokens=2048))
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    if cache_dir:
        from ppg.lm.clients import DiskCachedLMClient
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{provider}_{model.replace('/', '_')}.json")
        lm = DiskCachedLMClient(lm, cache_path)

    return lm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train PPG and evaluate all baselines on a benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("benchmark", choices=SUPPORTED_BENCHMARKS,
                        help="Benchmark to run")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LM provider (default: openai)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Task LM model name (default: gpt-4o-mini)")
    parser.add_argument("--reflection-model", default="gpt-4o",
                        dest="reflection_model",
                        help="GEPA reflection LM — more capable model recommended (default: gpt-4o)")
    parser.add_argument("--train-n",  type=int, default=100,  dest="train_n",
                        help="Training examples (default: 100)")
    parser.add_argument("--val-n",   type=int, default=50,   dest="val_n",
                        help="Validation examples for MIPROv2/GEPA (default: 50)")
    parser.add_argument("--test-n",  type=int, default=500,   dest="test_n",
                        help="Test examples (default: 500)")
    parser.add_argument("--seed",    type=int, default=0,
                        help="Global random seed (default: 0)")
    parser.add_argument("--warmup",  type=int, default=200,
                        help="PPG warmup episodes (default: 200)")
    parser.add_argument("--train-ep",type=int, default=1000, dest="train_ep",
                        help="PPG train episodes (default: 1000)")
    parser.add_argument("--finetune",type=int, default=200,
                        help="PPG finetune episodes (default: 200)")
    parser.add_argument("--ppg-calibration", choices=["val_path", "dynamic", "none"],
                        default="val_path", dest="ppg_calibration",
                        help="Deployment calibration for PPG after training: "
                             "val_path = fixed best path from validation, "
                             "dynamic = greedy bandit routing per input, "
                             "none = no calibration (default: val_path)")
    parser.add_argument("--ppg-path-candidates", type=int, default=0,
                        dest="ppg_path_candidates",
                        help="Number of utility-ranked paths to validate for PPG; "
                             "0 searches all complete paths (default: 0)")
    parser.add_argument("--run-base",  action="store_true", dest="run_base",
                        help="Run base model eval only (raw input, no prompt engineering). Skips PPG unless --include-ppg.")
    parser.add_argument("--run-mipro", action="store_true", dest="run_mipro",
                        help="Run MIPROv2 only (requires dspy-ai). Skips PPG unless --include-ppg.")
    parser.add_argument("--run-gepa",  action="store_true", dest="run_gepa",
                        help="Run GEPA only (requires gepa). Skips PPG unless --include-ppg.")
    parser.add_argument("--include-ppg", action="store_true", dest="include_ppg",
                        help="Include PPG training + eval when using --run-mipro/--run-gepa")
    parser.add_argument("--gepa-calls", type=int, default=500, dest="gepa_calls",
                        help="GEPA max_metric_calls heavy budget (default: 500)")
    parser.add_argument("--mmlu-subject", default="all", dest="mmlu_subject",
                        help="MMLU subject (default: all)")
    parser.add_argument("--cache-dir",   default=".lm_cache", dest="cache_dir",
                        help="Disk cache dir for LM responses (default: .lm_cache)")
    parser.add_argument("--no-cache",    action="store_true", dest="no_cache",
                        help="Disable disk caching of LM calls")
    parser.add_argument("--output-dir",  default="results", dest="output_dir",
                        help="Directory to write JSON results (default: results/)")
    parser.add_argument("--checkpoint-dir", default=None, dest="checkpoint_dir",
                        help="Save policy checkpoints here during training")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel episode workers for PPG training (default: 1; "
                             "set to os.cpu_count() for max throughput)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress tqdm/rich progress bars")

    # --- Production mode: enables all new features ---
    parser.add_argument("--production", action="store_true",
                        help="Use production configs (Pareto reward, GRPO, reflection, "
                             "evolution, branching, semantic features, self-consistency)")
    parser.add_argument("--log-dir", default=None, dest="log_dir",
                        help="Structured logging directory (default: ppg_logs/<bench>_<timestamp>)")
    parser.add_argument("--enable-wandb", action="store_true", dest="enable_wandb",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--no-reflection", action="store_true", dest="no_reflection",
                        help="Disable reflection loop (even in --production)")
    parser.add_argument("--no-evolution", action="store_true", dest="no_evolution",
                        help="Disable fragment evolution (even in --production)")
    parser.add_argument("--no-branching", action="store_true", dest="no_branching",
                        help="Disable failure-mode branching (even in --production)")
    parser.add_argument("--no-pareto", action="store_true", dest="no_pareto",
                        help="Use scalarized reward instead of Pareto (even in --production)")
    parser.add_argument("--few-shot", action="store_true", dest="few_shot",
                        help="Include few-shot example fragments in the graph")
    parser.add_argument("--ppg-reflection-model", default=None, dest="ppg_reflection_model",
                        help="LM for PPG reflection/evolution (default: same as --model)")
    args = parser.parse_args()

    show_progress = not args.quiet
    cache_dir     = None if args.no_cache else args.cache_dir
    bench         = args.benchmark

    # Determine which methods to run.
    # --run-gepa / --run-mipro without --include-ppg → external only, skip PPG.
    # Default (no external flags) → PPG + internal baselines.
    has_external = args.run_mipro or args.run_gepa or args.run_base
    run_ppg = args.include_ppg or not has_external

    try:
        from rich.console import Console as _Console
        from rich.rule import Rule as _Rule
        _console = _Console()
        def _header(text: str) -> None:
            _console.rule(f"[bold]{text}[/bold]")
        def _step_rule(n: int, total: int, label: str) -> None:
            _console.rule(
                f"[cyan]\\[[bold]{n}/{total}[/bold]][/cyan] [white]{label}[/white]",
                style="dim",
            )
        def _info(text: str) -> None:
            _console.print(f"  [dim]{text}[/dim]")
    except ImportError:
        def _header(text: str) -> None:
            print(f"\n=== {text} ===\n")
        def _step_rule(n: int, total: int, label: str) -> None:
            print(f"[{n}/{total}] {label}")
        def _info(text: str) -> None:
            print(f"      {text}")

    _header(f"PPG Benchmark: {bench}  |  model: {args.model}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    _step_rule(1, 6, "Loading dataset splits...")
    train_ex, val_ex, test_ex, metric, constraint_checker, objective = load_splits(
        benchmark=bench,
        n_train=args.train_n,
        n_val=args.val_n,
        n_test=args.test_n,
        seed=args.seed,
        mmlu_subject=args.mmlu_subject,
    )
    _info(f"train={len(train_ex)}  val={len(val_ex)}  test={len(test_ex)}")
    train_examples = to_training(train_ex)

    # ------------------------------------------------------------------
    # 2. Build LM client
    # ------------------------------------------------------------------
    _step_rule(2, 6, "Building LM client...")
    from ppg.lm.clients import CountingLMClient
    lm = CountingLMClient(make_lm(args.provider, args.model, cache_dir))

    # ------------------------------------------------------------------
    # 3. Build graph (needed by all methods for seed prompt)
    # ------------------------------------------------------------------
    _step_rule(1, 4, "Building graph...")
    from ppg.data.fragments import build_graph
    from ppg.core.executor import PromptAssembler

    use_prod = args.production

    graph_key = _GRAPH_MAP[bench]
    graph     = build_graph(graph_key, topology="rich",
                            include_few_shot=args.few_shot)
    assembler = PromptAssembler(graph)

    constraint_as_task = bench == "ifbench"

    # --- Logger ---
    from ppg.logging_utils import PPGLogger, NullLogger, LogConfig
    if args.log_dir or (use_prod and run_ppg):
        log_ts = time.strftime("%Y%m%d_%H%M%S")
        log_dir = args.log_dir or f"ppg_logs/{bench}_{log_ts}"
        ppg_logger = PPGLogger(LogConfig(
            log_dir=log_dir,
            enable_wandb=args.enable_wandb,
            wandb_project="ppg",
            wandb_run_name=f"{bench}_{args.model}_{log_ts}",
        ))
        _info(f"logging → {log_dir}")
    else:
        ppg_logger = NullLogger()

    all_metrics:       dict[str, "BaselineMetrics"] = {}
    optimized_prompts: dict[str, str]               = {}
    _splits = {"train": train_ex, "val": val_ex, "test": test_ex}
    train_time = 0.0
    stats = None

    # ==================================================================
    # PPG path: train → calibrate → eval + internal baselines
    # ==================================================================
    if run_ppg:
        from ppg.bandits.linucb import LinUCBPolicy
        from ppg.core import ExecutorConfig, FeatureExtractor, PPGExecutor
        from ppg.training.credit import CreditAssigner, CreditAssignerConfig
        from ppg.training.reward import RewardComputer, RewardConfig
        from ppg.training.trainer import PPGTrainer, TrainerConfig

        _step_rule(2, 4, "Building PPG components...")
        policy    = LinUCBPolicy(graph)

        feat_extractor = FeatureExtractor.production() if use_prod else FeatureExtractor()
        exec_config    = ExecutorConfig.production() if use_prod else ExecutorConfig(escalation_enabled=False)
        executor  = PPGExecutor(
            graph=graph,
            selector=policy,
            lm=lm,
            feature_extractor=feat_extractor,
            config=exec_config,
        )

        # --- Reward (Pareto or scalarized) ---
        reward_cfg_base = dict(
            constraint_as_task=constraint_as_task,
            skip_variance=constraint_as_task,
        )
        if use_prod and not args.no_pareto:
            from ppg.training.reward import ParetoRewardComputer
            prod_cfg = RewardConfig.production(**reward_cfg_base)
            reward = ParetoRewardComputer(
                task_metric=metric,
                lm=lm,
                assembler=assembler,
                constraint_checker=constraint_checker,
                config=prod_cfg,
                logger=ppg_logger,
            )
            _info("reward: ParetoRewardComputer")
        else:
            rcfg = RewardConfig.production(**reward_cfg_base) if use_prod else RewardConfig(**reward_cfg_base)
            reward = RewardComputer(
                task_metric=metric,
                lm=lm,
                assembler=assembler,
                constraint_checker=constraint_checker,
                config=rcfg,
            )

        credit_cfg = CreditAssignerConfig()
        if use_prod:
            credit_cfg.skip_source = False
            credit_cfg.skip_terminal = False
            credit_cfg.p_ablate = 0.15

        # F1 benchmarks: use ExactMatch for credit assignment (cleaner 0/1 signal)
        credit_metric = None
        if bench in ("hotpotqa", "drop", "truthfulqa"):
            from ppg.training.reward import ExactMatchMetric
            credit_metric = ExactMatchMetric()

        credit = CreditAssigner(
            lm=lm,
            assembler=assembler,
            task_metric=metric,
            config=credit_cfg,
            constraint_checker=constraint_checker,
            constraint_as_task=constraint_as_task,
            credit_metric=credit_metric,
        )

        # --- Reflection / Evolution / Branching (production mode) ---
        reflection_loop = None
        evolver = None
        brancher = None

        if use_prod:
            refl_lm = lm
            if args.ppg_reflection_model:
                from ppg.lm.clients import CountingLMClient
                refl_lm = CountingLMClient(make_lm(args.provider, args.ppg_reflection_model, cache_dir))

            if not args.no_reflection:
                from ppg.training.reflection import ReflectionLoop, ReflectionConfig
                reflection_loop = ReflectionLoop(
                    lm=refl_lm,
                    config=ReflectionConfig(
                        enabled=True,
                        score_threshold=0.75,
                        reflect_fraction=0.5,
                    ),
                    constraint_checker=constraint_checker,
                )
                _info("reflection: enabled (threshold=0.75, fraction=0.5)")

            if not args.no_evolution:
                from ppg.training.evolution import FragmentEvolver, EvolutionConfig
                evolver = FragmentEvolver(
                    lm=refl_lm,
                    config=EvolutionConfig(enabled=True, evolve_every=500),
                    reflection=reflection_loop,
                    benchmark=bench,
                )
                _info("evolution: enabled (every 500 episodes)")

            if not args.no_branching and reflection_loop is not None:
                from ppg.training.branching import FailureModeBrancher, BranchingConfig
                brancher = FailureModeBrancher(
                    lm=refl_lm,
                    reflection=reflection_loop,
                    config=BranchingConfig(enabled=True, branch_every=500, min_reflections=20),
                )
                _info("branching: enabled (every 500 episodes, min 20 reflections)")

        # --- Trainer config ---
        if use_prod:
            prod_overrides = dict(
                n_warmup_episodes=args.warmup,
                n_train_episodes=args.train_ep,
                n_finetune_episodes=args.finetune,
                checkpoint_dir=args.checkpoint_dir,
                show_progress=show_progress,
                n_workers=args.workers,
            )
            if args.train_ep < 5000:
                ratio = args.train_ep / 5000
                prod_overrides["alpha_train"] = max(0.2, 0.8 * ratio)
                prod_overrides["alpha_finetune"] = 0.05
            trainer_cfg = TrainerConfig.production(**prod_overrides)
        else:
            trainer_cfg = TrainerConfig(
                n_warmup_episodes=args.warmup,
                n_train_episodes=args.train_ep,
                n_finetune_episodes=args.finetune,
                checkpoint_dir=args.checkpoint_dir,
                show_progress=show_progress,
                n_workers=args.workers,
            )

        trainer = PPGTrainer(
            executor=executor,
            policy=policy,
            reward_computer=reward,
            credit_assigner=credit,
            config=trainer_cfg,
            reflection_loop=reflection_loop,
            evolver=evolver,
            brancher=brancher,
            logger=ppg_logger,
        )

        # --- Train ---
        _step_rule(3, 4, f"Training PPG ({args.warmup}+{args.train_ep}+{args.finetune} episodes)...")
        lm.reset()
        _lm_inner = lm._lm if hasattr(lm, '_lm') else lm
        if hasattr(_lm_inner, 'reset_stats'):
            _lm_inner.reset_stats()
        t0 = time.time()
        stats = trainer.train(train_examples)
        train_time = time.time() - t0
        train_api_calls = lm.reset()
        train_real_calls = _lm_inner.n_misses if hasattr(_lm_inner, 'n_misses') else train_api_calls
        _info(f"done in {train_time:.0f}s  "
              f"mean_reward={stats.mean_reward('train'):.4f}  "
              f"task_acc={stats.task_accuracy('train'):.4f}  "
              f"api_calls={train_api_calls}")

        # Show cache stats so users can distinguish real API cost from counted calls
        if hasattr(_lm_inner, 'n_hits'):
            _info(f"cache: {_lm_inner.n_hits} hits, {_lm_inner.n_misses} misses, "
                  f"hit_rate={_lm_inner.hit_rate:.1%}  "
                  f"(real API calls ≈ {_lm_inner.n_misses})")

        # --- Calibrate ---
        ppg_path = None
        ppg_calibration_calls = 0
        ppg_calibration_real_calls = 0
        ppg_calibration_info = None
        if args.ppg_calibration == "val_path" and val_ex:
            _info("Calibrating PPG deployment path on validation split...")
            from ppg.eval.path_search import select_path_by_validation

            if args.ppg_path_candidates > 0:
                max_candidates = args.ppg_path_candidates
            elif use_prod:
                max_candidates = 20
            else:
                max_candidates = None
            lm.reset()
            if hasattr(_lm_inner, 'reset_stats'):
                _lm_inner.reset_stats()
            t_cal = time.time()
            selected = select_path_by_validation(
                graph=graph,
                examples=val_ex,
                lm=lm,
                metric=metric,
                constraint_checker=constraint_checker,
                max_candidates=max_candidates,
                n_workers=args.workers,
                show_progress=show_progress,
            )
            ppg_calibration_calls = lm.reset()
            ppg_calibration_real_calls = (
                _lm_inner.n_misses if hasattr(_lm_inner, 'n_misses')
                else ppg_calibration_calls
            )
            ppg_path = selected.path
            ppg_calibration_info = {
                "val_score":       round(selected.val_score, 4),
                "mean_tokens":     round(selected.mean_tokens, 1),
                "utility":         round(selected.utility, 4),
                "n_paths_scored":  selected.n_paths_scored,
                "total_paths":     selected.total_paths,
                "api_calls":       ppg_calibration_calls,
                "real_api_calls":  ppg_calibration_real_calls,
                "time_s":          round(time.time() - t_cal, 1),
            }
            _info(
                f"selected path val={selected.val_score:.4f}  "
                f"avg_tok={selected.mean_tokens:.1f}  "
                f"paths={selected.n_paths_scored}/{selected.total_paths}  "
                f"api_calls={ppg_calibration_calls}  "
                f"real_calls={ppg_calibration_real_calls}"
            )
        elif args.ppg_calibration == "dynamic":
            ppg_path = None
            ppg_calibration_info = {"mode": "dynamic", "api_calls": 0}
            _info("PPG will use greedy bandit routing per input at eval time")

        # --- Evaluate PPG + internal baselines ---
        from ppg.eval.harness import EvalConfig, EvalHarness, EvalReport

        internal_baselines = ["base_model", "flat_all", "static_best", "random_gating", "highest_utility"]
        harness = EvalHarness(
            executor=executor,
            metric=metric,
            lm=lm,
            config=EvalConfig(
                baselines=internal_baselines,
                ppg_path=ppg_path,
                show_progress=show_progress,
                n_workers=args.workers,
            ),
            constraint_checker=constraint_checker,
            logger=ppg_logger,
        )

        _step_rule(4, 4, "Evaluating PPG + internal baselines...")
        real_opt_calls = train_real_calls + ppg_calibration_real_calls
        all_metrics["ppg"] = harness.evaluate_splits(
            "ppg", _splits, lm_counter=lm,
            opt_calls=real_opt_calls,
        )["test"]
        optimized_prompts["ppg"] = (
            build_path_prompt(graph, ppg_path)
            if ppg_path is not None
            else "[dynamic: learned LinUCB routing per input]"
        )

        for name in internal_baselines:
            _info(f"  evaluating {name}...")
            all_metrics[name] = harness.evaluate_splits(
                name, _splits, lm_counter=lm, opt_calls=0
            )["test"]

        optimized_prompts["flat_all"]        = build_seed_prompt(graph)
        optimized_prompts["static_best"]     = build_best_path_prompt(graph)
        optimized_prompts["random_gating"]   = "[dynamic: random node selection per input]"
        optimized_prompts["highest_utility"] = build_best_path_prompt(graph)

    # ==================================================================
    # MIPROv2 path: compile → eval (independent)
    # ==================================================================
    if args.run_mipro:
        _header(f"MIPROv2 (heavy) — {bench}  |  {args.model}")
        _step_rule(1, 2, "Compiling MIPROv2 (auto='heavy')...")
        try:
            import dspy
            dspy_model_str = f"openai/{args.model}" if args.provider == "openai" \
                             else f"anthropic/{args.model}"
            dspy_lm = dspy.LM(dspy_model_str)
            dspy.configure(lm=dspy_lm)
            mipro_counter = _DSPyCounter(dspy_lm)
            from ppg.eval.external import MIPROv2Baseline
            seed_prompt = build_seed_prompt(graph)
            mipro = MIPROv2Baseline(metric=metric, auto="heavy",
                                    constraint_checker=constraint_checker)
            mipro_counter.reset()
            t0_mipro = time.time()
            mipro.compile(trainset=train_ex, valset=val_ex, seed_instructions=seed_prompt)
            mipro_opt_calls = mipro_counter.reset()
            mipro_compile_time = time.time() - t0_mipro
            _info(f"compiled in {mipro_compile_time:.0f}s  opt_calls={mipro_opt_calls}")

            _step_rule(2, 2, "Evaluating MIPROv2...")
            from ppg.eval.harness import EvalConfig, EvalHarness
            mipro_harness = EvalHarness(
                executor=None,
                metric=metric,
                lm=lm,
                config=EvalConfig(baselines=[], show_progress=show_progress, n_workers=args.workers),
                constraint_checker=constraint_checker,
            )
            mipro_harness.register_external("miprov2", mipro)
            all_metrics["miprov2"] = mipro_harness.evaluate_splits(
                "miprov2", _splits, lm_counter=mipro_counter, opt_calls=mipro_opt_calls
            )["test"]
            optimized_prompts["miprov2"] = mipro._prompt_prefix or seed_prompt
        except ImportError as e:
            _info(f"SKIP — {e}")

    # ==================================================================
    # GEPA path: compile → eval (independent)
    # ==================================================================
    if args.run_gepa:
        _header(f"GEPA (heavy) — {bench}  |  {args.model}")
        _step_rule(1, 2, f"Compiling GEPA (max_metric_calls={args.gepa_calls})...")
        try:
            import dspy as _dspy
            from ppg.eval.external import GEPABaseline
            _gepa_model_str = f"openai/{args.model}" if args.provider == "openai" \
                              else f"anthropic/{args.model}"
            _gepa_dspy_lm = _dspy.LM(_gepa_model_str)
            _dspy.configure(lm=_gepa_dspy_lm)
            gepa_counter = _DSPyCounter(_gepa_dspy_lm)
            seed_prompt = build_seed_prompt(graph)
            reflection_lm = (
                _dspy.LM(f"openai/{args.reflection_model}")
                if args.reflection_model else None
            )
            gepa = GEPABaseline(
                metric=metric,
                reflection_lm=reflection_lm,
                max_metric_calls=args.gepa_calls,
                seed=args.seed,
                constraint_checker=constraint_checker,
            )
            gepa_counter.reset()
            t0_gepa = time.time()
            gepa.compile(
                trainset=train_ex,
                valset=val_ex,
                seed_instructions=seed_prompt,
            )
            gepa_opt_calls = gepa_counter.reset()
            gepa_compile_time = time.time() - t0_gepa
            _info(f"compiled in {gepa_compile_time:.0f}s  opt_calls={gepa_opt_calls}")

            _step_rule(2, 2, "Evaluating GEPA...")
            from ppg.eval.harness import EvalConfig, EvalHarness
            gepa_harness = EvalHarness(
                executor=None,
                metric=metric,
                lm=lm,
                config=EvalConfig(baselines=[], show_progress=show_progress, n_workers=args.workers),
                constraint_checker=constraint_checker,
            )
            gepa_harness.register_external("gepa", gepa)
            all_metrics["gepa"] = gepa_harness.evaluate_splits(
                "gepa", _splits, lm_counter=gepa_counter, opt_calls=gepa_opt_calls
            )["test"]
            optimized_prompts["gepa"] = gepa._prompt_prefix or seed_prompt
        except ImportError as e:
            _info(f"SKIP — {e}")

    # ==================================================================
    # Base model: raw input → LM, no prompt engineering
    # ==================================================================
    if args.run_base:
        _header(f"Base model — {bench}  |  {args.model}")
        _step_rule(1, 1, "Evaluating base model (raw input, no prompt)...")
        from ppg.eval.harness import EvalConfig, EvalHarness
        base_harness = EvalHarness(
            executor=None,
            metric=metric,
            lm=lm,
            config=EvalConfig(
                baselines=["base_model"],
                show_progress=show_progress,
                n_workers=args.workers,
            ),
            constraint_checker=constraint_checker,
        )
        all_metrics["base_model"] = base_harness.evaluate_splits(
            "base_model", _splits, lm_counter=lm, opt_calls=0
        )["test"]
        optimized_prompts["base_model"] = "[raw input — no prompt engineering]"

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    if not all_metrics:
        print("\nNo methods ran. Check flags.")
        return

    rows = []
    for name, m in all_metrics.items():
        rows.append(m.as_dict())
    rows.sort(key=lambda d: d["task_accuracy"], reverse=True)
    winner = rows[0]["name"]

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.panel import Panel

        rtable = Table(
            title=f"[bold]Results — {bench}  |  {args.model}[/bold]",
            box=box.ROUNDED,
            show_lines=False,
            header_style="bold cyan",
            padding=(0, 1),
        )
        rtable.add_column("System",     style="bold",    no_wrap=True, min_width=16)
        rtable.add_column("TaskAcc",    justify="right", min_width=8)
        rtable.add_column("StdTask",    justify="right", min_width=8)
        rtable.add_column("AvgTok",     justify="right", min_width=8)
        rtable.add_column("Constraint", justify="right", min_width=11)
        rtable.add_column("LM Calls",   justify="right", min_width=9)

        for row in rows:
            is_winner = row["name"] == winner
            name_str  = (f"[bold green]{row['name']}[/bold green]" if is_winner
                         else row["name"])
            acc_str   = (f"[bold green]{row['task_accuracy']:.4f}[/bold green]"
                         if is_winner else f"{row['task_accuracy']:.4f}")
            rtable.add_row(
                name_str,
                acc_str,
                f"{row['std_task']:.4f}",
                f"{row['mean_tokens']:.1f}",
                f"{row['mean_constraint']:.4f}",
                str(row["lm_calls"]),
            )

        rc = Console()
        rc.print(rtable)
        rc.print(f"  Winner: [bold green]{winner}[/bold green]")
        if train_time > 0:
            rc.print(f"  Training: {train_time:.0f}s")
    except ImportError:
        print(f"\n{'System':<18} {'TaskAcc':>8} {'StdTask':>8} {'AvgTok':>8} "
              f"{'Constraint':>11} {'LMCalls':>8}")
        print("-" * 65)
        for row in rows:
            print(f"{row['name']:<18} {row['task_accuracy']:>8.4f} {row['std_task']:>8.4f} "
                  f"{row['mean_tokens']:>8.1f} {row['mean_constraint']:>11.4f} "
                  f"{row['lm_calls']:>8}")
        print(f"\nWinner: {winner}")

    # ------------------------------------------------------------------
    # Print optimized prompts
    # ------------------------------------------------------------------
    if optimized_prompts:
        try:
            from rich.console import Console as _RC
            from rich.panel import Panel as _RP
            from rich.rule import Rule as _RR
            from rich.text import Text as _RT
            _pc = _RC()
            _pc.print()
            _pc.print(_RR("[bold]Optimized Prompts[/bold]"))
            for _mname, _prompt in optimized_prompts.items():
                _pc.print(_RP(
                    _RT(_prompt),
                    title=f"[bold cyan]{_mname}[/bold cyan]",
                    expand=False,
                ))
        except ImportError:
            print("\n=== Optimized Prompts ===")
            for _mname, _prompt in optimized_prompts.items():
                print(f"\n--- {_mname} ---")
                print(_prompt)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = bench
    if bench == "mmlu" and args.mmlu_subject != "all":
        tag = f"mmlu_{args.mmlu_subject}"
    # Include method name in filename when running single external method
    method_suffix = ""
    if not run_ppg:
        methods_run = []
        if args.run_mipro and "miprov2" in all_metrics:
            methods_run.append("miprov2")
        if args.run_gepa and "gepa" in all_metrics:
            methods_run.append("gepa")
        if methods_run:
            method_suffix = "_" + "_".join(methods_run)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{tag}_{args.model.replace('/', '_')}{method_suffix}_{timestamp}.json"

    payload = {
        "benchmark":      bench,
        "mmlu_subject":   args.mmlu_subject if bench == "mmlu" else None,
        "model":          args.model,
        "provider":       args.provider,
        "reflection_model": args.reflection_model if args.run_gepa else None,
        "methods_run":    list(all_metrics.keys()),
        "n_train":        len(train_examples),
        "n_val":          len(val_ex),
        "n_test":         len(test_ex),
        "seed":           args.seed,
        "train_time_s":   round(train_time, 1) if run_ppg else None,
        "training_stats": stats.summary() if stats else None,
        "ppg_calibration": ppg_calibration_info if run_ppg else None,
        "results":        rows,
        "winner":         winner,
        "optimized_prompts": optimized_prompts,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ------------------------------------------------------------------
    # Diagnostic report (production mode)
    # ------------------------------------------------------------------
    if run_ppg and use_prod and hasattr(ppg_logger, "diagnostic_report"):
        report_text = ppg_logger.diagnostic_report()
        if report_text:
            print(f"\n{'=' * 60}")
            print("PPG DIAGNOSTIC REPORT")
            print(f"{'=' * 60}")
            print(report_text)


if __name__ == "__main__":
    main()
