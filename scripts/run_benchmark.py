#!/usr/bin/env python3
"""
PPG full benchmark runner.

Trains PPG from scratch on a benchmark's training split, then evaluates it
against all baselines on the test split. MIPROv2 and GEPA are compiled in
"heavy" mode (maximum search budget).

Baselines evaluated
-------------------
ppg            : trained PPG (LinUCB bandit + LOO credit + variance penalty)
flat_all       : all graph nodes concatenated, no routing
static_best    : greedy highest-utility fixed path, no routing
random_gating  : random node selection, no learning
highest_utility: greedy on learned utility scores, no UCB
miprov2        : MIPROv2 (DSPy) — requires --run-mipro + dspy-ai installed
gepa           : GEPA optimize_anything — requires --run-gepa + gepa installed

Graph fallback map (benchmarks without dedicated fragments use the closest domain)
----------------------------------------------------------------------------------
ifbench      → ifeval graph     (instruction following)
drop         → hotpotqa graph   (reading comprehension)
truthfulqa   → hotpotqa graph   (open-domain QA)
bigbench_hard→ gsm8k graph      (multi-step reasoning)
arc_challenge→ gsm8k graph      (MCQ reasoning)
livebench_math→ gsm8k graph     (math reasoning)
mmlu         → gsm8k graph      (MCQ reasoning)

Usage
-----
# Quick smoke test (no external optimizers):
python scripts/run_benchmark.py gsm8k \\
    --model gpt-4o-mini

# Full run with all baselines (uses defaults: 100 train / 50 val / 500 test):
python scripts/run_benchmark.py gsm8k \\
    --model gpt-4o-mini --run-mipro --run-gepa --reflection-model gpt-4o

# IFBench (constraint following):
python scripts/run_benchmark.py ifbench \\
    --model gpt-4o-mini --run-mipro --run-gepa

# BigBenchHard (specific task):
python scripts/run_benchmark.py bigbench_hard \\
    --model gpt-4o-mini --bbh-task causal_judgement \\
    --run-mipro --run-gepa

# MMLU (specific subject):
python scripts/run_benchmark.py mmlu \\
    --model gpt-4o-mini --mmlu-subject abstract_algebra

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
    "ifeval":        "ifeval",
    "ifbench":       "ifeval",     # instruction-following domain
    "hotpotqa":      "hotpotqa",
    "drop":          "hotpotqa",   # reading-comprehension domain
    "mbpp":          "mbpp",
    "truthfulqa":    "hotpotqa",   # open-domain QA
    "bigbench_hard": "gsm8k",      # multi-step reasoning
    "arc_challenge": "gsm8k",      # MCQ reasoning
    "livebench_math":"gsm8k",      # math reasoning
    "mmlu":          "gsm8k",      # MCQ reasoning
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
    bbh_task:     str,
    mmlu_subject: str,
):
    """
    Returns (train_examples, val_examples, test_examples, metric, constraint_checker, objective).

    For benchmarks with a single HF split, performs a deterministic
    train/val/test partition. For benchmarks with separate HF splits, uses
    them directly and caps to n_train / n_val / n_test.
    """
    from ppg.eval.benchmarks.loaders import (
        ARCChallengeLoader, BigBenchHardLoader, DROPLoader, GSM8KLoader,
        HotpotQALoader, IFBenchLoader, IFEvalLoader, LiveBenchMathLoader,
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

    def _cap(lst, n):
        out = list(lst)
        rng.shuffle(out)
        return out[:n]

    constraint_checker = None
    objective = "Optimize the prompt to maximize task accuracy."

    # -----------------------------------------------------------------------
    if benchmark == "gsm8k":
        loader = GSM8KLoader()
        train  = _cap(loader.load("train", seed=seed), n_train)
        val    = _cap(loader.load("test",  seed=seed + 1), n_val)
        test   = _cap(loader.load("test",  seed=seed + 2), n_test)
        metric = loader.recommended_metric()
        objective = "Maximize exact-match accuracy on multi-step arithmetic word problems."

    # -----------------------------------------------------------------------
    elif benchmark == "ifeval":
        loader = IFEvalLoader()
        all_ex = loader.load("train", seed=seed)
        train, val, test = _split_single(all_ex, n_train, n_val, n_test)
        metric = loader.recommended_metric()
        constraint_checker = loader.recommended_constraint_checker()
        objective = (
            "Maximize instruction-following compliance — responses must satisfy "
            "all explicit format and content constraints in the prompt."
        )

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
        val    = _cap(loader.load("validation", seed=seed + 1), n_val)
        test   = _cap(loader.load("validation", seed=seed + 2), n_test)
        metric = loader.recommended_metric()
        objective = "Maximize token-F1 on multi-hop reading-comprehension questions."

    # -----------------------------------------------------------------------
    elif benchmark == "drop":
        loader = DROPLoader()
        train  = _cap(loader.load("train",      seed=seed),     n_train)
        val    = _cap(loader.load("validation", seed=seed + 1), n_val)
        test   = _cap(loader.load("validation", seed=seed + 2), n_test)
        metric = loader.recommended_metric()
        objective = "Maximize F1 on discrete-reasoning passages (arithmetic, counting, sorting)."

    # -----------------------------------------------------------------------
    elif benchmark == "mbpp":
        loader = MBPPLoader()
        train  = _cap(loader.load("train", seed=seed),     n_train)
        val    = _cap(loader.load("test",  seed=seed + 1), n_val)
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
    elif benchmark == "bigbench_hard":
        loader = BigBenchHardLoader()
        all_ex = loader.load(task=bbh_task, split="test", seed=seed)
        train, val, test = _split_single(all_ex, n_train, n_val, n_test)
        metric = loader.recommended_metric()
        objective = f"Maximize exact-match accuracy on BIG-Bench Hard task: {bbh_task}."

    # -----------------------------------------------------------------------
    elif benchmark == "arc_challenge":
        loader = ARCChallengeLoader()
        train  = _cap(loader.load("train", seed=seed),     n_train)
        val    = _cap(loader.load("test",  seed=seed + 1), n_val)
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
        # dev split is tiny (5 per subject); supplement with validation when n_train > available
        dev_ex  = loader.load(subject=mmlu_subject, split="dev",        seed=seed)
        val_ex  = loader.load(subject=mmlu_subject, split="validation", seed=seed)
        test_ex = loader.load(subject=mmlu_subject, split="test",       seed=seed)
        train_pool = dev_ex + val_ex
        rng.shuffle(train_pool)
        train = train_pool[:n_train]
        val   = _cap(test_ex, n_val)
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
        TrainingExample(x=ex.x, y_star=ex.y_star, constraints=ex.constraints)
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


# ---------------------------------------------------------------------------
# LM client factory
# ---------------------------------------------------------------------------

def make_lm(provider: str, model: str, cache_dir: str | None):
    if provider == "openai":
        from ppg.lm.clients import OpenAIClient, OpenAIConfig, DiskCachedLMClient
        lm = OpenAIClient(OpenAIConfig(model=model, temperature=0.0, max_tokens=512))
    elif provider == "anthropic":
        from ppg.lm.clients import AnthropicClient, AnthropicConfig, DiskCachedLMClient
        lm = AnthropicClient(AnthropicConfig(model=model, temperature=0.0, max_tokens=512))
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
    parser.add_argument("--run-mipro", action="store_true", dest="run_mipro",
                        help="Compile and run MIPROv2 baseline (requires dspy-ai)")
    parser.add_argument("--run-gepa",  action="store_true", dest="run_gepa",
                        help="Compile and run GEPA baseline (requires gepa)")
    parser.add_argument("--gepa-calls", type=int, default=500, dest="gepa_calls",
                        help="GEPA max_metric_calls heavy budget (default: 500)")
    parser.add_argument("--bbh-task",    default="causal_judgement", dest="bbh_task",
                        help="BigBenchHard task name (default: causal_judgement)")
    parser.add_argument("--mmlu-subject",default="all", dest="mmlu_subject",
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
    args = parser.parse_args()

    show_progress = not args.quiet
    cache_dir     = None if args.no_cache else args.cache_dir
    bench         = args.benchmark

    print(f"\n=== PPG Benchmark: {bench} | model: {args.model} ===\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/6] Loading dataset splits...")
    train_ex, val_ex, test_ex, metric, constraint_checker, objective = load_splits(
        benchmark=bench,
        n_train=args.train_n,
        n_val=args.val_n,
        n_test=args.test_n,
        seed=args.seed,
        bbh_task=args.bbh_task,
        mmlu_subject=args.mmlu_subject,
    )
    print(f"      train={len(train_ex)}  val={len(val_ex)}  test={len(test_ex)}")
    train_examples = to_training(train_ex)

    # ------------------------------------------------------------------
    # 2. Build LM client
    # ------------------------------------------------------------------
    print("[2/6] Building LM client...")
    lm = make_lm(args.provider, args.model, cache_dir)

    # ------------------------------------------------------------------
    # 3. Build graph + PPG components
    # ------------------------------------------------------------------
    print("[3/6] Building PPG graph and components...")
    from ppg.data.fragments import build_graph
    from ppg.bandits.linucb import LinUCBPolicy
    from ppg.core import ExecutorConfig, FeatureExtractor, PPGExecutor
    from ppg.core.executor import PromptAssembler
    from ppg.training.credit import CreditAssigner, CreditAssignerConfig
    from ppg.training.reward import RewardComputer, RewardConfig
    from ppg.training.trainer import PPGTrainer, TrainerConfig

    graph_key = _GRAPH_MAP[bench]
    graph     = build_graph(graph_key, topology="rich")
    policy    = LinUCBPolicy(graph)
    executor  = PPGExecutor(
        graph=graph,
        selector=policy,
        lm=lm,
        feature_extractor=FeatureExtractor(),
        config=ExecutorConfig(escalation_enabled=False),
    )
    assembler = PromptAssembler(graph)
    # For constraint-only benchmarks (IFEval, IFBench), constraint satisfaction
    # IS the task signal — ExactMatch against y_star would always be 0.
    constraint_as_task = bench in ("ifeval", "ifbench")
    reward    = RewardComputer(
        task_metric=metric,
        lm=lm,
        assembler=assembler,
        constraint_checker=constraint_checker,
        config=RewardConfig(constraint_as_task=constraint_as_task),
    )
    credit    = CreditAssigner(
        lm=lm,
        assembler=assembler,
        task_metric=metric,
        config=CreditAssignerConfig(),
    )
    trainer   = PPGTrainer(
        executor=executor,
        policy=policy,
        reward_computer=reward,
        credit_assigner=credit,
        config=TrainerConfig(
            n_warmup_episodes=args.warmup,
            n_train_episodes=args.train_ep,
            n_finetune_episodes=args.finetune,
            checkpoint_dir=args.checkpoint_dir,
            show_progress=show_progress,
            n_workers=args.workers,
        ),
    )

    # ------------------------------------------------------------------
    # 4. Train PPG
    # ------------------------------------------------------------------
    print(f"[4/6] Training PPG ({args.warmup}+{args.train_ep}+{args.finetune} episodes)...")
    t0 = time.time()
    stats = trainer.train(train_examples)
    train_time = time.time() - t0
    print(f"      done in {train_time:.0f}s  "
          f"mean_reward={stats.mean_reward('train'):.4f}  "
          f"task_acc={stats.task_accuracy('train'):.4f}")

    # ------------------------------------------------------------------
    # 5. Compile external baselines (MIPROv2 + GEPA, heavy mode)
    # ------------------------------------------------------------------
    external_baselines: dict = {}

    if args.run_mipro:
        print("[5a] Compiling MIPROv2 (auto='heavy') ...")
        try:
            import dspy
            dspy_model_str = f"openai/{args.model}" if args.provider == "openai" \
                             else f"anthropic/{args.model}"
            dspy.configure(lm=dspy.LM(dspy_model_str))

            from ppg.eval.external import MIPROv2Baseline
            seed_prompt = build_seed_prompt(graph)
            mipro = MIPROv2Baseline(metric=metric, auto="heavy")
            mipro.compile(trainset=train_ex, valset=val_ex, seed_instructions=seed_prompt)
            external_baselines["miprov2"] = mipro
            print("      MIPROv2 compiled successfully.")
        except ImportError as e:
            print(f"      SKIP — {e}")

    if args.run_gepa:
        print(f"[5b] Compiling GEPA (max_metric_calls={args.gepa_calls}, heavy) ...")
        try:
            from ppg.eval.external import GEPABaseline
            seed_prompt = build_seed_prompt(graph)
            gepa = GEPABaseline(
                metric=metric,
                lm_client=lm,
                reflection_lm=f"openai/{args.reflection_model}",
                max_metric_calls=args.gepa_calls,
                n_eval_examples=20,
                seed=args.seed,
            )
            gepa.compile(
                trainset=train_ex,
                valset=val_ex,
                seed_prompt=seed_prompt,
                objective=objective,
            )
            external_baselines["gepa"] = gepa
            print("      GEPA compiled successfully.")
        except ImportError as e:
            print(f"      SKIP — {e}")

    if not args.run_mipro and not args.run_gepa:
        print("[5/6] Skipping external baselines (use --run-mipro / --run-gepa to enable).")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("[6/6] Evaluating all baselines on test set...")
    from ppg.eval.harness import EvalConfig, EvalHarness

    baselines = ["flat_all", "static_best", "random_gating", "highest_utility"]
    if "miprov2" in external_baselines:
        baselines.append("miprov2")
    if "gepa" in external_baselines:
        baselines.append("gepa")

    harness = EvalHarness(
        executor=executor,
        metric=metric,
        lm=lm,
        config=EvalConfig(baselines=baselines, show_progress=show_progress),
        constraint_checker=constraint_checker,
        external_baselines=external_baselines if external_baselines else None,
    )
    report = harness.evaluate(test_ex)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    table = report.comparison_table()
    print(f"\n{'System':<18} {'TaskAcc':>8} {'StdTask':>8} {'AvgTok':>8} "
          f"{'Constraint':>11} {'LMCalls':>8}")
    print("-" * 65)
    for row in table:
        print(f"{row['name']:<18} {row['task_accuracy']:>8.4f} {row['std_task']:>8.4f} "
              f"{row['mean_tokens']:>8.1f} {row['mean_constraint']:>11.4f} "
              f"{row['lm_calls']:>8}")

    winner = report.winner()
    ppg_vs_flat = report.ppg_delta("flat_all")
    print(f"\nWinner: {winner}  |  PPG vs flat_all: {ppg_vs_flat:+.4f}")
    print(f"Training time: {train_time:.0f}s")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = bench
    if bench == "bigbench_hard":
        tag = f"bbh_{args.bbh_task}"
    elif bench == "mmlu" and args.mmlu_subject != "all":
        tag = f"mmlu_{args.mmlu_subject}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{tag}_{args.model.replace('/', '_')}_{timestamp}.json"

    payload = {
        "benchmark":      bench,
        "bbh_task":       args.bbh_task if bench == "bigbench_hard" else None,
        "mmlu_subject":   args.mmlu_subject if bench == "mmlu" else None,
        "model":          args.model,
        "provider":       args.provider,
        "reflection_model": args.reflection_model if args.run_gepa else None,
        "n_train":        len(train_examples),
        "n_val":          len(val_ex),
        "n_test":         len(test_ex),
        "seed":           args.seed,
        "train_time_s":   round(train_time, 1),
        "training_stats": stats.summary(),
        "results":        table,
        "winner":         winner,
        "ppg_vs_flat_all": round(ppg_vs_flat, 4),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
