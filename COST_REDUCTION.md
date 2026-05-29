# PPG API Cost Reduction — Analysis & Roadmap

Goal: cut LM API spend without degrading benchmark quality (SOTA-preserving).
Scope: every code path that calls `lm.complete()` / `lm.sample()`.

## Where the money goes

Cost model for **one benchmark × one model**, production config
(`TrainerConfig.production`, `ExecutorConfig.production`, defaults from the code):

| Phase | LM calls | Driver |
|---|---:|---|
| Warm-up (500 ep) | 500 | random routing, 1 call/ep |
| Train (5,000 ep) | 20,750 | **GRPO k=4 (20,000)** + LOO ablation p=0.15 (750) |
| Fine-tune (1,000 ep) | 4,000 | **GRPO k=4 still runs in fine-tune** |
| Calibration (path search) | ~6,000 | ~30 paths × 200 val |
| Eval (PPG, k=3 + ~30% escalate) | ~1,650 | self-consistency |
| **Total** | **~32,900** | — |

**GRPO sampling alone is ~73% of all calls.** Training is ~77%. Two structural
facts make almost everything else cheap-to-fix:

1. **No provider-native prompt caching** anywhere (`ppg/lm/clients.py` sends no
   `cache_control` / cache headers). PPG prompts are mostly a *static fragment
   prefix* + a small variable input — the ideal caching shape, currently unused.
2. **No Batch API.** Training, warm-up, calibration and eval are all offline /
   throughput-bound, yet every call goes through the synchronous endpoint at
   full price.

Ranked recommendations below. Each notes expected savings and *why quality is
preserved*. Items are independent unless noted.

---

## Tier 1 — Large savings, ~zero quality risk

### 1. Route all offline calls through the Batch API  → ~50% on ~97% of calls
Training, warm-up, calibration, and non-interactive eval are not latency
sensitive. OpenAI Batch and Anthropic Message Batches both bill at **−50%** with
identical models/outputs. Add a `BatchLMClient` wrapper behind the existing
`LMClient` protocol; collect a mini-batch of prompts (the trainer already groups
`n_workers` episodes and GRPO already generates k prompts at once) and submit as
one batch job. **Quality: identical** — same model, same sampling params, only
the delivery channel changes. This is the single highest-leverage change.

### 2. Provider-native prompt caching  → 50–90% of *input* tokens
A PPG prompt = stable fragment templates (+ system msg, few-shot) followed by the
per-example `{input}`. Across GRPO's k=4 calls, self-consistency's k=3 calls,
LOO ablations, and perturbation calls, the prefix is largely identical.
- Anthropic: mark the assembled static prefix with `cache_control:{type:"ephemeral"}`.
- OpenAI: automatic prompt caching triggers on shared ≥1024-token prefixes.
Ensure `PromptAssembler` always emits **static fragments first, `{input}` last**
(reorder if a graph puts input early). **Quality: identical** — caching only
changes billing of repeated prefix tokens. Compounds with everything else.

### 3. Drop GRPO in fine-tune; make k adaptive in train  → ~12,000–15,000 calls
- Fine-tune runs at `alpha_finetune=0.05` (near-greedy) yet still pays
  `k_grpo=4` → 4,000 calls that teach the bandit almost nothing. Set
  `k_grpo_paths=1` for the fine-tune phase: **−3,000 calls**, negligible quality
  impact (exploitation phase, low advantage signal).
- In train, make k *adaptive*: only spend extra paths when the LinUCB arm is
  uncertain (high posterior variance) or early in the phase; collapse to k=2
  once an edge's reward estimate stabilizes. Group-relative advantage with k=2–3
  retains most of the gradient signal of k=4. Expected **−8,000 to −12,000**
  train calls. *Verify with an ablation: k∈{2,3,4} vs. final task accuracy.*

---

## Tier 2 — Large savings, low quality risk (verify with a small ablation)

### 4. Mixed-model auxiliary calls  → 40–70% on LOO + perturbation (+ warm-up)
LOO credit (`credit.py`) and perturbation-variance (`reward.py`) only need a
*relative / comparative* signal — "did removing this node hurt?", "how much does
the score wobble under perturbation?". They do **not** need the SOTA model's
absolute answer quality. Route these auxiliary calls (and the random-routing
warm-up, which exists only to seed coverage) to a cheaper model
(e.g. `*-mini` / Haiku). Keep the main task call and eval on the SOTA model.
A cheaper proxy preserves the *ranking* that credit/variance depend on at a
fraction of the price. **Quality: main-path outputs unchanged.** *Verify the
cheap proxy correlates with full-model marginals on a sample before committing.*

### 5. Adaptive (early-exit) self-consistency at eval  → 30–60% of eval calls
Eval unconditionally draws k=3 then escalates. Replace with adaptive sampling:
draw 1, draw more **only** until the majority answer is statistically settled
(adaptive-consistency style stopping), then escalate on residual disagreement as
today. Easy inputs resolve in 1 call; only genuinely ambiguous ones use the full
budget. **Quality: matched or better** — published adaptive-consistency results
show equal accuracy at ~40% fewer samples.

### 6. Racing / successive-halving for path calibration  → 2–5× on calibration
`select_path_by_validation` scores each candidate path on the *full* val set.
Instead: score all candidates on a small val subset, keep the top survivors,
then score only survivors on the full set (Hyperband/UCB racing). The learned
utility pre-ranking already exists to seed this. **Quality: preserved** — the
final winner is still validated on full val; only clearly-dominated paths are
dropped early.

---

## Tier 3 — Structural / smaller wins

### 7. Right-size `max_tokens` per call type
All calls use `max_tokens=512`. LOO and perturbation calls only need an answer
span, not full reasoning. Cap auxiliary calls (and MCQ/short-answer tasks) at a
much smaller budget → direct output-token savings. **Quality: unaffected** for
extraction-scored tasks; keep full budget where chain-of-thought is scored.

### 8. Enable plateau early-stopping in training
`early_stop_window` defaults to 0 (off). Turning it on halts a phase once mean
reward stops improving — caps wasted episodes on already-converged runs.
**Quality: preserved** by definition (stops only at plateau).

### 9. Reuse responses across GRPO / credit / variance within an episode
These three mechanisms fire independently in the same episode and sometimes
assemble near-identical prompts (e.g. an ablated path can coincide with a GRPO
path sample). Coordinate them through one in-episode memo so a given
`(path, input)` is called once. Compounds with the exact-hash disk cache already
present. **Quality: identical** — pure deduplication.

### 10. Selectively run eval baselines
The harness can run up to 6 baselines, each multiplying eval cost. For routine
runs, evaluate PPG + 1–2 baselines; run the full comparison only for the paper
table. Reporting choice, not a quality change.

---

## Suggested sequencing

1. **Batch API + prompt caching** (Tier 1.1, 1.2) — biggest $/effort, zero risk,
   compounds with the rest. Do first.
2. **GRPO: kill in fine-tune, adaptive-k in train** (1.3) — kills the 73% driver.
3. **Mixed-model aux calls** (2.4) and **adaptive self-consistency** (2.5) —
   each behind a one-flag ablation to confirm parity.
4. Calibration racing (2.6), token sizing (3.7), early-stop (3.8), dedup (3.9).

Combined, Tier 1 alone plausibly cuts spend **~3–4×** (50% batch × large prompt-
cache discount on input tokens × ~half the GRPO calls). Tiers 2–3 push further on
the remaining auxiliary and eval calls.

---

## Implementation status (all items shipped)

Every item above is implemented behind config flags; `761 passed` (was 750 +
11 new tests in `tests/test_cost_reduction.py`). Nothing changes default
behaviour unless a flag is set, except the `--production` presets, which now
bundle the safe cost controls.

| Item | Where | How to enable |
|---|---|---|
| 1. Batch API (−50%) | `OpenAIBatchClient`, `BatchLMClient` (`ppg/lm/clients.py`) | `--batch-api` |
| 2. Prompt caching | `AnthropicClient._system_param`, `enable_prompt_cache` config | `--enable-prompt-cache` |
| 3. GRPO fine-tune off + adaptive-k | `TrainerConfig.{k_grpo_paths_finetune,grpo_adaptive,…}`, `_grpo_k`, `LinUCBPolicy.path_uncertainty` | on by default in `--production`; tune via `TrainerConfig` |
| 4. Mixed-model aux calls | `aux_lm` on `RewardComputer` + `CreditAssigner` | `--aux-model MODEL --aux-max-tokens N` |
| 5. Adaptive self-consistency | `ExecutorConfig.adaptive_*`, `_complete_adaptive` | on by default in `--production` |
| 6. Calibration racing | `select_path_by_validation(racing_subset, racing_survivors)`, `_race_paths` | `--racing-subset K --racing-survivors M` |
| 7. Token sizing | `--aux-max-tokens` (default 256) on aux calls | with `--aux-model` |
| 8. Plateau early-stop | `TrainerConfig.early_stop_window` | on by default in `--production` (window=200) |
| 9. In-episode dedup | `MemoizingLMClient` (`ppg/lm/clients.py`) | auto when `--no-cache`; disk cache covers the cached case |

Example production run with the full cost stack:

```bash
python scripts/run_benchmark.py gsm8k --model gpt-4.1-mini --production \
    --batch-api --enable-prompt-cache \
    --aux-model gpt-4o-mini --aux-max-tokens 256 \
    --racing-subset 25 --racing-survivors 8
```

Notes:
- Batch clients expose `complete_batch`; `DiskCachedLMClient`, `MemoizingLMClient`,
  `CountingLMClient`, and the perturbation-variance path all forward to it and
  deduplicate, so a native Batch backend propagates everywhere automatically.
- GRPO adaptive-k spends the full `k_grpo_paths` budget only when
  `LinUCBPolicy.path_uncertainty(edges, phi) >= grpo_uncertainty_threshold`;
  converged paths collapse to `k_grpo_min`, and fine-tune uses
  `k_grpo_paths_finetune` (1 = off).
- Adaptive self-consistency floors at 2 draws (a single draw is trivially
  unanimous) and stops once the lead is insurmountable or the top answer holds
  `adaptive_confidence` of the votes.

## Verification plan (don't ship blind)
- Add the existing `CountingLMClient` around each phase and log calls/episode
  before vs. after — the repo already supports this.
- For every quality-touching item (3, 4, 5, 6), run the matched-budget harness on
  one or two benchmarks and confirm task accuracy is within noise (use the
  existing CI / std reporting in `ppg/eval/harness.py`).
- Gate each change behind a config flag so ablations are reproducible.
