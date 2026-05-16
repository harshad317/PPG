"""
External prompt-optimizer baselines for PPG evaluation.

MIPROv2Baseline  — wraps DSPy MIPROv2 (Bayesian instruction + few-shot tuning)
GEPABaseline     — wraps GEPA optimize_anything (evolutionary trace-based search)

Both expose the same two-phase interface used by EvalHarness:
    baseline.compile(trainset, ...)   # optimize prompt offline
    baseline.run(example)             # -> (response: str, tokens: int)

Installation
------------
MIPROv2 : pip install dspy-ai
GEPA    : pip install gepa   (or: pip install git+https://github.com/gepa-ai/gepa.git)
"""

from __future__ import annotations

import random
from typing import Optional

from ppg.eval.harness import EvalExample
from ppg.training.reward import TaskMetric


# ---------------------------------------------------------------------------
# MIPROv2Baseline
# ---------------------------------------------------------------------------

class MIPROv2Baseline:
    """
    MIPROv2 prompt optimizer baseline using DSPy.

    Requires: pip install dspy-ai

    MIPROv2 performs Bayesian search over instruction candidates and few-shot
    example orderings for a DSPy program. For PPG comparison the "program" is
    a single Predict node whose initial instructions are the seed_prompt
    (e.g. the flat_all assembled prompt). MIPROv2 then optimises those
    instructions on the trainset.

    Parameters
    ----------
    metric        : TaskMetric — used inside the DSPy metric function
    auto          : MIPROv2 search budget ("light" | "medium" | "heavy")

    Usage
    -----
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)

        baseline = MIPROv2Baseline(metric=ExactMatchMetric())
        baseline.compile(trainset=train_examples, seed_instructions=flat_prompt)
        response, tokens = baseline.run(test_example)
    """

    def __init__(
        self,
        metric: TaskMetric,
        auto:   str = "medium",
    ):
        self._metric         = metric
        self._auto           = auto
        self._program        = None   # set by compile()
        self._prompt_prefix  = ""    # optimized instructions extracted after compile()

    # ------------------------------------------------------------------

    def compile(
        self,
        trainset:          list,
        seed_instructions: str = "",
    ) -> None:
        """
        Run MIPROv2 optimisation on trainset.

        Expects dspy.configure(lm=...) already called before this.

        Parameters
        ----------
        trainset          : list of TrainingExample or EvalExample
        seed_instructions : initial prompt text; MIPROv2 will mutate this
        """
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "DSPy required for MIPROv2Baseline: pip install dspy-ai"
            ) from None

        metric = self._metric

        dspy_train = [
            dspy.Example(input=ex.x, y_star=ex.y_star).with_inputs("input")
            for ex in trainset
        ]

        def dspy_metric(example, prediction, trace=None):
            try:
                response = getattr(prediction, "response", str(prediction))
                return float(metric.score(response, example.y_star))
            except Exception:
                return 0.0

        instructions = seed_instructions

        class _PPGModule(dspy.Module):
            def __init__(self):
                super().__init__()
                sig = dspy.Signature("input -> response")
                if instructions:
                    sig = sig.with_instructions(instructions)
                self.predict = dspy.Predict(sig)

            def forward(self, input):
                return self.predict(input=input)

        optimizer = dspy.MIPROv2(metric=dspy_metric, auto=self._auto)
        self._program = optimizer.compile(_PPGModule(), trainset=dspy_train)

        # Extract optimized instructions for fair prompt-token accounting in run().
        # DSPy stores them on the compiled predict's signature; fall back to seed.
        try:
            self._prompt_prefix = self._program.predict.signature.instructions or seed_instructions
        except AttributeError:
            self._prompt_prefix = seed_instructions

    def run(self, example: EvalExample) -> tuple[str, int]:
        """Return (response, prompt_token_count) for one example.

        Token count covers the estimated prompt sent to the LM (optimized
        instructions + input), matching how all other baselines count tokens.
        Response tokens are excluded to keep accounting comparable.
        """
        if self._program is None:
            raise RuntimeError("Call compile() before run()")
        pred     = self._program(input=example.x)
        response = getattr(pred, "response", str(pred))
        proxy    = f"{self._prompt_prefix}\n\nInput: {example.x}" if self._prompt_prefix else f"Input: {example.x}"
        return response, len(proxy.split())


# ---------------------------------------------------------------------------
# GEPABaseline
# ---------------------------------------------------------------------------

class GEPABaseline:
    """
    GEPA (Generative Evolutionary Prompt Adaptation) baseline.

    Requires: pip install gepa
    GitHub  : https://github.com/gepa-ai/gepa

    Uses gepa.optimize_anything() — the evaluator receives a candidate prompt
    string, runs it against a mini-batch of training examples via lm_client,
    and returns the mean task score. GEPA's reflection LM reads the logged
    traces to diagnose failures and propose improvements.

    Parameters
    ----------
    metric           : TaskMetric — used inside the evaluator to score responses
    lm_client        : LMClient — used for task inference inside evaluator and
                       for final run() inference after optimisation
    reflection_lm    : model identifier for GEPA's reflection LM
                       (e.g. "openai/gpt-4o", "anthropic/claude-opus-4-7")
    max_metric_calls : GEPA evaluation budget (total evaluator calls)
    n_eval_examples  : examples per evaluator call; trades cost vs. signal
    seed             : RNG seed for reproducible mini-batch sampling

    Usage
    -----
        baseline = GEPABaseline(
            metric=ExactMatchMetric(),
            lm_client=my_lm,
            reflection_lm="openai/gpt-4o",
        )
        baseline.compile(
            trainset=train_examples,
            valset=val_examples,
            seed_prompt=flat_prompt,
            objective="Maximize exact-match accuracy on math word problems.",
        )
        response, tokens = baseline.run(test_example)
    """

    def __init__(
        self,
        metric:           TaskMetric,
        lm_client,
        reflection_lm:    str = "openai/gpt-4o",
        max_metric_calls: int = 150,
        n_eval_examples:  int = 20,
        seed:             int = 42,
    ):
        self._metric           = metric
        self._lm               = lm_client
        self._reflection_lm    = reflection_lm
        self._max_metric_calls = max_metric_calls
        self._n_eval           = n_eval_examples
        self._rng              = random.Random(seed)
        self._optimized_prompt: Optional[str] = None

    # ------------------------------------------------------------------

    def compile(
        self,
        trainset:  list,
        valset:    list,
        seed_prompt: str,
        objective: str = "Optimize this prompt to maximize task accuracy.",
    ) -> None:
        """
        Run GEPA optimisation.

        Parameters
        ----------
        trainset    : training examples (TrainingExample or EvalExample);
                      used as fallback eval pool when valset is empty
        valset      : validation examples; the evaluator samples from this set
                      (not trainset) to give GEPA unbiased candidate scores,
                      matching gepa.optimize()'s intended train/val split
        seed_prompt : starting prompt text; GEPA mutates this
        objective   : natural-language description of optimisation goal
        """
        try:
            import gepa.optimize_anything as oa
            from gepa.optimize_anything import GEPAConfig, EngineConfig, optimize_anything
        except ImportError:
            raise ImportError(
                "GEPA required: pip install gepa  "
                "or: pip install git+https://github.com/gepa-ai/gepa.git"
            ) from None

        metric    = self._metric
        lm        = self._lm
        n_eval    = self._n_eval
        eval_pool = list(valset) if valset else list(trainset)
        rng       = self._rng

        def evaluator(candidate: str) -> float:
            batch = rng.sample(eval_pool, min(n_eval, len(eval_pool)))
            scores: list[float] = []
            for ex in batch:
                prompt = f"{candidate}\n\nInput: {ex.x}"
                try:
                    response = lm.complete(prompt)
                    oa.log(f"Input: {ex.x[:120]}")
                    oa.log(f"Output: {response[:200]}")
                    scores.append(metric.score(response, ex.y_star))
                except Exception as e:
                    oa.log(f"Error: {e}")
                    scores.append(0.0)
            return float(sum(scores) / len(scores)) if scores else 0.0

        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=self._max_metric_calls,
                reflection_lm=self._reflection_lm,
            )
        )

        result = optimize_anything(
            seed_candidate=seed_prompt,
            evaluator=evaluator,
            objective=objective,
            config=config,
        )

        bc = result.best_candidate
        if isinstance(bc, str):
            self._optimized_prompt = bc
        elif isinstance(bc, dict):
            # gepa.optimize() style: {"system_prompt": ...}
            self._optimized_prompt = bc.get("system_prompt", str(bc))
        else:
            self._optimized_prompt = str(bc)

    def run(self, example: EvalExample) -> tuple[str, int]:
        """Return (response, token_count) for one example."""
        if self._optimized_prompt is None:
            raise RuntimeError("Call compile() before run()")
        prompt   = f"{self._optimized_prompt}\n\nInput: {example.x}"
        response = self._lm.complete(prompt)
        return response, len(prompt.split())
