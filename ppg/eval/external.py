"""
External prompt-optimizer baselines for PPG evaluation.

MIPROv2Baseline  — wraps DSPy MIPROv2 (Bayesian instruction + few-shot tuning)
GEPABaseline     — wraps DSPy GEPA (generative evolutionary prompt adaptation)

Both expose the same two-phase interface used by EvalHarness:
    baseline.compile(trainset, ...)   # optimize prompt offline
    baseline.run(example)             # -> (response: str, tokens: int)

Both require dspy.configure(lm=...) to be called before compile().

Installation
------------
pip install dspy-ai
"""

from __future__ import annotations

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
    a single Predict node whose initial instructions are the seed_instructions
    (e.g. the flat_all assembled prompt). MIPROv2 then optimises those
    instructions on the trainset.

    Parameters
    ----------
    metric : TaskMetric — used inside the DSPy metric function
    auto   : MIPROv2 search budget ("light" | "medium" | "heavy")

    Usage
    -----
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)

        baseline = MIPROv2Baseline(metric=ExactMatchMetric())
        baseline.compile(trainset=train_examples, seed_instructions=flat_prompt)
        response, tokens = baseline.run(test_example)
    """

    @classmethod
    def verify(cls) -> dict:
        """
        Check DSPy availability before calling compile().

        Returns dict with keys:
          available   : bool
          version     : str (if available)
          has_miprov2 : bool (if available)
          error       : str (if not available)
        """
        try:
            import dspy
            return {
                "available":    True,
                "version":      getattr(dspy, "__version__", "unknown"),
                "has_miprov2":  hasattr(dspy, "MIPROv2"),
            }
        except ImportError:
            return {
                "available": False,
                "error":     "dspy-ai not installed: pip install dspy-ai",
            }

    def __init__(
        self,
        metric:             TaskMetric,
        auto:               str = "medium",
        constraint_checker: object = None,
    ):
        self._metric        = metric
        self._auto          = auto
        self._checker       = constraint_checker
        self._program       = None   # set by compile()
        self._prompt_prefix = ""     # optimized instructions extracted after compile()

    # ------------------------------------------------------------------

    def compile(
        self,
        trainset:          list,
        valset:            list = (),
        seed_instructions: str = "",
    ) -> None:
        """
        Run MIPROv2 optimisation on trainset.

        Expects dspy.configure(lm=...) already called before this.

        Parameters
        ----------
        trainset          : list of TrainingExample or EvalExample
        valset            : held-out validation examples for candidate selection;
                            passed to MIPROv2.compile(valset=...) so DSPy does not
                            have to split trainset internally
        seed_instructions : initial prompt text; MIPROv2 will mutate this
        """
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "DSPy required for MIPROv2Baseline: pip install dspy-ai"
            ) from None

        metric  = self._metric
        checker = self._checker

        def _make_ex(ex):
            return dspy.Example(
                input=ex.x,
                y_star=ex.y_star,
                constraints=getattr(ex, "constraints", None) or [],
                metadata=getattr(ex, "metadata", None) or {},
            ).with_inputs("input")

        dspy_train = [_make_ex(ex) for ex in trainset]
        dspy_val   = [_make_ex(ex) for ex in valset] if valset else None

        def dspy_metric(example, prediction, trace=None):
            try:
                response    = getattr(prediction, "response", str(prediction))
                constraints = getattr(example, "constraints", None) or []
                metadata    = getattr(example, "metadata", None) or {}
                if constraints and checker is not None:
                    return float(checker.check(response, constraints, metadata))
                return float(metric.score(response, example.y_star))
            except Exception:
                return 0.0

        instructions = seed_instructions

        class _PPGModule(dspy.Module):
            def __init__(self):
                super().__init__()
                sig = dspy.Signature("input -> response", instructions=instructions) \
                    if instructions else dspy.Signature("input -> response")
                self.predict = dspy.Predict(sig)

            def forward(self, input):
                return self.predict(input=input)

        optimizer = dspy.MIPROv2(metric=dspy_metric, auto=self._auto)
        compile_kwargs = {"trainset": dspy_train}
        if dspy_val is not None:
            compile_kwargs["valset"] = dspy_val
        self._program = optimizer.compile(_PPGModule(), **compile_kwargs)

        try:
            self._prompt_prefix = self._program.predict.signature.instructions or seed_instructions
        except AttributeError:
            self._prompt_prefix = seed_instructions

    def run(self, example: EvalExample) -> tuple[str, int]:
        """Return (response, prompt_token_count) for one example."""
        if self._program is None:
            raise RuntimeError("Call compile() before run()")
        pred     = self._program(input=example.x)
        response = getattr(pred, "response", str(pred))
        from ppg.core.tokenizer import count_tokens
        proxy = f"{self._prompt_prefix}\n\nInput: {example.x}" if self._prompt_prefix else f"Input: {example.x}"
        return response, count_tokens(proxy)


# ---------------------------------------------------------------------------
# GEPABaseline
# ---------------------------------------------------------------------------

class GEPABaseline:
    """
    GEPA (Generative Evolutionary Prompt Adaptation) baseline using DSPy.

    Requires: pip install dspy-ai
    GitHub  : https://github.com/stanfordnlp/dspy

    Uses dspy.teleprompt.GEPA — a DSPy teleprompter that evolves prompt
    instructions via a reflection LM reading scored feedback traces.
    The metric must return (score, feedback_str) to enable GEPA's reflection
    loop; GEPABaseline wraps the plain TaskMetric and generates feedback text.

    Parameters
    ----------
    metric           : TaskMetric — scores (response, y_star) pairs
    reflection_lm    : DSPy LM object or model string used for GEPA reflection
                       (e.g. dspy.LM("openai/gpt-4o") or "openai/gpt-4o");
                       defaults to the globally configured DSPy LM when None
    max_metric_calls : GEPA evaluation budget (total evaluator calls)
    seed             : RNG seed for reproducible optimisation

    Usage
    -----
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)

        baseline = GEPABaseline(
            metric=ExactMatchMetric(),
            reflection_lm=dspy.LM("openai/gpt-4o"),
        )
        baseline.compile(
            trainset=train_examples,
            valset=val_examples,
            seed_instructions=flat_prompt,
        )
        response, tokens = baseline.run(test_example)
    """

    @classmethod
    def verify(cls) -> dict:
        """
        Check DSPy + GEPA availability before calling compile().

        Returns dict with keys:
          available : bool
          version   : str (if available)
          has_gepa  : bool (if available)
          error     : str (if not available)
        """
        try:
            import dspy
            from dspy.teleprompt import GEPA  # noqa: F401
            return {
                "available": True,
                "version":   getattr(dspy, "__version__", "unknown"),
                "has_gepa":  True,
            }
        except ImportError as exc:
            return {
                "available": False,
                "error":     f"dspy-ai not installed or missing GEPA: pip install dspy-ai ({exc})",
            }
        except AttributeError:
            try:
                import dspy
                return {
                    "available": True,
                    "version":   getattr(dspy, "__version__", "unknown"),
                    "has_gepa":  False,
                }
            except ImportError:
                return {"available": False, "error": "dspy-ai not installed: pip install dspy-ai"}

    # ------------------------------------------------------------------

    def __init__(
        self,
        metric:             TaskMetric,
        reflection_lm:      object = None,
        max_metric_calls:   int = 150,
        seed:               int = 42,
        constraint_checker: object = None,
    ):
        self._metric           = metric
        self._reflection_lm    = reflection_lm
        self._max_metric_calls = max_metric_calls
        self._seed             = seed
        self._checker          = constraint_checker
        self._program: Optional[object] = None
        self._prompt_prefix:  str = ""

    def compile(
        self,
        trainset:          list,
        valset:            list = (),
        seed_instructions: str = "",
    ) -> None:
        """
        Run GEPA optimisation on trainset.

        Expects dspy.configure(lm=...) already called before this.

        Parameters
        ----------
        trainset          : training examples (TrainingExample or EvalExample)
        valset            : validation examples for candidate selection
        seed_instructions : initial prompt text; GEPA will evolve this
        """
        try:
            import dspy
            from dspy.teleprompt import GEPA
        except ImportError:
            raise ImportError(
                "DSPy required for GEPABaseline: pip install dspy-ai"
            ) from None

        # DSPy 3.x uses ScoreWithFeedback; fall back to (score, feedback) tuple for 2.x
        try:
            from dspy.teleprompt.gepa.gepa import ScoreWithFeedback as _SWF
        except ImportError:
            _SWF = None

        metric       = self._metric
        checker      = self._checker
        instructions = seed_instructions

        def _make_ex(ex):
            return dspy.Example(
                input=ex.x,
                y_star=ex.y_star,
                constraints=getattr(ex, "constraints", None) or [],
                metadata=getattr(ex, "metadata", None) or {},
            ).with_inputs("input")

        dspy_train = [_make_ex(ex) for ex in trainset]
        dspy_val   = [_make_ex(ex) for ex in valset] if valset else None

        def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            """GEPAFeedbackMetric: return ScoreWithFeedback (DSPy 3.x) or (score, str) (2.x)."""
            try:
                response    = getattr(pred, "response", str(pred))
                constraints = getattr(gold, "constraints", None) or []
                metadata    = getattr(gold, "metadata", None) or {}
                if constraints and checker is not None:
                    score = float(checker.check(response, constraints, metadata))
                else:
                    score = float(metric.score(response, gold.y_star))
            except Exception:
                response = ""
                score    = 0.0
            feedback = (
                f"Expected: {str(gold.y_star)[:300]!r}\n"
                f"Got:      {str(response)[:300]!r}\n"
                f"Score:    {score:.3f}"
            )
            if _SWF is not None:
                return _SWF(score=score, feedback=feedback)
            return score, feedback

        class _PPGModule(dspy.Module):
            def __init__(self):
                super().__init__()
                sig = dspy.Signature("input -> response", instructions=instructions) \
                    if instructions else dspy.Signature("input -> response")
                self.predict = dspy.Predict(sig)

            def forward(self, input):
                return self.predict(input=input)

        gepa_kwargs: dict = {
            "metric":           gepa_metric,
            "max_metric_calls": self._max_metric_calls,
            "seed":             self._seed,
        }
        if self._reflection_lm is not None:
            gepa_kwargs["reflection_lm"] = self._reflection_lm

        optimizer = GEPA(**gepa_kwargs)

        compile_kwargs = {"trainset": dspy_train}
        if dspy_val is not None:
            compile_kwargs["valset"] = dspy_val
        self._program = optimizer.compile(_PPGModule(), **compile_kwargs)

        try:
            self._prompt_prefix = self._program.predict.signature.instructions or seed_instructions
        except AttributeError:
            self._prompt_prefix = seed_instructions

    def run(self, example: EvalExample) -> tuple[str, int]:
        """Return (response, prompt_token_count) for one example."""
        if self._program is None:
            raise RuntimeError("Call compile() before run()")
        pred     = self._program(input=example.x)
        response = getattr(pred, "response", str(pred))
        from ppg.core.tokenizer import count_tokens
        proxy = f"{self._prompt_prefix}\n\nInput: {example.x}" if self._prompt_prefix else f"Input: {example.x}"
        return response, count_tokens(proxy)
