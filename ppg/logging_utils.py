"""
Structured logging for PPG training, evaluation, and diagnostics.

PPGLogger captures every signal in the system:
  - Per-episode: reward components, path choice, credit, reflection, GRPO
  - Per-phase: aggregate stats, convergence curves
  - Bandit: arm selection frequencies, UCB scores, mu_hat norms
  - Pareto: archive size, front size, dominance rank distribution
  - Evolution: mutations, crossovers, pruning
  - Branching: new branches, failure modes targeted
  - Evaluation: per-example breakdown, baseline comparison

Backends:
  - JSONL file (always on) — one JSON line per event, grep-friendly
  - CSV summary (always on) — one row per episode, spreadsheet-friendly
  - W&B (optional) — real-time dashboards, artifact tracking
  - Console (optional) — periodic summary prints

Usage:
    logger = PPGLogger("runs/gsm8k_v1")
    trainer = PPGTrainer(..., logger=logger)
    # ... training ...
    logger.diagnostic_report()
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class LogConfig:
    log_dir:           str  = "ppg_logs"
    log_every:         int  = 1
    summary_every:     int  = 100
    console_every:     int  = 200
    enable_wandb:      bool = False
    wandb_project:     str  = "ppg"
    wandb_run_name:    Optional[str] = None
    log_predictions:   bool = True
    log_arm_stats:     bool = True
    log_pareto:        bool = True
    max_prediction_len: int = 200


class PPGLogger:
    """
    Unified logger for the entire PPG pipeline.

    All log methods are no-ops when the logger is not initialized,
    so components can call logger methods unconditionally.
    """

    def __init__(self, config: Optional[LogConfig] = None):
        self.cfg = config or LogConfig()
        self._dir = Path(self.cfg.log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self._dir / "events.jsonl"
        self._csv_path   = self._dir / "episodes.csv"
        self._jsonl_f    = open(self._jsonl_path, "a")
        self._csv_writer = None
        self._csv_f      = None
        self._csv_headers_written = False

        self._wandb_run = None
        if self.cfg.enable_wandb:
            self._init_wandb()

        self._episode_count = 0
        self._phase_start_time: Optional[float] = None

        # Accumulators for summary stats
        self._phase_rewards:     list[float] = []
        self._phase_tasks:       list[float] = []
        self._phase_constraints: list[float] = []
        self._phase_tokens:      list[int]   = []
        self._phase_reflections: int = 0
        self._phase_credits:     int = 0

        # Global accumulators for diagnostic report
        self._all_rewards:    list[float] = []
        self._all_tasks:      list[float] = []
        self._path_counter:   dict[str, int] = defaultdict(int)
        self._failure_modes:  dict[str, int] = defaultdict(int)
        self._node_utilities: dict[str, list[float]] = defaultdict(list)
        self._grpo_advantages: list[float] = []
        self._pareto_front_sizes: list[int] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if self._jsonl_f:
            self._jsonl_f.close()
        if self._csv_f:
            self._csv_f.close()
        if self._wandb_run:
            self._wandb_run.finish()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Core logging methods
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode:     int,
        phase:       str,
        reward_components: dict,
        path:        list[str],
        token_count: int,
        input_text:  str = "",
        prediction:  str = "",
        reference:   str = "",
        credit:      Optional[dict] = None,
        reflection:  Optional[dict] = None,
        grpo:        Optional[dict] = None,
        constraints: Optional[list[str]] = None,
    ):
        """Log one training episode with all available signals."""
        self._episode_count = episode

        # Accumulators
        self._all_rewards.append(reward_components.get("r_total", 0))
        self._all_tasks.append(reward_components.get("r_task", 0))
        self._phase_rewards.append(reward_components.get("r_total", 0))
        self._phase_tasks.append(reward_components.get("r_task", 0))
        self._phase_constraints.append(reward_components.get("r_constraint", 0))
        self._phase_tokens.append(token_count)

        path_key = "->".join(path)
        self._path_counter[path_key] += 1

        if credit:
            self._phase_credits += 1
        if reflection:
            self._phase_reflections += 1
            for cat in reflection.get("failure_categories", []):
                self._failure_modes[cat] += 1

        if episode % self.cfg.log_every != 0:
            return

        event = {
            "type": "episode",
            "episode": episode,
            "phase": phase,
            "ts": time.time(),
            **reward_components,
            "path": path,
            "path_len": len(path),
            "tokens": token_count,
        }

        if self.cfg.log_predictions:
            event["input"] = input_text[:self.cfg.max_prediction_len]
            event["prediction"] = prediction[:self.cfg.max_prediction_len]
            event["reference"] = reference[:self.cfg.max_prediction_len]

        if constraints:
            event["n_constraints"] = len(constraints)

        if credit:
            event["credit"] = credit

        if reflection:
            event["reflection"] = reflection

        if grpo:
            event["grpo"] = grpo
            for adv in grpo.get("advantages", []):
                self._grpo_advantages.append(adv)

        self._write_jsonl(event)
        self._write_csv_row(event)
        self._log_wandb(event)

    def log_phase_start(self, phase: str, n_episodes: int):
        self._phase_start_time = time.time()
        self._phase_rewards.clear()
        self._phase_tasks.clear()
        self._phase_constraints.clear()
        self._phase_tokens.clear()
        self._phase_reflections = 0
        self._phase_credits = 0

        self._write_jsonl({
            "type": "phase_start",
            "phase": phase,
            "n_episodes": n_episodes,
            "ts": time.time(),
        })

    def log_phase_end(self, phase: str):
        elapsed = time.time() - self._phase_start_time if self._phase_start_time else 0
        n = len(self._phase_rewards)

        summary = {
            "type": "phase_end",
            "phase": phase,
            "n_episodes": n,
            "elapsed_sec": round(elapsed, 1),
            "mean_reward": round(float(np.mean(self._phase_rewards)), 4) if n else 0,
            "mean_task": round(float(np.mean(self._phase_tasks)), 4) if n else 0,
            "std_task": round(float(np.std(self._phase_tasks)), 4) if n else 0,
            "mean_constraint": round(float(np.mean(self._phase_constraints)), 4) if n else 0,
            "mean_tokens": round(float(np.mean(self._phase_tokens)), 1) if n else 0,
            "n_reflections": self._phase_reflections,
            "n_credits": self._phase_credits,
            "ts": time.time(),
        }
        self._write_jsonl(summary)
        self._log_wandb(summary)
        self._print_phase_summary(summary)

    def log_arm_stats(self, arm_stats: dict[str, dict]):
        """Log per-edge bandit arm diagnostics."""
        if not self.cfg.log_arm_stats:
            return
        event = {
            "type": "arm_stats",
            "episode": self._episode_count,
            "arms": arm_stats,
            "ts": time.time(),
        }
        self._write_jsonl(event)

        if self._wandb_run:
            flat = {}
            for label, stats in arm_stats.items():
                for k, v in stats.items():
                    flat[f"arms/{label}/{k}"] = v
            self._wandb_run.log(flat, step=self._episode_count)

    def log_pareto(self, archive_size: int, front_size: int,
                   dominance_rank: float = 0.0):
        """Log Pareto archive state."""
        if not self.cfg.log_pareto:
            return
        self._pareto_front_sizes.append(front_size)
        event = {
            "type": "pareto",
            "episode": self._episode_count,
            "archive_size": archive_size,
            "front_size": front_size,
            "dominance_rank": round(dominance_rank, 4),
            "ts": time.time(),
        }
        self._write_jsonl(event)
        self._log_wandb({"pareto/archive_size": archive_size,
                         "pareto/front_size": front_size,
                         "pareto/dominance_rank": dominance_rank})

    def log_evolution(self, actions: list[str], stats: dict):
        """Log fragment evolution cycle."""
        if not actions:
            return
        event = {
            "type": "evolution",
            "episode": self._episode_count,
            "actions": actions,
            "stats": stats,
            "ts": time.time(),
        }
        self._write_jsonl(event)
        self._log_wandb({"evolution/n_mutations": stats.get("n_mutations", 0),
                         "evolution/n_crossovers": stats.get("n_crossovers", 0),
                         "evolution/n_pruned": stats.get("n_pruned", 0)})

    def log_branching(self, actions: list[str], stats: dict):
        """Log failure-mode branching."""
        if not actions:
            return
        event = {
            "type": "branching",
            "episode": self._episode_count,
            "actions": actions,
            "stats": stats,
            "ts": time.time(),
        }
        self._write_jsonl(event)
        self._log_wandb({"branching/n_branches": stats.get("n_branches", 0)})

    def log_grpo(self, group_rewards: list[float], advantages: list[float]):
        """Log GRPO group statistics."""
        event = {
            "type": "grpo",
            "episode": self._episode_count,
            "group_rewards": [round(r, 4) for r in group_rewards],
            "advantages": [round(a, 4) for a in advantages],
            "group_mean": round(float(np.mean(group_rewards)), 4),
            "group_std": round(float(np.std(group_rewards)), 4),
            "ts": time.time(),
        }
        self._write_jsonl(event)
        self._log_wandb({"grpo/group_mean": float(np.mean(group_rewards)),
                         "grpo/group_std": float(np.std(group_rewards)),
                         "grpo/max_advantage": float(max(advantages)) if advantages else 0})

    def log_eval_example(
        self,
        method:     str,
        idx:        int,
        score:      float,
        tokens:     int,
        input_text: str = "",
        prediction: str = "",
        reference:  str = "",
        constraint_score: Optional[float] = None,
        path:       Optional[list[str]] = None,
    ):
        """Log one evaluation example."""
        event = {
            "type": "eval_example",
            "method": method,
            "idx": idx,
            "score": round(score, 4),
            "tokens": tokens,
            "ts": time.time(),
        }
        if constraint_score is not None:
            event["constraint_score"] = round(constraint_score, 4)
        if path:
            event["path"] = path
        if self.cfg.log_predictions:
            event["input"] = input_text[:self.cfg.max_prediction_len]
            event["prediction"] = prediction[:self.cfg.max_prediction_len]
            event["reference"] = reference[:self.cfg.max_prediction_len]
        self._write_jsonl(event)

    def log_eval_summary(self, method: str, metrics: dict):
        """Log evaluation summary for one method."""
        event = {
            "type": "eval_summary",
            "method": method,
            **metrics,
            "ts": time.time(),
        }
        self._write_jsonl(event)
        self._log_wandb({f"eval/{method}/{k}": v for k, v in metrics.items()
                         if isinstance(v, (int, float))})

    def log_node_utility(self, node_id: str, ftype: str, utility: float, n: int):
        """Track fragment utility over time."""
        self._node_utilities[f"{ftype}:{node_id[:8]}"].append(utility)
        self._write_jsonl({
            "type": "node_utility",
            "episode": self._episode_count,
            "node_id": node_id,
            "ftype": ftype,
            "utility": round(utility, 4),
            "n_samples": n,
            "ts": time.time(),
        })

    # ------------------------------------------------------------------
    # Diagnostic report
    # ------------------------------------------------------------------

    def diagnostic_report(self) -> str:
        """
        Generate a comprehensive diagnostic report.

        Answers: what's failing, what's working, where to push next.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("PPG DIAGNOSTIC REPORT")
        lines.append("=" * 70)

        n = len(self._all_rewards)
        if n == 0:
            lines.append("No episodes logged yet.")
            return "\n".join(lines)

        # Overall performance
        lines.append(f"\n--- OVERALL ({n} episodes) ---")
        lines.append(f"  Mean reward:    {np.mean(self._all_rewards):.4f}")
        lines.append(f"  Mean task:      {np.mean(self._all_tasks):.4f}")
        lines.append(f"  Task accuracy:  {np.mean([1.0 if t >= 0.99 else 0.0 for t in self._all_tasks]):.1%}")

        # Reward distribution
        r = np.array(self._all_rewards)
        lines.append(f"  Reward p25/p50/p75: {np.percentile(r,25):.3f} / {np.percentile(r,50):.3f} / {np.percentile(r,75):.3f}")

        # Convergence: compare first 20% vs last 20%
        split = max(1, n // 5)
        early = np.mean(self._all_tasks[:split])
        late  = np.mean(self._all_tasks[-split:])
        lines.append(f"  Early task (first 20%): {early:.4f}")
        lines.append(f"  Late task  (last  20%): {late:.4f}")
        lines.append(f"  Improvement:            {late - early:+.4f}")

        # Path diversity
        lines.append(f"\n--- PATH DIVERSITY ---")
        lines.append(f"  Unique paths: {len(self._path_counter)}")
        top_paths = sorted(self._path_counter.items(), key=lambda x: -x[1])[:5]
        for path, count in top_paths:
            frac = count / n
            lines.append(f"  {frac:5.1%}  {path}")

        # Failure modes
        if self._failure_modes:
            lines.append(f"\n--- FAILURE MODES ---")
            total_failures = sum(self._failure_modes.values())
            for mode, count in sorted(self._failure_modes.items(), key=lambda x: -x[1]):
                lines.append(f"  {mode:<20s}  {count:>5d}  ({count/total_failures:.0%})")

        # Fragment utilities
        if self._node_utilities:
            lines.append(f"\n--- FRAGMENT UTILITIES (latest) ---")
            for label, utils in sorted(self._node_utilities.items()):
                if utils:
                    lines.append(f"  {label:<35s}  {utils[-1]:+.4f}  (n={len(utils)})")

        # Feature activation status
        lines.append(f"\n--- FEATURE STATUS ---")
        lines.append(f"  GRPO events:      {len(self._grpo_advantages) if self._grpo_advantages else 0}")
        lines.append(f"  Pareto snapshots:  {len(self._pareto_front_sizes) if self._pareto_front_sizes else 0}")
        n_reflections = sum(self._failure_modes.values()) if self._failure_modes else 0
        lines.append(f"  Reflections:       {n_reflections}")

        # GRPO stats
        if self._grpo_advantages:
            adv = np.array(self._grpo_advantages)
            lines.append(f"\n--- GRPO ADVANTAGES ---")
            lines.append(f"  Mean advantage:  {np.mean(adv):+.4f}")
            lines.append(f"  Std advantage:   {np.std(adv):.4f}")
            lines.append(f"  Max advantage:   {np.max(adv):+.4f}")
            lines.append(f"  Min advantage:   {np.min(adv):+.4f}")
        else:
            lines.append(f"\n--- GRPO ---")
            lines.append(f"  NOT ACTIVE — check k_grpo_paths > 1 and workers config")

        # Pareto
        if self._pareto_front_sizes:
            lines.append(f"\n--- PARETO ARCHIVE ---")
            lines.append(f"  Final front size: {self._pareto_front_sizes[-1]}")
            lines.append(f"  Max front size:   {max(self._pareto_front_sizes)}")
        else:
            lines.append(f"\n--- PARETO ---")
            lines.append(f"  NOT ACTIVE — check --production and no --no-pareto")

        # Actionable recommendations
        lines.append(f"\n--- RECOMMENDATIONS ---")
        recs = self._recommendations(early, late, n)
        for i, rec in enumerate(recs, 1):
            lines.append(f"  {i}. {rec}")

        lines.append("=" * 70)
        report = "\n".join(lines)

        report_path = self._dir / "diagnostic_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        return report

    def _recommendations(self, early_task: float, late_task: float, n: int) -> list[str]:
        recs = []

        if late_task - early_task < 0.01 and n > 500:
            recs.append("Training not converging — try higher alpha_train or more warmup episodes")

        if late_task < 0.3:
            recs.append("Very low task accuracy — check if metric/scorer matches benchmark format")

        if len(self._path_counter) <= 2 and n > 200:
            recs.append("Low path diversity — bandit collapsed to one path. Increase alpha or add fragments")

        constraint_failures = self._failure_modes.get("constraint", 0)
        format_failures = self._failure_modes.get("format", 0)
        total_failures = sum(self._failure_modes.values()) if self._failure_modes else 1
        if constraint_failures > total_failures * 0.3:
            recs.append("Constraint failures dominate — strengthen output_contract fragments or add constraint-specific few-shot examples")
        if format_failures > total_failures * 0.3:
            recs.append("Format failures dominate — add explicit format instructions to output_contract or reasoning_style fragments")

        reasoning_failures = self._failure_modes.get("reasoning", 0)
        if reasoning_failures > total_failures * 0.4:
            recs.append("Reasoning failures dominate — try stronger reasoning_style fragments or enable escalation")

        if self._grpo_advantages and np.std(self._grpo_advantages) < 0.01:
            recs.append("GRPO advantages near zero — paths too similar. Add more fragment variants")

        if not recs:
            recs.append("Training looks healthy. Run evaluation to check generalization.")

        return recs

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

    def _write_jsonl(self, event: dict):
        line = json.dumps(event, default=_json_default)
        self._jsonl_f.write(line + "\n")
        self._jsonl_f.flush()

    def _write_csv_row(self, event: dict):
        if event.get("type") != "episode":
            return

        if self._csv_f is None:
            self._csv_f = open(self._csv_path, "a", newline="")

        flat = {
            "episode":       event.get("episode"),
            "phase":         event.get("phase"),
            "r_task":        event.get("r_task"),
            "r_constraint":  event.get("r_constraint"),
            "r_cost":        event.get("r_cost"),
            "r_variance":    event.get("r_variance"),
            "r_total":       event.get("r_total"),
            "tokens":        event.get("tokens"),
            "path_len":      event.get("path_len"),
            "n_constraints": event.get("n_constraints", 0),
            "has_credit":    1 if event.get("credit") else 0,
            "has_reflection": 1 if event.get("reflection") else 0,
        }

        if not self._csv_headers_written:
            self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=list(flat.keys()))
            self._csv_writer.writeheader()
            self._csv_headers_written = True

        self._csv_writer.writerow(flat)
        self._csv_f.flush()

    def _print_phase_summary(self, summary: dict):
        phase = summary["phase"]
        n = summary["n_episodes"]
        elapsed = summary["elapsed_sec"]
        mt = summary["mean_task"]
        mr = summary["mean_reward"]
        st = summary["std_task"]
        mc = summary["mean_constraint"]
        tok = summary["mean_tokens"]
        refl = summary["n_reflections"]
        cred = summary["n_credits"]

        print(f"\n  [{phase}] {n} ep in {elapsed:.0f}s | "
              f"task={mt:.3f}±{st:.3f} | reward={mr:.3f} | "
              f"constraint={mc:.3f} | tok={tok:.0f} | "
              f"reflect={refl} credit={cred}")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------

    def _init_wandb(self):
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config={"log_dir": str(self._dir)},
                reinit=True,
            )
        except ImportError:
            self._wandb_run = None

    def _log_wandb(self, data: dict):
        if self._wandb_run is None:
            return
        flat = {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                flat[k] = v
            elif isinstance(v, str) and k in ("type", "phase"):
                pass
        if flat:
            self._wandb_run.log(flat, step=self._episode_count)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


class NullLogger:
    """No-op logger for when logging is disabled. All methods are no-ops."""
    def log_episode(self, *a, **kw): pass
    def log_phase_start(self, *a, **kw): pass
    def log_phase_end(self, *a, **kw): pass
    def log_arm_stats(self, *a, **kw): pass
    def log_pareto(self, *a, **kw): pass
    def log_evolution(self, *a, **kw): pass
    def log_branching(self, *a, **kw): pass
    def log_grpo(self, *a, **kw): pass
    def log_eval_example(self, *a, **kw): pass
    def log_eval_summary(self, *a, **kw): pass
    def log_node_utility(self, *a, **kw): pass
    def diagnostic_report(self) -> str: return "No logger configured."
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
