"""Tests for ppg/training/trainer.py."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from ppg.bandits.linucb import LinUCBPolicy
from ppg.core import (
    ExecutorConfig,
    FeatureExtractor,
    FragmentType,
    PPGExecutor,
    PPGraphBuilder,
)
from ppg.core.executor import PromptAssembler
from ppg.training.credit import CreditAssigner, CreditAssignerConfig
from ppg.training.reward import ExactMatchMetric, RewardComputer, RewardConfig
from ppg.training.trainer import (
    EpisodeResult,
    PPGTrainer,
    TrainerConfig,
    TrainingExample,
    TrainingStats,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

class FixedLM:
    """Always returns a fixed string. Counts calls."""
    def __init__(self, response: str = "42"):
        self.response = response
        self.n_calls  = 0

    def complete(self, prompt: str) -> str:
        self.n_calls += 1
        return self.response


def make_graph():
    b = PPGraphBuilder()
    b.add_fragment(FragmentType.TASK_FRAMING,    "Task: {input}")
    b.add_fragment(FragmentType.REASONING_STYLE, "Think.")
    b.add_fragment(FragmentType.OUTPUT_CONTRACT, "Answer:")
    ids = b.node_ids()
    b.connect_chain(*ids)
    return b.build(), ids


def make_components(lm=None, correct_response="42"):
    if lm is None:
        lm = FixedLM(correct_response)
    graph, ids = make_graph()
    policy   = LinUCBPolicy(graph, alpha=0.5)
    executor = PPGExecutor(
        graph=graph,
        selector=policy,
        lm=lm,
        feature_extractor=FeatureExtractor(),
        config=ExecutorConfig(escalation_enabled=False),
    )
    assembler = PromptAssembler(graph)
    metric    = ExactMatchMetric()
    reward    = RewardComputer(
        task_metric=metric,
        lm=lm,
        assembler=assembler,
        config=RewardConfig(skip_variance=True),
    )
    credit = CreditAssigner(
        lm=lm,
        assembler=assembler,
        task_metric=metric,
        config=CreditAssignerConfig(p_ablate=1.0, min_path_length=2),
    )
    return executor, policy, reward, credit, graph


def make_dataset(n: int = 10, answer: str = "42") -> list[TrainingExample]:
    return [TrainingExample(x=f"q{i}", y_star=answer) for i in range(n)]


def make_trainer(lm=None, cfg=None, correct_response="42"):
    executor, policy, reward, credit, graph = make_components(
        lm=lm, correct_response=correct_response
    )
    trainer = PPGTrainer(
        executor=executor,
        policy=policy,
        reward_computer=reward,
        credit_assigner=credit,
        config=cfg or TrainerConfig(
            n_warmup_episodes=5,
            n_train_episodes=5,
            n_finetune_episodes=5,
            checkpoint_dir=None,
        ),
    )
    return trainer, policy, executor


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------

class TestTrainingExample:
    def test_defaults(self):
        ex = TrainingExample(x="hello", y_star="world")
        assert ex.constraints == []

    def test_with_constraints(self):
        ex = TrainingExample(x="q", y_star="a", constraints=["bullet"])
        assert ex.constraints == ["bullet"]


# ---------------------------------------------------------------------------
# TrainingStats
# ---------------------------------------------------------------------------

class TestTrainingStats:
    def test_empty_stats(self):
        s = TrainingStats()
        assert s.n_episodes() == 0
        assert s.mean_reward() == pytest.approx(0.0)
        assert s.task_accuracy() == pytest.approx(0.0)
        assert s.reward_history() == []

    def _make_result(self, phase, total_r, task_r):
        from ppg.training.reward import RewardComponents
        r = RewardComponents(task=task_r, constraint=0.0, cost=0.0,
                             variance=0.0, total=total_r)
        return EpisodeResult(phase=phase, episode=0, reward=r,
                             credit=None, path=[], token_count=5)

    def test_record_and_count(self):
        s = TrainingStats()
        s.record(self._make_result("train", 1.0, 0.9))
        s.record(self._make_result("train", 0.5, 0.5))
        assert s.n_episodes() == 2
        assert s.n_episodes("train") == 2
        assert s.n_episodes("warmup") == 0

    def test_mean_reward_all_phases(self):
        s = TrainingStats()
        s.record(self._make_result("warmup", 1.0, 1.0))
        s.record(self._make_result("train",  0.0, 0.0))
        assert s.mean_reward() == pytest.approx(0.5)

    def test_mean_reward_per_phase(self):
        s = TrainingStats()
        s.record(self._make_result("warmup", 1.0, 1.0))
        s.record(self._make_result("train",  0.2, 0.2))
        assert s.mean_reward("warmup") == pytest.approx(1.0)
        assert s.mean_reward("train")  == pytest.approx(0.2)

    def test_task_accuracy(self):
        s = TrainingStats()
        s.record(self._make_result("train", 0.9, 1.0))
        s.record(self._make_result("train", 0.1, 0.0))
        assert s.task_accuracy("train") == pytest.approx(0.5)

    def test_summary_keys(self):
        s = TrainingStats()
        for ph in ("warmup", "train", "finetune"):
            s.record(self._make_result(ph, 0.5, 0.5))
        summary = s.summary()
        assert set(summary.keys()) == {"warmup", "train", "finetune"}
        for ph in summary:
            assert "n_episodes" in summary[ph]
            assert "mean_reward" in summary[ph]
            assert "task_accuracy" in summary[ph]

    def test_summary_excludes_zero_episode_phases(self):
        s = TrainingStats()
        s.record(self._make_result("train", 0.5, 0.5))
        summary = s.summary()
        assert "warmup" not in summary
        assert "finetune" not in summary

    def test_results_returns_copy(self):
        s = TrainingStats()
        s.record(self._make_result("train", 0.5, 0.5))
        r = s.results
        r.clear()
        assert s.n_episodes() == 1  # original unaffected


# ---------------------------------------------------------------------------
# TrainerConfig defaults
# ---------------------------------------------------------------------------

class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.n_warmup_episodes   == 200
        assert cfg.n_train_episodes    == 1000
        assert cfg.n_finetune_episodes == 200
        assert cfg.alpha_train         == pytest.approx(0.5)
        assert cfg.alpha_finetune      == pytest.approx(0.1)
        assert cfg.checkpoint_dir is None

    def test_custom(self):
        cfg = TrainerConfig(n_warmup_episodes=10, alpha_train=1.0, seed=7)
        assert cfg.n_warmup_episodes == 10
        assert cfg.seed == 7


# ---------------------------------------------------------------------------
# PPGTrainer._make_cycle
# ---------------------------------------------------------------------------

class TestMakeCycle:
    def test_exact_length(self):
        trainer, *_ = make_trainer()
        ds = make_dataset(4)
        cycle = trainer._make_cycle(ds, 10)
        assert len(cycle) == 10

    def test_zero_returns_empty(self):
        trainer, *_ = make_trainer()
        ds = make_dataset(4)
        assert trainer._make_cycle(ds, 0) == []

    def test_smaller_than_dataset(self):
        trainer, *_ = make_trainer()
        ds = make_dataset(10)
        cycle = trainer._make_cycle(ds, 3)
        assert len(cycle) == 3

    def test_all_elements_from_dataset(self):
        trainer, *_ = make_trainer()
        ds = make_dataset(5)
        cycle = trainer._make_cycle(ds, 15)
        xs_in_dataset = {ex.x for ex in ds}
        for ex in cycle:
            assert ex.x in xs_in_dataset


# ---------------------------------------------------------------------------
# PPGTrainer.train — episode counts
# ---------------------------------------------------------------------------

class TestTrainerEpisodeCounts:
    def test_total_episodes(self):
        cfg = TrainerConfig(n_warmup_episodes=3, n_train_episodes=4,
                            n_finetune_episodes=2, seed=0)
        trainer, _, _ = make_trainer(cfg=cfg)
        stats = trainer.train(make_dataset(5))
        assert stats.n_episodes() == 9

    def test_phase_episode_counts(self):
        cfg = TrainerConfig(n_warmup_episodes=3, n_train_episodes=4,
                            n_finetune_episodes=2, seed=0)
        trainer, _, _ = make_trainer(cfg=cfg)
        stats = trainer.train(make_dataset(5))
        assert stats.n_episodes("warmup")   == 3
        assert stats.n_episodes("train")    == 4
        assert stats.n_episodes("finetune") == 2

    def test_zero_warmup(self):
        cfg = TrainerConfig(n_warmup_episodes=0, n_train_episodes=3,
                            n_finetune_episodes=0, seed=0)
        trainer, _, _ = make_trainer(cfg=cfg)
        stats = trainer.train(make_dataset(5))
        assert stats.n_episodes("warmup")   == 0
        assert stats.n_episodes("train")    == 3

    def test_empty_dataset_raises(self):
        trainer, *_ = make_trainer()
        with pytest.raises(ValueError):
            trainer.train([])

    def test_single_example_dataset(self):
        cfg = TrainerConfig(n_warmup_episodes=2, n_train_episodes=2,
                            n_finetune_episodes=1, seed=0)
        trainer, *_ = make_trainer(cfg=cfg)
        stats = trainer.train(make_dataset(1))
        assert stats.n_episodes() == 5


# ---------------------------------------------------------------------------
# PPGTrainer.train — policy is updated
# ---------------------------------------------------------------------------

class TestPolicyUpdates:
    def test_policy_updates_after_training(self):
        cfg = TrainerConfig(n_warmup_episodes=2, n_train_episodes=3,
                            n_finetune_episodes=1, seed=0)
        trainer, policy, _ = make_trainer(cfg=cfg)
        trainer.train(make_dataset(5))
        assert policy.total_updates > 0

    def test_warmup_uses_random_selector(self):
        """During warmup, executor.selector should be RandomSelector."""
        from ppg.core.executor import RandomSelector

        observed_selectors: list[type] = []

        class SentinelLM:
            def complete(self, prompt):
                return "42"

        executor, policy, reward, credit, graph = make_components(lm=SentinelLM())

        class SpyTrainer(PPGTrainer):
            def _run_episode(self, example, phase, train_mode):
                if phase == "warmup":
                    observed_selectors.append(type(self.executor.selector))
                return super()._run_episode(example, phase, train_mode)

        cfg = TrainerConfig(n_warmup_episodes=3, n_train_episodes=0,
                            n_finetune_episodes=0)
        trainer = SpyTrainer(executor=executor, policy=policy,
                             reward_computer=reward, credit_assigner=credit,
                             config=cfg)
        trainer.train(make_dataset(5))
        assert all(s is RandomSelector for s in observed_selectors)

    def test_policy_selector_restored_after_warmup(self):
        cfg = TrainerConfig(n_warmup_episodes=2, n_train_episodes=2,
                            n_finetune_episodes=0, seed=0)
        trainer, policy, executor = make_trainer(cfg=cfg)
        trainer.train(make_dataset(5))
        assert executor.selector is policy

    def test_alpha_restored_after_all_phases(self):
        original_alpha = 0.5
        cfg = TrainerConfig(n_warmup_episodes=1, n_train_episodes=1,
                            n_finetune_episodes=1, alpha_train=2.0,
                            alpha_finetune=0.01, seed=0)
        trainer, policy, _ = make_trainer(cfg=cfg)
        policy.alpha = original_alpha
        trainer.train(make_dataset(5))
        assert policy.alpha == pytest.approx(original_alpha)


# ---------------------------------------------------------------------------
# PPGTrainer.train — reward recorded
# ---------------------------------------------------------------------------

class TestRewardRecording:
    def test_all_results_have_reward(self):
        cfg = TrainerConfig(n_warmup_episodes=2, n_train_episodes=2,
                            n_finetune_episodes=1, seed=0)
        trainer, *_ = make_trainer(cfg=cfg)
        stats = trainer.train(make_dataset(5))
        for result in stats.results:
            assert result.reward is not None
            assert isinstance(result.reward.total, float)

    def test_correct_lm_gives_high_task_reward(self):
        """LM always returns correct answer → task reward = 1.0."""
        cfg = TrainerConfig(n_warmup_episodes=0, n_train_episodes=5,
                            n_finetune_episodes=0, seed=0)
        trainer, *_ = make_trainer(correct_response="42", cfg=cfg)
        stats = trainer.train(make_dataset(5, answer="42"))
        assert stats.task_accuracy("train") == pytest.approx(1.0)

    def test_wrong_lm_gives_zero_task_reward(self):
        """LM always returns wrong answer → task reward = 0.0."""
        cfg = TrainerConfig(n_warmup_episodes=0, n_train_episodes=5,
                            n_finetune_episodes=0, seed=0)
        trainer, *_ = make_trainer(correct_response="WRONG", cfg=cfg)
        stats = trainer.train(make_dataset(5, answer="42"))
        assert stats.task_accuracy("train") == pytest.approx(0.0)

    def test_result_path_is_list_of_strings(self):
        cfg = TrainerConfig(n_warmup_episodes=1, n_train_episodes=0,
                            n_finetune_episodes=0, seed=0)
        trainer, *_ = make_trainer(cfg=cfg)
        stats = trainer.train(make_dataset(2))
        for r in stats.results:
            assert isinstance(r.path, list)
            assert all(isinstance(n, str) for n in r.path)


# ---------------------------------------------------------------------------
# PPGTrainer — on_episode callback
# ---------------------------------------------------------------------------

class TestOnEpisodeCallback:
    def test_callback_called_for_every_episode(self):
        calls: list[int] = []

        def cb(episode_idx, result):
            calls.append(episode_idx)

        cfg = TrainerConfig(n_warmup_episodes=2, n_train_episodes=3,
                            n_finetune_episodes=1, seed=0)
        trainer, *_ = make_trainer(cfg=cfg)
        trainer.on_episode = cb
        trainer.train(make_dataset(5))
        assert len(calls) == 6

    def test_callback_receives_episode_result(self):
        results: list[EpisodeResult] = []

        def cb(idx, result):
            results.append(result)

        cfg = TrainerConfig(n_warmup_episodes=1, n_train_episodes=0,
                            n_finetune_episodes=0, seed=0)
        trainer, *_ = make_trainer(cfg=cfg)
        trainer.on_episode = cb
        trainer.train(make_dataset(3))
        assert isinstance(results[0], EpisodeResult)
        assert results[0].phase == "warmup"

    def test_episode_indices_monotone(self):
        indices: list[int] = []

        cfg = TrainerConfig(n_warmup_episodes=2, n_train_episodes=3,
                            n_finetune_episodes=1, seed=0)
        trainer, *_ = make_trainer(cfg=cfg)
        trainer.on_episode = lambda i, r: indices.append(i)
        trainer.train(make_dataset(5))
        assert indices == list(range(6))


# ---------------------------------------------------------------------------
# PPGTrainer — checkpointing
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_checkpoint_file_created(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = TrainerConfig(
                n_warmup_episodes=0,
                n_train_episodes=5,
                n_finetune_episodes=0,
                checkpoint_dir=d,
                checkpoint_every=3,
                seed=0,
            )
            trainer, *_ = make_trainer(cfg=cfg)
            trainer.train(make_dataset(5))
            files = os.listdir(d)
            # Episode 0 and 3 trigger checkpoints (0%3==0, 3%3==0)
            assert len(files) >= 1
            assert all(f.endswith(".npz") for f in files)

    def test_no_checkpoint_when_dir_none(self):
        cfg = TrainerConfig(n_warmup_episodes=0, n_train_episodes=5,
                            n_finetune_episodes=0, checkpoint_dir=None, seed=0)
        trainer, *_ = make_trainer(cfg=cfg)
        # Should not raise even with no checkpoint_dir
        trainer.train(make_dataset(5))

    def test_checkpoint_loadable(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = TrainerConfig(
                n_warmup_episodes=0,
                n_train_episodes=3,
                n_finetune_episodes=0,
                checkpoint_dir=d,
                checkpoint_every=1,
                seed=0,
            )
            trainer, policy, _ = make_trainer(cfg=cfg)
            trainer.train(make_dataset(5))

            files = sorted(os.listdir(d))
            assert files, "No checkpoint files found"

            # Load into fresh policy and verify it doesn't crash
            graph, _ = make_graph()
            policy2 = LinUCBPolicy(graph)
            policy2.load(os.path.join(d, files[0]))
            # If load succeeded, total_updates should be >= 0
            assert policy2.total_updates >= 0


# ---------------------------------------------------------------------------
# PPGTrainer — reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_reward_history(self):
        cfg = TrainerConfig(n_warmup_episodes=3, n_train_episodes=3,
                            n_finetune_episodes=2, seed=42)
        ds = make_dataset(5)

        trainer1, *_ = make_trainer(cfg=cfg)
        stats1 = trainer1.train(ds)

        trainer2, *_ = make_trainer(cfg=cfg)
        stats2 = trainer2.train(ds)

        assert stats1.reward_history() == pytest.approx(stats2.reward_history())
