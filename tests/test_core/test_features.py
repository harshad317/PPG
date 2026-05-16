"""Tests for ppg/core/features.py."""

import numpy as np
import pytest

from ppg.core import (
    FEATURE_DIM,
    FEATURE_NAMES,
    N_CLUSTERS,
    RuntimeFeatures,
    FeatureExtractor,
    default_normalizer,
    verbatim_normalizer,
)
from ppg.core.features import _consistency_features, _HashCluster


# ---------------------------------------------------------------------------
# default_normalizer
# ---------------------------------------------------------------------------

class TestDefaultNormalizer:
    def test_extracts_last_number(self):
        assert default_normalizer("The answer is 42.") == "42"

    def test_extracts_negative_number(self):
        assert default_normalizer("Result: -7") == "-7"

    def test_extracts_decimal(self):
        assert default_normalizer("3.14 is pi") == "3.14"

    def test_last_number_wins(self):
        assert default_normalizer("Step 1 gives 3, step 2 gives 7.") == "7"

    def test_fallback_strips_punctuation(self):
        result = default_normalizer("Yes!")
        assert result == "yes"

    def test_empty_string(self):
        result = default_normalizer("")
        assert result == ""


# ---------------------------------------------------------------------------
# _consistency_features
# ---------------------------------------------------------------------------

class TestConsistencyFeatures:
    def test_single_sample_returns_zero(self):
        sc, ent = _consistency_features(["42"])
        assert sc == 0.0
        assert ent == 0.0

    def test_full_agreement_zero_disagreement(self):
        sc, ent = _consistency_features(["42", "42", "42"])
        assert sc == pytest.approx(0.0)
        assert ent == pytest.approx(0.0)

    def test_uniform_distribution_max_entropy(self):
        # 4 unique answers with 4 samples -> maximum entropy
        samples = ["1", "2", "3", "4"]
        sc, ent = _consistency_features(samples, normalizer=verbatim_normalizer)
        assert sc == pytest.approx(0.75)      # 1 - 1/4
        assert ent == pytest.approx(1.0, abs=1e-6)

    def test_partial_agreement(self):
        # 3 agree on "42", 1 disagrees
        samples = ["42", "42", "42", "0"]
        sc, ent = _consistency_features(samples, normalizer=verbatim_normalizer)
        assert sc == pytest.approx(1.0 - 3/4)

    def test_entropy_in_unit_interval(self):
        import random
        rng = random.Random(0)
        for _ in range(20):
            k = rng.randint(2, 10)
            samples = [str(rng.randint(0, 3)) for _ in range(k)]
            sc, ent = _consistency_features(samples, normalizer=verbatim_normalizer)
            assert 0.0 <= sc <= 1.0
            assert 0.0 <= ent <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# RuntimeFeatures
# ---------------------------------------------------------------------------

class TestRuntimeFeatures:
    def test_default_vector_shape(self):
        f = RuntimeFeatures()
        v = f.as_vector()
        assert v.shape == (FEATURE_DIM,)
        assert v.dtype == np.float64

    def test_default_neutral_values(self):
        f = RuntimeFeatures()
        v = f.as_vector()
        assert v[FEATURE_NAMES.index("input_length_norm")] == 0.0
        assert v[FEATURE_NAMES.index("sc_disagreement")] == 0.0
        assert v[FEATURE_NAMES.index("verifier_score")] == pytest.approx(0.5)
        assert v[FEATURE_NAMES.index("tool_success")] == 0.0
        assert v[FEATURE_NAMES.index("tool_failure")] == 0.0

    def test_unknown_cluster_all_zero_onehot(self):
        f = RuntimeFeatures(embed_cluster=-1)
        v = f.as_vector()
        cluster_slice = v[6:]   # embed_cluster_0..3
        assert np.all(cluster_slice == 0.0)

    def test_valid_cluster_one_hot(self):
        for c in range(N_CLUSTERS):
            f = RuntimeFeatures(embed_cluster=c)
            v = f.as_vector()
            cluster_slice = v[6:]
            expected = np.zeros(N_CLUSTERS)
            expected[c] = 1.0
            assert np.allclose(cluster_slice, expected)

    def test_as_vector_subset_matches_full(self):
        f = RuntimeFeatures(input_length_norm=0.7, sc_disagreement=0.3)
        full = f.as_vector()
        subset_names = ["input_length_norm", "sc_disagreement"]
        subset = f.as_vector_subset(subset_names)
        assert subset[0] == pytest.approx(full[FEATURE_NAMES.index("input_length_norm")])
        assert subset[1] == pytest.approx(full[FEATURE_NAMES.index("sc_disagreement")])

    def test_with_tool_outcome_success(self):
        f = RuntimeFeatures().with_tool_outcome(True)
        assert f.tool_success == 1.0
        assert f.tool_failure == 0.0

    def test_with_tool_outcome_failure(self):
        f = RuntimeFeatures().with_tool_outcome(False)
        assert f.tool_success == 0.0
        assert f.tool_failure == 1.0

    def test_with_verifier_clips(self):
        f = RuntimeFeatures().with_verifier(1.5)
        assert f.verifier_score == pytest.approx(1.0)
        f2 = RuntimeFeatures().with_verifier(-0.3)
        assert f2.verifier_score == pytest.approx(0.0)

    def test_serialization_roundtrip(self):
        f = RuntimeFeatures(
            input_length_norm=0.42,
            sc_disagreement=0.1,
            entropy_approx=0.3,
            verifier_score=0.9,
            tool_success=1.0,
            tool_failure=0.0,
            embed_cluster=2,
        )
        f2 = RuntimeFeatures.from_dict(f.to_dict())
        assert np.allclose(f.as_vector(), f2.as_vector())


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def test_pre_lm_returns_runtime_features(self):
        fx = FeatureExtractor(max_input_tokens=100)
        feat = fx.pre_lm("Hello world")
        assert isinstance(feat, RuntimeFeatures)

    def test_pre_lm_length_norm_whitespace_proxy(self):
        fx = FeatureExtractor(max_input_tokens=10)
        feat = fx.pre_lm("a b c d e")   # 5 whitespace tokens
        assert feat.input_length_norm == pytest.approx(0.5)

    def test_pre_lm_clips_at_one(self):
        fx = FeatureExtractor(max_input_tokens=5)
        feat = fx.pre_lm("a b c d e f g h")  # 8 tokens > max 5
        assert feat.input_length_norm == pytest.approx(1.0)

    def test_pre_lm_neutral_post_fields(self):
        fx = FeatureExtractor()
        feat = fx.pre_lm("test")
        assert feat.sc_disagreement == 0.0
        assert feat.entropy_approx == 0.0
        assert feat.verifier_score == pytest.approx(0.5)

    def test_post_lm_full_agreement(self):
        fx = FeatureExtractor()
        feat = fx.post_lm("What is 2+2?", samples=["4", "4", "4"])
        assert feat.sc_disagreement == pytest.approx(0.0)
        assert feat.entropy_approx == pytest.approx(0.0)

    def test_post_lm_full_disagreement(self):
        fx = FeatureExtractor(normalizer=verbatim_normalizer)
        feat = fx.post_lm("Q", samples=["a", "b", "c", "d"])
        assert feat.sc_disagreement == pytest.approx(0.75)
        assert feat.entropy_approx == pytest.approx(1.0, abs=1e-6)

    def test_post_lm_with_verifier(self):
        fx = FeatureExtractor()
        feat = fx.post_lm("Q", samples=["42"], verifier_score=0.8)
        assert feat.verifier_score == pytest.approx(0.8)

    def test_post_lm_with_tool_success(self):
        fx = FeatureExtractor()
        feat = fx.post_lm("Q", samples=["ok"], tool_success=True)
        assert feat.tool_success == 1.0
        assert feat.tool_failure == 0.0

    def test_post_lm_with_tool_failure(self):
        fx = FeatureExtractor()
        feat = fx.post_lm("Q", samples=["err"], tool_success=False)
        assert feat.tool_success == 0.0
        assert feat.tool_failure == 1.0

    def test_custom_tokenizer(self):
        # tokenizer returns exact char count
        fx = FeatureExtractor(
            max_input_tokens=10,
            tokenizer=lambda t: len(t),
        )
        feat = fx.pre_lm("hello")   # 5 chars / 10 = 0.5
        assert feat.input_length_norm == pytest.approx(0.5)

    def test_hash_cluster_deterministic(self):
        fx = FeatureExtractor()
        f1 = fx.pre_lm("same input")
        f2 = fx.pre_lm("same input")
        assert f1.embed_cluster == f2.embed_cluster

    def test_hash_cluster_in_range(self):
        fx = FeatureExtractor()
        for text in ["short", "a" * 100, "1234", "test input here"]:
            feat = fx.pre_lm(text)
            assert 0 <= feat.embed_cluster < N_CLUSTERS

    def test_vector_shape_from_post_lm(self):
        fx = FeatureExtractor()
        feat = fx.post_lm("Q", samples=["a", "b"])
        assert feat.as_vector().shape == (FEATURE_DIM,)


# ---------------------------------------------------------------------------
# FEATURE_NAMES / FEATURE_DIM consistency
# ---------------------------------------------------------------------------

class TestFeatureSchema:
    def test_feature_dim_matches_names(self):
        assert FEATURE_DIM == len(FEATURE_NAMES)

    def test_all_names_unique(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_cluster_slots_count(self):
        cluster_slots = [n for n in FEATURE_NAMES if n.startswith("embed_cluster_")]
        assert len(cluster_slots) == N_CLUSTERS

    def test_guard_uses_same_feature_dim(self):
        from ppg.core import Guard, FEATURE_DIM as GFD
        assert GFD == FEATURE_DIM
        g = Guard.all_pass()
        assert len(g.weights) == FEATURE_DIM
