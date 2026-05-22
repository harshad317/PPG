"""Tests for ppg/lm/clients.py."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from ppg.lm.clients import (
    AnthropicClient,
    AnthropicConfig,
    CountingLMClient,
    DiskCachedLMClient,
    OpenAIClient,
    OpenAIConfig,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class FixedLM:
    def __init__(self, response: str = "42"):
        self.response = response
        self.n_calls  = 0

    def complete(self, prompt: str) -> str:
        self.n_calls += 1
        return self.response


class SamplingLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.n_sample_calls = 0
        self.n_complete_calls = 0

    def complete(self, prompt: str) -> str:
        self.n_complete_calls += 1
        return self.responses[0]

    def sample(self, prompt: str, n: int) -> list[str]:
        self.n_sample_calls += 1
        return self.responses[:n]


def cache_path_in(d: str) -> str:
    return os.path.join(d, "cache.json")


# ---------------------------------------------------------------------------
# OpenAIConfig
# ---------------------------------------------------------------------------

class TestOpenAIConfig:
    def test_defaults(self):
        cfg = OpenAIConfig()
        assert cfg.model       == "gpt-4o-mini"
        assert cfg.temperature == pytest.approx(0.0)
        assert cfg.max_tokens  == 512
        assert cfg.max_retries == 3
        assert cfg.parse_retries == 2

    def test_custom(self):
        cfg = OpenAIConfig(model="gpt-4o", max_tokens=1024, temperature=0.7)
        assert cfg.model      == "gpt-4o"
        assert cfg.max_tokens == 1024


# ---------------------------------------------------------------------------
# AnthropicConfig
# ---------------------------------------------------------------------------

class TestAnthropicConfig:
    def test_defaults(self):
        cfg = AnthropicConfig()
        assert "claude" in cfg.model
        assert cfg.temperature == pytest.approx(0.0)
        assert cfg.max_tokens  == 512
        assert cfg.max_retries == 3
        assert cfg.parse_retries == 2

    def test_custom(self):
        cfg = AnthropicConfig(model="claude-opus-4-7", max_tokens=256)
        assert cfg.model      == "claude-opus-4-7"
        assert cfg.max_tokens == 256


# ---------------------------------------------------------------------------
# OpenAIClient — mocked
# ---------------------------------------------------------------------------

def _make_openai_mock_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_openai_mock_choices(texts: list[str]):
    choices = []
    for text in texts:
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        choices.append(choice)
    resp = MagicMock()
    resp.choices = choices
    return resp


class TestOpenAIClient:
    def test_import_error_without_package(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                OpenAIClient()

    def test_complete_returns_response_text(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_mock_response("The answer is 42")
        )
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = OpenAIClient(api_key="test-key")
            result = client.complete("What is 6×7?")
        assert result == "The answer is 42"

    def test_complete_passes_model_and_temperature(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_mock_response("ok")
        )
        cfg = OpenAIConfig(model="gpt-4o", temperature=0.5, max_tokens=64)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = OpenAIClient(config=cfg, api_key="k")
            client.complete("hi")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"]       == "gpt-4o"
        assert call_kwargs["temperature"] == pytest.approx(0.5)
        assert call_kwargs["max_tokens"]  == 64

    def test_prompt_sent_as_user_message(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_mock_response("ok")
        )
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = OpenAIClient(api_key="k")
            client.complete("Test prompt")
        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert user_msgs[0]["content"] == "Test prompt"

    def test_empty_content_returns_empty_string(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_mock_response("")
        )
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = OpenAIClient(api_key="k")
            result = client.complete("hi")
        assert result == ""

    def test_max_retries_passed_to_sdk(self):
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        cfg = OpenAIConfig(max_retries=5)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            OpenAIClient(config=cfg, api_key="k")
        init_kwargs = mock_openai.OpenAI.call_args.kwargs
        assert init_kwargs["max_retries"] == 5

    def test_sample_requests_n_choices_with_sample_temperature(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            _make_openai_mock_choices(["a", "b", "c"])
        )
        cfg = OpenAIConfig(temperature=0.0, sample_temperature=0.7)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = OpenAIClient(config=cfg, api_key="k")
            samples = client.sample("prompt", 3)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["n"] == 3
        assert call_kwargs["temperature"] == pytest.approx(0.7)
        assert samples == ["a", "b", "c"]

    def test_complete_retries_json_decode_error(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            json.JSONDecodeError("Expecting value", "", 0),
            _make_openai_mock_response("recovered"),
        ]
        cfg = OpenAIConfig(parse_retries=1)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            with patch("ppg.lm.clients.time.sleep"):
                client = OpenAIClient(config=cfg, api_key="k")
                result = client.complete("prompt")
        assert result == "recovered"
        assert mock_client.chat.completions.create.call_count == 2

    def test_sample_retries_json_decode_error(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            json.JSONDecodeError("Expecting value", "", 0),
            _make_openai_mock_choices(["a", "b"]),
        ]
        cfg = OpenAIConfig(parse_retries=1)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            with patch("ppg.lm.clients.time.sleep"):
                client = OpenAIClient(config=cfg, api_key="k")
                samples = client.sample("prompt", 2)
        assert samples == ["a", "b"]
        assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# AnthropicClient — mocked
# ---------------------------------------------------------------------------

def _make_anthropic_mock_response(text: str):
    content_block = MagicMock()
    content_block.text = text
    resp = MagicMock()
    resp.content = [content_block]
    return resp


class TestAnthropicClient:
    def test_import_error_without_package(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                AnthropicClient()

    def test_complete_returns_response_text(self):
        mock_anthropic = MagicMock()
        mock_client    = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = (
            _make_anthropic_mock_response("Forty-two")
        )
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicClient(api_key="test-key")
            result = client.complete("What is 6×7?")
        assert result == "Forty-two"

    def test_complete_passes_model_and_temperature(self):
        mock_anthropic = MagicMock()
        mock_client    = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = (
            _make_anthropic_mock_response("ok")
        )
        cfg = AnthropicConfig(model="claude-opus-4-7", temperature=0.3, max_tokens=128)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicClient(config=cfg, api_key="k")
            client.complete("hi")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"]       == "claude-opus-4-7"
        assert call_kwargs["temperature"] == pytest.approx(0.3)
        assert call_kwargs["max_tokens"]  == 128

    def test_prompt_sent_as_user_message(self):
        mock_anthropic = MagicMock()
        mock_client    = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = (
            _make_anthropic_mock_response("ok")
        )
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicClient(api_key="k")
            client.complete("My prompt")
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        assert messages[0]["role"]    == "user"
        assert messages[0]["content"] == "My prompt"

    def test_empty_content_list_returns_empty_string(self):
        mock_anthropic = MagicMock()
        mock_client    = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        resp = MagicMock()
        resp.content = []
        mock_client.messages.create.return_value = resp
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicClient(api_key="k")
            result = client.complete("hi")
        assert result == ""

    def test_max_retries_passed_to_sdk(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = MagicMock()
        cfg = AnthropicConfig(max_retries=7)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            AnthropicClient(config=cfg, api_key="k")
        init_kwargs = mock_anthropic.Anthropic.call_args.kwargs
        assert init_kwargs["max_retries"] == 7

    def test_sample_uses_sample_temperature(self):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_anthropic_mock_response("a"),
            _make_anthropic_mock_response("b"),
        ]
        cfg = AnthropicConfig(temperature=0.0, sample_temperature=0.8)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicClient(config=cfg, api_key="k")
            samples = client.sample("prompt", 2)
        assert samples == ["a", "b"]
        temperatures = [
            call.kwargs["temperature"]
            for call in mock_client.messages.create.call_args_list
        ]
        assert temperatures == [pytest.approx(0.8), pytest.approx(0.8)]


# ---------------------------------------------------------------------------
# DiskCachedLMClient
# ---------------------------------------------------------------------------

class TestDiskCachedLMClientBasic:
    def test_complete_returns_response(self):
        with tempfile.TemporaryDirectory() as d:
            wrapped = FixedLM("hello")
            cached  = DiskCachedLMClient(wrapped, cache_path_in(d))
            result  = cached.complete("hi")
            assert result == "hello"

    def test_miss_calls_wrapped_lm(self):
        with tempfile.TemporaryDirectory() as d:
            wrapped = FixedLM("42")
            cached  = DiskCachedLMClient(wrapped, cache_path_in(d))
            cached.complete("q1")
            assert wrapped.n_calls == 1

    def test_hit_does_not_call_wrapped_lm(self):
        with tempfile.TemporaryDirectory() as d:
            wrapped = FixedLM("42")
            cached  = DiskCachedLMClient(wrapped, cache_path_in(d))
            cached.complete("same prompt")
            cached.complete("same prompt")
            assert wrapped.n_calls == 1

    def test_different_prompts_both_miss(self):
        with tempfile.TemporaryDirectory() as d:
            wrapped = FixedLM("x")
            cached  = DiskCachedLMClient(wrapped, cache_path_in(d))
            cached.complete("prompt A")
            cached.complete("prompt B")
            assert wrapped.n_calls == 2

    def test_hit_returns_cached_value(self):
        with tempfile.TemporaryDirectory() as d:
            wrapped = FixedLM("first")
            cached  = DiskCachedLMClient(wrapped, cache_path_in(d))
            cached.complete("q")
            # Change wrapped response — hit should still return "first"
            wrapped.response = "second"
            result = cached.complete("q")
            assert result == "first"

    def test_sample_uses_distinct_cache_entries(self):
        with tempfile.TemporaryDirectory() as d:
            wrapped = SamplingLM(["a", "b", "c"])
            cached = DiskCachedLMClient(wrapped, cache_path_in(d))

            first = cached.sample("q", 3)
            second = cached.sample("q", 3)

            assert first == ["a", "b", "c"]
            assert second == ["a", "b", "c"]
            assert wrapped.n_sample_calls == 1
            assert cached.cache_size() == 3


class TestCountingLMClient:
    def test_sample_counts_generated_completions(self):
        wrapped = SamplingLM(["a", "b", "c"])
        counted = CountingLMClient(wrapped)

        assert counted.sample("q", 3) == ["a", "b", "c"]
        assert counted.call_count == 3


class TestDiskCachedLMClientDiagnostics:
    def test_n_hits_n_misses_initial(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            assert cached.n_hits   == 0
            assert cached.n_misses == 0

    def test_n_misses_increments(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            cached.complete("a")
            cached.complete("b")
            assert cached.n_misses == 2

    def test_n_hits_increments(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            cached.complete("q")
            cached.complete("q")
            cached.complete("q")
            assert cached.n_hits   == 2
            assert cached.n_misses == 1

    def test_hit_rate_zero_initially(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            assert cached.hit_rate == pytest.approx(0.0)

    def test_hit_rate_all_misses(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            cached.complete("a")
            cached.complete("b")
            assert cached.hit_rate == pytest.approx(0.0)

    def test_hit_rate_mixed(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            cached.complete("q")   # miss
            cached.complete("q")   # hit
            cached.complete("q")   # hit
            # 2 hits / 3 total
            assert cached.hit_rate == pytest.approx(2 / 3)

    def test_cache_size_zero_before_use(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            assert cached.cache_size() == 0

    def test_cache_size_grows_on_misses(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            cached.complete("a")
            cached.complete("b")
            assert cached.cache_size() == 2

    def test_cache_size_stable_on_hits(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            cached.complete("q")
            cached.complete("q")
            assert cached.cache_size() == 1


class TestDiskCachedLMClientPersistence:
    def test_cache_file_created_on_miss(self):
        with tempfile.TemporaryDirectory() as d:
            path = cache_path_in(d)
            cached = DiskCachedLMClient(FixedLM("hi"), path)
            cached.complete("prompt")
            assert os.path.exists(path)

    def test_cache_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = cache_path_in(d)
            cached = DiskCachedLMClient(FixedLM("hello"), path)
            cached.complete("q1")
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert len(data) == 1

    def test_corrupt_cache_file_is_quarantined(self):
        with tempfile.TemporaryDirectory() as d:
            path = cache_path_in(d)
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"broken":')

            cached = DiskCachedLMClient(FixedLM("fresh"), path)
            assert cached.complete("prompt") == "fresh"

            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert len(data) == 1
            assert any(name.startswith("cache.json.corrupt.") for name in os.listdir(d))

    def test_parallel_first_use_keeps_cache_valid(self):
        with tempfile.TemporaryDirectory() as d:
            path = cache_path_in(d)
            cached = DiskCachedLMClient(FixedLM("ok"), path)

            prompts = [f"prompt-{i}" for i in range(20)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(cached.complete, prompts))

            assert results == ["ok"] * len(prompts)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert len(data) == len(prompts)

    def test_cache_survives_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = cache_path_in(d)
            # First instance writes cache
            c1 = DiskCachedLMClient(FixedLM("stored"), path)
            c1.complete("persistent prompt")
            # Second instance reads cache — wrapped LM never called
            inner = FixedLM("should not be returned")
            c2 = DiskCachedLMClient(inner, path)
            result = c2.complete("persistent prompt")
            assert result == "stored"
            assert inner.n_calls == 0

    def test_cache_key_is_sha256_of_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            path = cache_path_in(d)
            cached = DiskCachedLMClient(FixedLM("r"), path)
            prompt = "hello world"
            cached.complete(prompt)
            with open(path) as f:
                data = json.load(f)
            expected_key = hashlib.sha256(prompt.encode()).hexdigest()
            assert expected_key in data
            assert data[expected_key] == "r"

    def test_multiple_entries_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            path = cache_path_in(d)
            wrapped = FixedLM("x")
            cached  = DiskCachedLMClient(wrapped, path)
            for i in range(5):
                wrapped.response = f"resp{i}"
                cached.complete(f"prompt{i}")
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 5


class TestDiskCachedLMClientClear:
    def test_clear_resets_cache_size(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM("x"), cache_path_in(d))
            cached.complete("a")
            cached.complete("b")
            cached.clear()
            assert cached.cache_size() == 0

    def test_clear_removes_cache_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = cache_path_in(d)
            cached = DiskCachedLMClient(FixedLM("x"), path)
            cached.complete("q")
            assert os.path.exists(path)
            cached.clear()
            assert not os.path.exists(path)

    def test_clear_resets_counters(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM("x"), cache_path_in(d))
            cached.complete("q")
            cached.complete("q")
            cached.clear()
            assert cached.n_hits   == 0
            assert cached.n_misses == 0

    def test_complete_works_after_clear(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM("42"), cache_path_in(d))
            cached.complete("q")
            cached.clear()
            result = cached.complete("q")
            assert result == "42"

    def test_clear_no_file_no_error(self):
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            cached.clear()  # file never created — should not raise


class TestDiskCachedLMClientProtocol:
    def test_satisfies_lm_client_protocol(self):
        from ppg.core.executor import LMClient
        with tempfile.TemporaryDirectory() as d:
            cached = DiskCachedLMClient(FixedLM(), cache_path_in(d))
            assert isinstance(cached, LMClient)

    def test_nested_caches_work(self):
        """DiskCachedLMClient wrapping another DiskCachedLMClient."""
        with tempfile.TemporaryDirectory() as d:
            inner  = DiskCachedLMClient(FixedLM("deep"), os.path.join(d, "inner.json"))
            outer  = DiskCachedLMClient(inner, os.path.join(d, "outer.json"))
            result = outer.complete("q")
            assert result == "deep"
