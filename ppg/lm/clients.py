"""
LMClient implementations for PPG.

All classes satisfy the LMClient protocol from ppg.core.executor:
    def complete(self, prompt: str) -> str

OpenAIClient / AnthropicClient
    Thin wrappers around the respective SDKs.
    SDK-native retry/backoff enabled via max_retries.
    Lazy import: ImportError is raised only when the class is instantiated,
    not at module load time, so tests that don't use the class can skip.

DiskCachedLMClient
    Wraps any LMClient and caches responses to a JSON file on disk.
    Key: SHA-256 of the prompt (hex). Thread-unsafe — single-process use only.
    Good for: offline dev, replay experiments, API cost control.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OpenAIConfig:
    model:       str   = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens:  int   = 512
    timeout:     float = 30.0
    max_retries: int   = 3
    system_msg:  str   = "You are a helpful assistant."


@dataclass
class AnthropicConfig:
    model:       str   = "claude-haiku-4-5-20251001"
    temperature: float = 0.0
    max_tokens:  int   = 512
    timeout:     float = 30.0
    max_retries: int   = 3
    system_msg:  str   = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------

class OpenAIClient:
    """
    Wraps openai.OpenAI to satisfy LMClient protocol.

    Parameters
    ----------
    config  : OpenAIConfig
    api_key : overrides OPENAI_API_KEY env var when provided
    """

    def __init__(
        self,
        config:  Optional[OpenAIConfig] = None,
        api_key: Optional[str] = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required: pip install openai"
            ) from None

        self.cfg = config or OpenAIConfig()
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=self.cfg.timeout,
            max_retries=self.cfg.max_retries,
        )

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {"role": "system",  "content": self.cfg.system_msg},
                {"role": "user",    "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------

class AnthropicClient:
    """
    Wraps anthropic.Anthropic to satisfy LMClient protocol.

    Parameters
    ----------
    config  : AnthropicConfig
    api_key : overrides ANTHROPIC_API_KEY env var when provided
    """

    def __init__(
        self,
        config:  Optional[AnthropicConfig] = None,
        api_key: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required: pip install anthropic"
            ) from None

        self.cfg = config or AnthropicConfig()
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=self.cfg.timeout,
            max_retries=self.cfg.max_retries,
        )

    def complete(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            system=self.cfg.system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text if message.content else ""


# ---------------------------------------------------------------------------
# DiskCachedLMClient
# ---------------------------------------------------------------------------

class DiskCachedLMClient:
    """
    Caches LM responses to a JSON file on disk.

    On first complete(prompt) call, checks cache. Hit: returns stored
    response immediately. Miss: calls wrapped LM, stores result, returns.

    Cache is loaded lazily on first use and written after every miss.
    Thread-unsafe — single-process only.

    Parameters
    ----------
    lm        : any LMClient (wrapped)
    cache_path : path to the JSON cache file
                 (created automatically if it doesn't exist)
    """

    def __init__(self, lm, cache_path: str):
        self._lm         = lm
        self._cache_path = cache_path
        self._cache:  Optional[dict[str, str]] = None  # lazy load

        # Diagnostics
        self._n_hits:  int = 0
        self._n_misses: int = 0

    # ------------------------------------------------------------------
    # LMClient protocol
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> str:
        self._ensure_loaded()
        key = self._hash(prompt)
        if key in self._cache:
            self._n_hits += 1
            return self._cache[key]
        response = self._lm.complete(prompt)
        self._cache[key] = response
        self._n_misses += 1
        self._flush()
        return response

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_hits(self) -> int:
        return self._n_hits

    @property
    def n_misses(self) -> int:
        return self._n_misses

    @property
    def hit_rate(self) -> float:
        total = self._n_hits + self._n_misses
        return self._n_hits / total if total > 0 else 0.0

    def cache_size(self) -> int:
        """Number of cached prompt-response pairs."""
        self._ensure_loaded()
        return len(self._cache)

    def clear(self) -> None:
        """Remove all cached entries and delete the cache file."""
        self._cache = {}
        if os.path.exists(self._cache_path):
            os.remove(self._cache_path)
        self._n_hits = 0
        self._n_misses = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _ensure_loaded(self) -> None:
        if self._cache is not None:
            return
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        else:
            self._cache = {}

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._cache_path)), exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=None)
