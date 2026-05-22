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
    Key: SHA-256 of the prompt (hex).
    Good for: offline dev, replay experiments, API cost control.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Optional


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OpenAIConfig:
    model:              str             = "gpt-4o-mini"
    temperature:        float           = 0.0
    max_tokens:         int             = 512
    timeout:            float           = 30.0
    max_retries:        int             = 3
    parse_retries:      int             = 2
    system_msg:         str             = "You are a helpful assistant."
    sample_temperature: Optional[float] = None


@dataclass
class AnthropicConfig:
    model:              str             = "claude-haiku-4-5-20251001"
    temperature:        float           = 0.0
    max_tokens:         int             = 512
    timeout:            float           = 30.0
    max_retries:        int             = 3
    parse_retries:      int             = 2
    system_msg:         str             = "You are a helpful assistant."
    sample_temperature: Optional[float] = None


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------

def _retry_parse_errors(fn, *, retries: int):
    """
    Retry SDK response-parse failures that usually mean an empty/non-JSON body.

    The official SDK handles normal transport/status retries. This outer guard
    covers the raw JSONDecodeError that can still surface during high-throughput
    benchmark runs before the SDK can wrap it in an API exception.
    """
    attempts = max(0, retries) + 1
    for attempt in range(attempts):
        try:
            return fn()
        except JSONDecodeError:
            if attempt == attempts - 1:
                raise
            _sleep_before_retry(attempt)

    raise RuntimeError("unreachable")


def _sleep_before_retry(attempt: int) -> None:
    delay = min(4.0, 0.5 * (2 ** attempt))
    jitter = 0.75 + random.random() * 0.5
    time.sleep(delay * jitter)


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
        response = _retry_parse_errors(
            lambda: self._client.chat.completions.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                messages=[
                    {"role": "system",  "content": self.cfg.system_msg},
                    {"role": "user",    "content": prompt},
                ],
            ),
            retries=self.cfg.parse_retries,
        )
        return response.choices[0].message.content or ""

    def sample(self, prompt: str, n: int) -> list[str]:
        """Return n independent completions for self-consistency decoding."""
        if n <= 1:
            return [self.complete(prompt)]
        response = _retry_parse_errors(
            lambda: self._client.chat.completions.create(
                model=self.cfg.model,
                temperature=(
                    self.cfg.sample_temperature
                    if self.cfg.sample_temperature is not None
                    else self.cfg.temperature
                ),
                max_tokens=self.cfg.max_tokens,
                n=n,
                messages=[
                    {"role": "system", "content": self.cfg.system_msg},
                    {"role": "user", "content": prompt},
                ],
            ),
            retries=self.cfg.parse_retries,
        )
        return [choice.message.content or "" for choice in response.choices]


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
        message = _retry_parse_errors(
            lambda: self._client.messages.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                system=self.cfg.system_msg,
                messages=[{"role": "user", "content": prompt}],
            ),
            retries=self.cfg.parse_retries,
        )
        return message.content[0].text if message.content else ""

    def sample(self, prompt: str, n: int) -> list[str]:
        """Return n completions for self-consistency decoding."""
        if n <= 1:
            return [self.complete(prompt)]
        temperature = (
            self.cfg.sample_temperature
            if self.cfg.sample_temperature is not None
            else self.cfg.temperature
        )
        samples = []
        for _ in range(n):
            message = _retry_parse_errors(
                lambda: self._client.messages.create(
                    model=self.cfg.model,
                    temperature=temperature,
                    max_tokens=self.cfg.max_tokens,
                    system=self.cfg.system_msg,
                    messages=[{"role": "user", "content": prompt}],
                ),
                retries=self.cfg.parse_retries,
            )
            samples.append(message.content[0].text if message.content else "")
        return samples


# ---------------------------------------------------------------------------
# DiskCachedLMClient
# ---------------------------------------------------------------------------

class CountingLMClient:
    """
    Wraps any LMClient and counts every complete() call.
    Thread-safe — safe to use with PPGTrainer(n_workers > 1).

    reset() returns the current count and atomically resets to zero.
    """

    def __init__(self, lm):
        self._lm    = lm
        self._count = 0
        self._lock  = threading.Lock()

    def complete(self, prompt: str) -> str:
        with self._lock:
            self._count += 1
        return self._lm.complete(prompt)

    def sample(self, prompt: str, n: int) -> list[str]:
        """Count sampled completions by generated completion, not request."""
        if n <= 1:
            return [self.complete(prompt)]
        with self._lock:
            self._count += n
        sampler = getattr(self._lm, "sample", None)
        if callable(sampler):
            return list(sampler(prompt, n))
        return [self._lm.complete(prompt) for _ in range(n)]

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._count

    def reset(self) -> int:
        """Return current count and reset to zero."""
        with self._lock:
            n = self._count
            self._count = 0
            return n


class DiskCachedLMClient:
    """
    Caches LM responses to a JSON file on disk.

    On first complete(prompt) call, checks cache. Hit: returns stored
    response immediately. Miss: calls wrapped LM, stores result, returns.

    Cache is loaded lazily on first use and written atomically after misses.
    Thread-safe within one process. Concurrent processes can share the same
    cache file for best-effort reuse, but last writer wins if they write at
    the same time.

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
        self._lock   = threading.Lock()   # guards cache dict + disk flush

        # Diagnostics
        self._n_hits:  int = 0
        self._n_misses: int = 0

    # ------------------------------------------------------------------
    # LMClient protocol
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> str:
        self._ensure_loaded()
        key = self._hash(prompt)

        with self._lock:
            if key in self._cache:
                self._n_hits += 1
                return self._cache[key]

        # LM call outside lock — allows concurrent in-flight calls for distinct prompts
        response = self._lm.complete(prompt)

        with self._lock:
            # Re-check: another thread may have populated same key concurrently
            if key not in self._cache:
                self._cache[key] = response
                self._n_misses += 1
                self._flush()
            else:
                self._n_hits += 1  # concurrent hit

        return response

    def sample(self, prompt: str, n: int) -> list[str]:
        """
        Cache self-consistency samples separately from deterministic complete().

        Each sample index gets its own cache key. This preserves reproducible
        replay while avoiding the prompt-only cache collapse where k samples all
        become the first cached completion.
        """
        if n <= 1:
            return [self.complete(prompt)]

        self._ensure_loaded()
        keys = [self._hash(f"{prompt}\0sample:{i}") for i in range(n)]
        samples: list[Optional[str]] = [None] * n
        missing: list[int] = []

        with self._lock:
            for i, key in enumerate(keys):
                if key in self._cache:
                    self._n_hits += 1
                    samples[i] = self._cache[key]
                else:
                    missing.append(i)

        if missing:
            sampler = getattr(self._lm, "sample", None)
            if callable(sampler):
                fresh = list(sampler(prompt, len(missing)))
            else:
                fresh = [self._lm.complete(prompt) for _ in missing]

            if len(fresh) < len(missing):
                fresh.extend(self._lm.complete(prompt) for _ in range(len(missing) - len(fresh)))

            with self._lock:
                for i, response in zip(missing, fresh):
                    key = keys[i]
                    if key not in self._cache:
                        self._cache[key] = response
                        self._n_misses += 1
                    else:
                        self._n_hits += 1
                    samples[i] = self._cache[key]
                self._flush()

        return [sample or "" for sample in samples]

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

    def reset_stats(self) -> tuple[int, int]:
        """Return (hits, misses) since last reset and zero the counters."""
        h, m = self._n_hits, self._n_misses
        self._n_hits = 0
        self._n_misses = 0
        return h, m

    def clear(self) -> None:
        """Remove all cached entries and delete the cache file."""
        with self._lock:
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
        with self._lock:
            if self._cache is not None:
                return
            self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        if not os.path.exists(self._cache_path):
            return {}

        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, JSONDecodeError):
            self._quarantine_cache_file()
            return {}

        if not isinstance(data, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in data.items()
        ):
            self._quarantine_cache_file()
            return {}

        return data

    def _quarantine_cache_file(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        suffix = f".corrupt.{int(time.time() * 1000)}.{os.getpid()}"
        try:
            os.replace(self._cache_path, self._cache_path + suffix)
        except OSError:
            pass

    def _flush(self) -> None:
        cache_dir = os.path.dirname(os.path.abspath(self._cache_path))
        os.makedirs(cache_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(self._cache_path) + ".",
            suffix=".tmp",
            dir=cache_dir,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=None)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._cache_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
