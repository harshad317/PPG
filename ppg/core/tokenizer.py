"""
Centralized token counting for PPG.

Uses tiktoken (cl100k_base) when available — covers GPT-4o, GPT-4.1, and
compatible models. Falls back to whitespace split when tiktoken is not installed.

cl100k_base is a good proxy for most API-backed models. For model-specific
accuracy swap the encoding name (e.g. o200k_base for GPT-4o / o-series).

Usage
-----
    from ppg.core.tokenizer import count_tokens
    n = count_tokens("Hello world")   # 2
"""

from __future__ import annotations

_enc = None

try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")
except Exception:
    # Covers ImportError (not installed) and network/SSL errors on first download.
    pass


def count_tokens(text: str) -> int:
    """Return token count for text. Uses tiktoken if available, else whitespace split."""
    if _enc is not None:
        return len(_enc.encode(text))
    return len(text.split())
