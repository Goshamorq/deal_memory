"""Shared test fixtures."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FakeLLMClient:
    """Replays canned responses in order. Lets tests run without network."""

    def __init__(self, responses: list[str] | Callable[..., str]) -> None:
        self._responses = responses
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        call = {
            "messages": messages,
            "temperature": temperature,
            "response_format": response_format,
            "max_tokens": max_tokens,
        }
        self.calls.append(call)
        if callable(self._responses):
            return self._responses(**call)
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp
