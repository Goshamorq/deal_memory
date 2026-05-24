"""Shared test fixtures."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest


class FakeLLMClient:
    """Replays canned responses in order. Lets tests run without network.

    Pass `responses` as a list of strings (returned in order) OR a callable
    that receives kwargs of each .chat() call and returns the string.
    """

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


def make_synth_response(
    *,
    transcript: str,
    budget: str | None = None,
    decision_maker: str | None = None,
    technical_requirements: str | None = None,
    objections: list[str] | None = None,
    promises: list[str] | None = None,
    next_step: str | None = None,
) -> str:
    """Build a JSON string matching the synth.py expected payload."""

    def field(value: str | None) -> dict[str, Any]:
        return {
            "value": value,
            "quote": value if value else None,
            "confidence": 1.0 if value else 0.0,
        }

    def items(values: list[str] | None) -> list[dict[str, Any]]:
        if not values:
            return []
        return [{"value": v, "quote": v, "confidence": 1.0} for v in values]

    payload = {
        "transcript": transcript,
        "ground_truth": {
            "budget": field(budget),
            "decision_maker": field(decision_maker),
            "technical_requirements": field(technical_requirements),
            "objections": items(objections),
            "promises": items(promises),
            "next_step": field(next_step),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


@pytest.fixture
def fake_llm_factory() -> Callable[..., FakeLLMClient]:
    """Returns a factory so tests can build a FakeLLMClient inline."""
    return FakeLLMClient
