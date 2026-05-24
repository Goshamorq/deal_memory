"""GigaChat client: OAuth + chat completions.

Single-file by design — Karpathy "no abstractions for single-use" until we
add a second LLM provider. `LLMClient` Protocol exists only so tests can
substitute a FakeLLMClient.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


@dataclass
class _Token:
    value: str
    expires_at: float  # unix seconds


class GigaChatClient:
    """Minimal GigaChat client with token caching and one 401-retry.

    Reads credentials from env on construction:
      - GIGACHAT_AUTH_KEY    (required, base64 client_id:secret)
      - GIGACHAT_SCOPE       (default GIGACHAT_API_PERS)
      - GIGACHAT_MODEL       (default GigaChat)
      - GIGACHAT_VERIFY_SSL  (default false — educational project)
    """

    def __init__(self, model: str | None = None) -> None:
        self._auth_key = os.environ.get("GIGACHAT_AUTH_KEY", "").strip()
        if not self._auth_key:
            raise RuntimeError(
                "GIGACHAT_AUTH_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self._model = model or os.environ.get("GIGACHAT_MODEL", "GigaChat")
        verify_env = os.environ.get("GIGACHAT_VERIFY_SSL", "false").lower()
        self._verify = verify_env in {"1", "true", "yes"}
        # Single transport with 2 retries on connect errors so a transient
        # 5xx mid-batch doesn't sink a 50-call run.
        transport = httpx.HTTPTransport(retries=2, verify=self._verify)
        self._client = httpx.Client(timeout=60.0, transport=transport, verify=self._verify)
        self._token: _Token | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GigaChatClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- OAuth ----

    def _fetch_token(self) -> _Token:
        resp = self._client.post(
            OAUTH_URL,
            headers={
                "Authorization": f"Basic {self._auth_key}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"scope": self._scope},
        )
        resp.raise_for_status()
        body = resp.json()
        # API returns expires_at in milliseconds since epoch.
        expires_at = float(body["expires_at"]) / 1000.0
        return _Token(value=body["access_token"], expires_at=expires_at)

    def _token_value(self) -> str:
        now = time.time()
        if self._token is None or self._token.expires_at - now < 30:
            self._token = self._fetch_token()
        return self._token.value

    # ---- Chat ----

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send chat completion; returns assistant message content string."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        for attempt in (1, 2):
            resp = self._client.post(
                CHAT_URL,
                headers={
                    "Authorization": f"Bearer {self._token_value()}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
            )
            if resp.status_code == 401 and attempt == 1:
                self._token = None
                continue
            resp.raise_for_status()
            break

        data = resp.json()
        return data["choices"][0]["message"]["content"]
