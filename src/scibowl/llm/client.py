from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass
class ChatClientConfig:
    provider: str
    model_name: str
    base_url: str
    api_key: str | None = None
    timeout_seconds: int = 120


class OpenAICompatibleChatClient:
    def __init__(self, config: ChatClientConfig) -> None:
        self.config = config

    @classmethod
    def from_env(
        cls,
        *,
        provider_env: str,
        model_env: str,
        base_url_env: str,
        api_key_env: str,
        timeout_env: str | None = None,
    ) -> "OpenAICompatibleChatClient | None":
        provider = os.getenv(provider_env, "").strip().lower()
        if provider not in {"openai_compatible", "openai-compatible"}:
            return None

        model_name = os.getenv(model_env, "").strip()
        base_url = os.getenv(base_url_env, "").strip()
        if not model_name or not base_url:
            raise ValueError(f"{model_env} and {base_url_env} must be set when {provider_env} is openai_compatible")

        return cls(
            ChatClientConfig(
                provider="openai_compatible",
                model_name=model_name,
                base_url=base_url.rstrip("/"),
                api_key=os.getenv(api_key_env, "").strip() or None,
                timeout_seconds=int(os.getenv(timeout_env, "120")) if timeout_env else 120,
            )
        )

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        req = request.Request(
            url=f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with status {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        message = body["choices"][0]["message"]["content"]
        if isinstance(message, list):
            message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        if not isinstance(message, str):
            raise RuntimeError("LLM response content was not text")
        return _extract_json_object(message)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = text[:500].replace("\n", "\\n")
        raise RuntimeError(f"Model returned invalid JSON: {exc}. Raw content prefix: {snippet}") from exc
