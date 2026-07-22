from __future__ import annotations

import os


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


def apply_runtime_model_overrides(
    *,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    writer_model: str | None = None,
    writer_base_url: str | None = None,
    verifier_model: str | None = None,
    verifier_base_url: str | None = None,
    writer_timeout_seconds: int | None = None,
    verifier_timeout_seconds: int | None = None,
    disable_writer_fallback: bool = False,
) -> None:
    resolved_ollama_base_url = ollama_base_url or DEFAULT_OLLAMA_BASE_URL

    if ollama_model:
        _set_env("SCIBOWL_WRITER_PROVIDER", "openai_compatible")
        _set_env("SCIBOWL_VERIFIER_PROVIDER", "openai_compatible")
        _set_env("SCIBOWL_WRITER_MODEL", writer_model or ollama_model)
        _set_env("SCIBOWL_VERIFIER_MODEL", verifier_model or ollama_model)
        _set_env("SCIBOWL_WRITER_BASE_URL", writer_base_url or resolved_ollama_base_url)
        _set_env("SCIBOWL_VERIFIER_BASE_URL", verifier_base_url or resolved_ollama_base_url)
        os.environ.pop("SCIBOWL_WRITER_API_KEY", None)
        os.environ.pop("SCIBOWL_VERIFIER_API_KEY", None)
    elif writer_model:
        _set_env("SCIBOWL_WRITER_PROVIDER", "openai_compatible")
        _set_env("SCIBOWL_WRITER_MODEL", writer_model)
    if writer_base_url:
        _set_env("SCIBOWL_WRITER_BASE_URL", writer_base_url)
    if verifier_model:
        _set_env("SCIBOWL_VERIFIER_PROVIDER", "openai_compatible")
        _set_env("SCIBOWL_VERIFIER_MODEL", verifier_model)
    if verifier_base_url:
        _set_env("SCIBOWL_VERIFIER_BASE_URL", verifier_base_url)
    if writer_timeout_seconds is not None:
        _set_env("SCIBOWL_WRITER_TIMEOUT_SECONDS", str(writer_timeout_seconds))
    if verifier_timeout_seconds is not None:
        _set_env("SCIBOWL_VERIFIER_TIMEOUT_SECONDS", str(verifier_timeout_seconds))
    if disable_writer_fallback:
        _set_env("SCIBOWL_DISABLE_WRITER_FALLBACK", "1")


def _set_env(name: str, value: str) -> None:
    os.environ[name] = value
