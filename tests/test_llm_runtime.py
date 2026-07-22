import os

from scibowl.llm.runtime import DEFAULT_OLLAMA_BASE_URL, apply_runtime_model_overrides


def test_apply_runtime_model_overrides_sets_ollama_writer_and_verifier_envs() -> None:
    previous = {name: os.environ.get(name) for name in _MODEL_ENV_NAMES}
    try:
        apply_runtime_model_overrides(ollama_model="qwen3.5:9b")

        assert os.environ["SCIBOWL_WRITER_PROVIDER"] == "openai_compatible"
        assert os.environ["SCIBOWL_VERIFIER_PROVIDER"] == "openai_compatible"
        assert os.environ["SCIBOWL_WRITER_MODEL"] == "qwen3.5:9b"
        assert os.environ["SCIBOWL_VERIFIER_MODEL"] == "qwen3.5:9b"
        assert os.environ["SCIBOWL_WRITER_BASE_URL"] == DEFAULT_OLLAMA_BASE_URL
        assert os.environ["SCIBOWL_VERIFIER_BASE_URL"] == DEFAULT_OLLAMA_BASE_URL
    finally:
        _restore_env(previous)


def test_apply_runtime_model_overrides_allows_individual_models() -> None:
    previous = {name: os.environ.get(name) for name in _MODEL_ENV_NAMES}
    try:
        apply_runtime_model_overrides(
            writer_model="qwen3.5:9b",
            writer_base_url="http://127.0.0.1:11434/v1",
            verifier_model="gpt-oss:20b",
            verifier_base_url="http://127.0.0.1:11434/v1",
            writer_timeout_seconds=600,
            verifier_timeout_seconds=300,
            disable_writer_fallback=True,
        )

        assert os.environ["SCIBOWL_WRITER_MODEL"] == "qwen3.5:9b"
        assert os.environ["SCIBOWL_VERIFIER_MODEL"] == "gpt-oss:20b"
        assert os.environ["SCIBOWL_WRITER_TIMEOUT_SECONDS"] == "600"
        assert os.environ["SCIBOWL_VERIFIER_TIMEOUT_SECONDS"] == "300"
        assert os.environ["SCIBOWL_DISABLE_WRITER_FALLBACK"] == "1"
    finally:
        _restore_env(previous)


_MODEL_ENV_NAMES = (
    'SCIBOWL_WRITER_TIMEOUT_SECONDS',
    'SCIBOWL_VERIFIER_TIMEOUT_SECONDS',
    'SCIBOWL_DISABLE_WRITER_FALLBACK',
    "SCIBOWL_WRITER_PROVIDER",
    "SCIBOWL_WRITER_MODEL",
    "SCIBOWL_WRITER_BASE_URL",
    "SCIBOWL_WRITER_API_KEY",
    "SCIBOWL_VERIFIER_PROVIDER",
    "SCIBOWL_VERIFIER_MODEL",
    "SCIBOWL_VERIFIER_BASE_URL",
    "SCIBOWL_VERIFIER_API_KEY",
)


def _restore_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
