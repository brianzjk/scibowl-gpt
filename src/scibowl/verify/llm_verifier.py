from __future__ import annotations

import os

from scibowl.llm.client import OpenAICompatibleChatClient
from scibowl.prompts.renderer import load_template, render_verifier_prompt
from scibowl.schema.generation import GeneratedDraft, QuestionSpec, RetrievalBundle


class PromptVerifierModel:
    def __init__(self, client: OpenAICompatibleChatClient, model_name: str, prompt_version: str = "verifier_v1") -> None:
        self.client = client
        self.model_name = model_name
        self.prompt_version = prompt_version

    def review(self, spec: QuestionSpec, draft: GeneratedDraft, bundle: RetrievalBundle) -> dict[str, object]:
        return self.client.complete_json(
            system_prompt=load_template("verifier_system.txt"),
            user_prompt=render_verifier_prompt(spec, draft, bundle),
            temperature=float(os.getenv("SCIBOWL_VERIFIER_TEMPERATURE", "0.0")),
        )


def build_verifier_model() -> PromptVerifierModel | None:
    client = OpenAICompatibleChatClient.from_env(
        provider_env="SCIBOWL_VERIFIER_PROVIDER",
        model_env="SCIBOWL_VERIFIER_MODEL",
        base_url_env="SCIBOWL_VERIFIER_BASE_URL",
        api_key_env="SCIBOWL_VERIFIER_API_KEY",
    )
    if client is None:
        return None
    return PromptVerifierModel(client=client, model_name=client.config.model_name)
