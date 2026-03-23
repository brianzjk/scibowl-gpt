from __future__ import annotations

import os

from scibowl.llm.client import OpenAICompatibleChatClient
from scibowl.prompts.renderer import load_template, render_writer_prompt
from scibowl.schema.common import Citation, ModelInfo
from scibowl.schema.generation import DraftQuestion, GeneratedDraft, QuestionSpec, RetrievalBundle
from scibowl.schema.question import Choice
from scibowl.utils.ids import make_id


class HeuristicWriterModel:
    def __init__(self, model_name: str = "heuristic-writer-v1", prompt_version: str = "writer_v1") -> None:
        self.model_info = ModelInfo(provider="local", model_name=model_name, prompt_version=prompt_version)

    def generate(self, spec: QuestionSpec, bundle: RetrievalBundle) -> GeneratedDraft:
        fact_text = bundle.fact_chunks[0].text if bundle.fact_chunks else f"a {spec.subcategory} concept"
        style_prefix = ""
        if bundle.style_examples:
            first_style = bundle.style_examples[0].question_text.split("?")[0].strip()
            style_prefix = first_style[:80]

        question_text = (
            f"{style_prefix}. "
            f"Using the grounded concept from {spec.subcategory}, answer this Science Bowl question: "
            f"{fact_text[:220].rstrip('.')}."
        ).strip()

        if spec.answer_mode.value == "multiple_choice":
            question_text += " W) option one X) option two Y) option three Z) option four"
            answer_text = "ANSWER: W) option one"
            choices = [
                Choice(label="W", text="option one"),
                Choice(label="X", text="option two"),
                Choice(label="Y", text="option three"),
                Choice(label="Z", text="option four"),
            ]
        else:
            answer_text = f"ANSWER: {spec.topic_focus[0] if spec.topic_focus else spec.subcategory}"
            choices = []

        return build_generated_draft(
            spec=spec,
            bundle=bundle,
            model_info=self.model_info,
            question_text=question_text,
            answer_text=answer_text,
            choices=choices,
        )


class PromptWriterModel:
    def __init__(
        self,
        client: OpenAICompatibleChatClient,
        model_name: str,
        prompt_version: str = "writer_v1",
    ) -> None:
        self.client = client
        self.model_info = ModelInfo(provider="openai_compatible", model_name=model_name, prompt_version=prompt_version)

    def generate(self, spec: QuestionSpec, bundle: RetrievalBundle) -> GeneratedDraft:
        payload = self.client.complete_json(
            system_prompt=load_template("writer_system.txt"),
            user_prompt=render_writer_prompt(spec, bundle),
            temperature=float(os.getenv("SCIBOWL_WRITER_TEMPERATURE", "0.2")),
        )
        question_text = str(payload["question_text"]).strip()
        answer_text = str(payload["answer_text"]).strip()
        choices = [
            Choice(label=str(choice["label"]).strip(), text=str(choice["text"]).strip())
            for choice in payload.get("choices", [])
        ]
        return build_generated_draft(
            spec=spec,
            bundle=bundle,
            model_info=self.model_info,
            question_text=question_text,
            answer_text=answer_text,
            choices=choices,
        )


def build_generated_draft(
    *,
    spec: QuestionSpec,
    bundle: RetrievalBundle,
    model_info: ModelInfo,
    question_text: str,
    answer_text: str,
    choices: list[Choice],
) -> GeneratedDraft:
    citations = [
        Citation(
            source_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            locator=chunk.locator,
        )
        for chunk in bundle.fact_chunks
    ]
    return GeneratedDraft(
        draft_id=make_id("draft"),
        spec_id=spec.spec_id,
        model_info=model_info,
        question=DraftQuestion(
            category=spec.category,
            subcategory=spec.subcategory,
            question_type=spec.question_type,
            answer_mode=spec.answer_mode,
            difficulty=spec.difficulty,
            question_text=question_text,
            answer_text=answer_text,
            choices=choices,
        ),
        citations=citations,
    )


def build_writer_model() -> PromptWriterModel | HeuristicWriterModel:
    client = OpenAICompatibleChatClient.from_env(
        provider_env="SCIBOWL_WRITER_PROVIDER",
        model_env="SCIBOWL_WRITER_MODEL",
        base_url_env="SCIBOWL_WRITER_BASE_URL",
        api_key_env="SCIBOWL_WRITER_API_KEY",
    )
    if client is None:
        return HeuristicWriterModel()
    return PromptWriterModel(client=client, model_name=client.config.model_name)
