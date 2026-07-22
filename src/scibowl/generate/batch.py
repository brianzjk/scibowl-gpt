from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from scibowl.eval.benchmark import build_evaluation_record
from scibowl.generate.orchestration import GenerationOrchestrator
from scibowl.schema.common import Category, QuestionType
from scibowl.schema.dataset import GeneratedQuestionRunRecord
from scibowl.schema.generation import QuestionSpec
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.ids import make_id, slugify
from scibowl.utils.io import ensure_parent, read_jsonl
from scibowl.utils.subcategories import DEFAULT_RANDOM_SUBCATEGORY_POOLS

DEFAULT_RANDOM_QUESTION_TYPES: tuple[QuestionType, ...] = (
    QuestionType.TOSSUP,
    QuestionType.BONUS,
)
DEFAULT_RANDOM_DIFFICULTIES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)


class GenerationJobConfig(BaseModel):
    job_id: str | None = None
    category: Category
    subcategory: str | None = None
    subcategory_mode: Literal["fixed", "random"] = "fixed"
    subcategory_pool: list[str] = Field(default_factory=list)
    question_type: QuestionType | None = QuestionType.TOSSUP
    question_type_mode: Literal["fixed", "random"] = "fixed"
    question_type_pool: list[QuestionType] = Field(default_factory=list)
    difficulty: int | None = None
    difficulty_mode: Literal["fixed", "random"] = "fixed"
    difficulty_pool: list[int] = Field(default_factory=list)
    count: int = 1
    topic_focus: list[str] = Field(default_factory=list)
    must_use_sources: list[str] = Field(default_factory=list)
    forbidden_topics: list[str] = Field(default_factory=list)

    @field_validator("category")
    @classmethod
    def reject_energy(cls, value: Category) -> Category:
        if value == Category.ENERGY:
            raise ValueError("energy generation is disabled; use human writing or extra tooling instead")
        return value

    @model_validator(mode="after")
    def validate_subcategory_settings(self) -> "GenerationJobConfig":
        self.subcategory = self.subcategory.strip() if self.subcategory else None
        self.subcategory_pool = [item.strip() for item in self.subcategory_pool if item.strip()]
        self.difficulty_pool = [int(item) for item in self.difficulty_pool]
        if self.subcategory_mode == "fixed" and not (self.subcategory and self.subcategory.strip()):
            raise ValueError("subcategory is required unless subcategory_mode is 'random'")
        if self.subcategory_mode == "random" and not self.subcategory_pool:
            default_pool = DEFAULT_RANDOM_SUBCATEGORY_POOLS.get(self.category.value)
            if not default_pool:
                raise ValueError(
                    f"no default random subcategory pool is configured for category '{self.category.value}'"
                )
        if self.question_type_mode == "fixed" and self.question_type is None:
            raise ValueError("question_type is required unless question_type_mode is 'random'")
        if self.difficulty_mode == "fixed" and self.difficulty is None:
            raise ValueError("difficulty is required unless difficulty_mode is 'random'")
        if self.difficulty is not None and not 1 <= self.difficulty <= 7:
            raise ValueError("difficulty must be between 1 and 7")
        if self.difficulty_pool and any(not 1 <= item <= 7 for item in self.difficulty_pool):
            raise ValueError("difficulty_pool values must all be between 1 and 7")
        return self


class GenerationBatchConfig(BaseModel):
    output_path: Path
    style_questions_path: Path
    textbook_chunks_path: Path
    random_seed: int | None = None
    jobs: list[GenerationJobConfig]


def load_generation_batch_config(path: Path) -> GenerationBatchConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return GenerationBatchConfig.model_validate(payload)


def run_generation_batch(config_path: Path) -> list[GeneratedQuestionRunRecord]:
    config = load_generation_batch_config(config_path)
    style_questions = read_jsonl(config.style_questions_path, NormalizedQuestion)
    textbook_chunks = _load_textbook_chunks(config.textbook_chunks_path)
    orchestrator = GenerationOrchestrator()
    rng = random.Random(config.random_seed)
    recent_fact_chunk_ids: deque[str] = deque(maxlen=18)
    recent_style_question_ids: deque[str] = deque(maxlen=18)

    records: list[GeneratedQuestionRunRecord] = []
    total_requested = sum(job.count for job in config.jobs)
    completed = 0
    ensure_parent(config.output_path)
    config.output_path.write_text("", encoding="utf-8")

    for job in config.jobs:
        job_id = job.job_id or _default_job_id(job)
        for index in range(job.count):
            selected_subcategory = _select_subcategory(job, rng)
            selected_question_type = _select_question_type(job, rng)
            selected_difficulty = _select_difficulty(job, rng)
            spec = QuestionSpec(
                spec_id=make_id("spec"),
                category=job.category,
                subcategory=selected_subcategory,
                question_type=selected_question_type,
                answer_mode=None,
                difficulty=selected_difficulty,
                topic_focus=job.topic_focus or [selected_subcategory],
                must_use_sources=job.must_use_sources,
                forbidden_topics=job.forbidden_topics,
                style_target_ids=[],
            )
            bundle, draft, report = orchestrator.run(
                spec,
                textbook_chunks,
                style_questions,
                avoid_fact_chunk_ids=set(recent_fact_chunk_ids),
                avoid_style_question_ids=set(recent_style_question_ids),
            )
            evaluation = build_evaluation_record("generation_batch", draft.draft_id, spec, draft, report)
            record = GeneratedQuestionRunRecord(
                run_id="generation_batch",
                job_id=job_id,
                spec=spec,
                bundle=bundle,
                draft=draft,
                report=report,
                evaluation=evaluation,
                metadata={
                    "job_id": job_id,
                    "job_index": index + 1,
                    "requested_count": job.count,
                    "subcategory_mode": job.subcategory_mode,
                    "selected_subcategory": selected_subcategory,
                    "question_type_mode": job.question_type_mode,
                    "selected_question_type": selected_question_type.value,
                    "answer_mode_mode": "writer_selected",
                    "selected_answer_mode": draft.question.answer_mode.value,
                    "difficulty_mode": job.difficulty_mode,
                    "selected_difficulty": selected_difficulty,
                },
            )
            records.append(record)
            with config.output_path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
            recent_fact_chunk_ids.extend(chunk.chunk_id for chunk in bundle.fact_chunks)
            recent_style_question_ids.extend(example.question_id for example in bundle.style_examples)
            completed += 1
            print(
                f"[{completed}/{total_requested}] "
                f"{job_id} #{index + 1}: "
                f"{selected_subcategory} / {selected_question_type.value} / "
                f"{draft.question.answer_mode.value} / difficulty {selected_difficulty}"
            )
    return records


def _load_textbook_chunks(path: Path) -> list[TextbookChunk]:
    if path.is_dir():
        chunks: list[TextbookChunk] = []
        for jsonl_path in sorted(path.glob("*.jsonl")):
            chunks.extend(read_jsonl(jsonl_path, TextbookChunk))
        return chunks
    return read_jsonl(path, TextbookChunk)


def _default_job_id(job: GenerationJobConfig) -> str:
    subcategory_label = job.subcategory if job.subcategory_mode == "fixed" else "random_subcategory"
    question_type_label = (
        job.question_type.value if job.question_type_mode == "fixed" and job.question_type is not None else "random_question_type"
    )
    difficulty_label = str(job.difficulty) if job.difficulty_mode == "fixed" and job.difficulty is not None else "random_difficulty"
    return slugify(
        f"{job.category.value}_{subcategory_label}_{question_type_label}_writer_selected_{difficulty_label}"
    )


def _select_subcategory(job: GenerationJobConfig, rng: random.Random) -> str:
    if job.subcategory_mode == "fixed":
        return str(job.subcategory).strip()
    pool = job.subcategory_pool or list(DEFAULT_RANDOM_SUBCATEGORY_POOLS[job.category.value])
    return rng.choice(pool)


def _select_question_type(job: GenerationJobConfig, rng: random.Random) -> QuestionType:
    if job.question_type_mode == "fixed":
        assert job.question_type is not None
        return job.question_type
    pool = job.question_type_pool or list(DEFAULT_RANDOM_QUESTION_TYPES)
    return rng.choice(pool)


def _select_difficulty(job: GenerationJobConfig, rng: random.Random) -> int:
    if job.difficulty_mode == "fixed":
        assert job.difficulty is not None
        return job.difficulty
    pool = job.difficulty_pool or list(DEFAULT_RANDOM_DIFFICULTIES)
    return rng.choice(pool)
