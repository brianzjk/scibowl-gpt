from scibowl.retrieval.search import _year_from_text, retrieve_bundle
from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.generation import QuestionSpec
from scibowl.schema.question import NormalizedQuestion, SourceMetadata
from scibowl.schema.textbook import TextbookChunk


def _spec(
    spec_id: str,
    *,
    category: Category = Category.EARTH_SPACE,
    subcategory: str = 'Hydrology',
    topic: str = 'groundwater aquifer',
    style_target_ids: list[str] | None = None,
) -> QuestionSpec:
    return QuestionSpec(
        spec_id=spec_id,
        category=category,
        subcategory=subcategory,
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=[topic],
        style_target_ids=style_target_ids or [],
    )


def _chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    topics: list[str] | None = None,
    pages: list[int] | None = None,
) -> TextbookChunk:
    return TextbookChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title='Test Textbook',
        chapter='Chapter 3 Water',
        section='3.2 Aquifers',
        pages=pages or [],
        topics=topics or [],
        text=text,
        char_count=len(text),
        token_count_est=len(text.split()),
        metadata={'corpus_version': 'textbook_v2'},
    )


def _style(
    question_id: str,
    source_id: str,
    text: str,
    *,
    year: int | None = None,
    quality: float | None = None,
    difficulty: int = 4,
) -> NormalizedQuestion:
    return NormalizedQuestion(
        question_id=question_id,
        source_type=SourceType.DATASET,
        source_id=source_id,
        category=Category.EARTH_SPACE,
        subcategory='Hydrology',
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=difficulty,
        question_text=text,
        answer_text='ANSWER: aquifer',
        source_metadata=SourceMetadata(
            year=year,
            original_quality=quality,
            original_difficulty=difficulty,
        ),
    )


def test_avoided_high_score_fact_is_never_returned() -> None:
    chunks = [
        _chunk(
            'avoid_me',
            'garrison_essentials_of_oceanography_5e',
            'Groundwater fills an aquifer below the water table. Groundwater groundwater aquifer.',
        ),
        _chunk(
            'keep_me',
            'tarbuck_earth_science',
            'An aquifer stores groundwater in permeable rock.',
        ),
    ]

    bundle = retrieve_bundle(
        _spec('avoid_test'),
        chunks,
        [],
        fact_top_k=2,
        avoid_fact_chunk_ids={'avoid_me'},
    )

    assert [chunk.chunk_id for chunk in bundle.fact_chunks] == ['keep_me']


def test_relevant_nonpreferred_book_beats_irrelevant_preferred_book() -> None:
    irrelevant_preferred = _chunk(
        'preferred_but_wrong',
        'garrison_essentials_of_oceanography_5e',
        'Ocean salinity changes as seawater freezes.',
        topics=['oceanography'],
    ).model_copy(update={'chapter': None, 'section': None})
    chunks = [
        irrelevant_preferred,
        _chunk(
            'nonpreferred_relevant',
            'tarbuck_earth_science',
            'Groundwater moves through a permeable aquifer below the water table.',
            topics=['groundwater'],
        ),
    ]

    bundle = retrieve_bundle(_spec('soft_route'), chunks, [], fact_top_k=2)

    assert [chunk.chunk_id for chunk in bundle.fact_chunks] == ['nonpreferred_relevant']


def test_zero_match_abstains_and_results_ignore_spec_id() -> None:
    irrelevant = _chunk(
        'classical',
        'halliday_resnick_krane_volume_1',
        'A block slides down an inclined plane with friction.',
        topics=['mechanics'],
    )
    zero_bundle = retrieve_bundle(
        _spec(
            'zero',
            category=Category.PHYSICS,
            subcategory='Quantum Physics',
            topic='quantum entanglement',
        ),
        [irrelevant],
        [],
    )
    assert zero_bundle.fact_chunks == []

    chunks = [
        _chunk('b', 'tarbuck_earth_science', 'An aquifer holds groundwater.'),
        _chunk('a', 'garrison_essentials_of_oceanography_5e', 'Groundwater can enter an aquifer.'),
    ]
    first = retrieve_bundle(_spec('first_id'), chunks, [])
    second = retrieve_bundle(_spec('second_id'), chunks, [])

    assert [chunk.chunk_id for chunk in first.fact_chunks] == [
        chunk.chunk_id for chunk in second.fact_chunks
    ]


def test_fact_search_rejects_one_generic_match_from_multi_term_topic() -> None:
    generic = _chunk(
        'generic_theorem',
        'halliday_resnick_krane_volume_1',
        'This theorem describes the motion of a classical particle.',
        topics=['mechanics'],
    )

    bundle = retrieve_bundle(
        _spec(
            'weak_match',
            category=Category.PHYSICS,
            subcategory='Banana Studies',
            topic='zqxj purple banana theorem',
        ),
        [generic],
        [],
    )

    assert bundle.fact_chunks == []


def test_duplicate_chunk_ids_are_removed_and_locator_uses_page_metadata() -> None:
    first = _chunk(
        'same_id',
        'tarbuck_earth_science',
        'Groundwater fills pore spaces in an aquifer.',
        pages=[42, 43],
    )
    duplicate = first.model_copy(update={'text': 'Groundwater in a duplicate record.'})

    bundle = retrieve_bundle(_spec('dedupe'), [duplicate, first], [], fact_top_k=3)

    assert len(bundle.fact_chunks) == 1
    assert bundle.fact_chunks[0].locator == (
        'Test Textbook; Chapter 3 Water; 3.2 Aquifers; pp. 42-43'
    )


def test_explicit_style_target_wins_and_recent_mit_is_default_preference() -> None:
    old = _style(
        'old_target',
        'nsb_set_17',
        'What underground layer stores water?',
        year=2018,
        quality=-1,
        difficulty=7,
    )
    recent = _style(
        'recent_mit',
        'mit_2025',
        'What term names the upper surface of groundwater?',
        year=2025,
        quality=1,
        difficulty=4,
    )

    default_bundle = retrieve_bundle(_spec('style_default'), [], [old, recent], style_top_k=1)
    target_bundle = retrieve_bundle(
        _spec('style_target', style_target_ids=['old_target']),
        [],
        [old, recent],
        style_top_k=1,
    )

    assert default_bundle.style_examples[0].question_id == 'recent_mit'
    assert target_bundle.style_examples[0].question_id == 'old_target'


def test_style_year_parser_accepts_underscored_source_ids() -> None:
    assert _year_from_text('mit_2024') == 2024
