import shutil
from pathlib import Path

from scibowl.ingest.textbooks import (
    _chunk_selected_pages,
    _clean_textbook_page,
    _extract_page_headings,
    _select_content_pages,
    _trim_pages_before_back_matter,
    ingest_textbook_text,
)
from scibowl.utils.ids import make_id
from scibowl.utils.text import chunk_paragraphs


def _make_temp_dir() -> Path:
    path = Path('tests_runtime') / make_id('chunk_case')
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_back_matter_detection_does_not_match_body_phrases() -> None:
    pages = [
        'Chapter 8\nA color index compares brightness measured through two filters.',
        'Chapter 9\nThe appendix of a fish anchors muscles to the body wall.',
        'Chapter 10\nThis section references the result from the prior chapter.',
    ]

    trimmed_pages, metadata = _trim_pages_before_back_matter(pages)

    assert trimmed_pages == pages
    assert metadata['back_matter_trimmed'] is False


def test_manual_page_bounds_keep_pdf_page_numbers() -> None:
    pages = ['cover', 'contents', 'chapter one', 'chapter two', 'index']

    selected, first_page, metadata = _select_content_pages(
        pages,
        start_page=3,
        end_page=4,
    )

    assert selected == ['chapter one', 'chapter two']
    assert first_page == 3
    assert metadata['content_start_page'] == 3
    assert metadata['content_end_page'] == 4


def test_ingest_textbook_has_bounded_stable_chunks_and_run_provenance() -> None:
    tmp_path = _make_temp_dir()
    text_path = tmp_path / 'book.txt'
    text_path.write_text(' '.join(f'word{index}' for index in range(420)), encoding='utf-8')

    first = ingest_textbook_text(
        text_path,
        document_id='test_book',
        max_words=180,
        overlap_words=30,
        ingest_run_id='corpus_fixed',
    )
    second = ingest_textbook_text(
        text_path,
        document_id='test_book',
        max_words=180,
        overlap_words=30,
    )

    assert len(first) == 3
    assert all(len(chunk.text.split()) <= 180 for chunk in first)
    assert all(chunk.pages == [1] for chunk in first)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.metadata['ingest_run_id'] for chunk in first}) == 1
    assert {chunk.metadata['ingest_run_id'] for chunk in first} == {'corpus_fixed'}
    assert first[0].metadata['source_sha256'] == second[0].metadata['source_sha256']
    shutil.rmtree(tmp_path)


def test_chunk_paragraphs_splits_long_blocks_with_word_overlap() -> None:
    words = [f'w{index}' for index in range(35)]

    chunks = chunk_paragraphs([' '.join(words)], max_words=15, overlap_words=5)

    assert [len(chunk.split()) for chunk in chunks] == [15, 15, 15]
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]
    assert chunks[1].split()[-5:] == chunks[2].split()[:5]


def test_page_chunks_overlap_across_pages_and_keep_page_spans() -> None:
    first_page = ' '.join(f'a{index}' for index in range(80))
    second_page = ' '.join(f'b{index}' for index in range(80))

    chunks = _chunk_selected_pages(
        [first_page, second_page],
        first_page=7,
        max_words=100,
        overlap_words=20,
    )

    assert [len(chunk.text.split()) for chunk in chunks] == [100, 80]
    assert chunks[0].pages == [7, 8]
    assert chunks[1].pages == [8]
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_cleaner_drops_marker_not_rest_of_page() -> None:
    cleaned = _clean_textbook_page(
        'Useful fact before.\nProblem 3.1 Calculate a value.\nUseful fact after.'
    )

    assert 'Problem 3.1' not in cleaned
    assert 'Useful fact before.' in cleaned
    assert 'Useful fact after.' in cleaned


def test_cleaner_keeps_summary_and_numbered_science_statements() -> None:
    cleaned = _clean_textbook_page(
        'Summary\n1. The first law conserves energy.\nThe chapter closes here.'
    )

    assert 'Summary' in cleaned
    assert '1. The first law conserves energy.' in cleaned


def test_cleaner_repairs_common_pdf_mojibake() -> None:
    cleaned = _clean_textbook_page('The atmosphere inï¬‚uences Earthâ€”and lifeâ€™s chemistry.')

    assert cleaned == 'The atmosphere influences Earth—and life’s chemistry.'


def test_cleaner_drops_page_with_selected_answers_marker() -> None:
    cleaned = _clean_textbook_page(
        'A summary fact about ATP.\nCONCEPT 5.6 TEST YOUR UNDERSTANDING\n'
        '1. Which process makes ATP?\nFor selected answers, see Appendix A.'
    )

    assert cleaned == ''


def test_cleaner_drops_dense_review_continuation_page() -> None:
    cleaned = _clean_textbook_page(
        'Chapter 4\n3. Why does the field change?\n4. Explain the observed current.\n'
        '5. Which direction does it point?\n6. What happens next?'
    )

    assert cleaned == ''


def test_cleaner_drops_embedded_visual_prompt() -> None:
    cleaned = _clean_textbook_page(
        'Chemiosmosis uses a proton gradient.\nVISUAL SKILLS Look at Figure 3.\n'
        'Explain how you would alter it.\nUse arrows in your answer.\nATP synthase makes ATP.'
    )

    assert 'VISUAL SKILLS' not in cleaned
    assert 'Explain how' not in cleaned
    assert 'ATP synthase makes ATP.' in cleaned


def test_manual_page_bounds_reject_out_of_range_values() -> None:
    pages = ['one', 'two']

    for start_page, end_page in ((0, None), (3, None), (None, 0), (None, 3)):
        try:
            _select_content_pages(pages, start_page=start_page, end_page=end_page)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid page bounds must raise ValueError')


def test_chunk_paragraphs_ignores_blank_values() -> None:
    assert chunk_paragraphs(['', '   ']) == []


def test_heading_detection_is_conservative_and_canonical() -> None:
    chapter, section = _extract_page_headings(
        '6 CHAPTER 1 THE SCIENCE OF LIFE\n1.1 Asking Questions\nBody text follows.',
        current_chapter=None,
        current_section=None,
    )

    assert chapter == 'Chapter 1'
    assert section == '1.1 Asking Questions'

    repeated_chapter, preserved_section = _extract_page_headings(
        'CHAPTER 1 10\n0.005 volts is a small value.\n13.7 billion years is an age.',
        current_chapter=chapter,
        current_section=section,
    )

    assert repeated_chapter == 'Chapter 1'
    assert preserved_section == '1.1 Asking Questions'

    prose_chapter, prose_section = _extract_page_headings(
        'Chapter 10 describes the next result.\n4.6 billion years ago Earth formed.',
        current_chapter=chapter,
        current_section=section,
    )

    assert prose_chapter == 'Chapter 1'
    assert prose_section == '1.1 Asking Questions'


def test_ingest_rejects_source_mismatch() -> None:
    tmp_path = _make_temp_dir()
    text_path = tmp_path / 'book.txt'
    text_path.write_text('A short source.', encoding='utf-8')

    try:
        ingest_textbook_text(text_path, expected_page_count=2)
    except ValueError as exc:
        assert 'page count mismatch' in str(exc)
    else:
        raise AssertionError('wrong page count must raise ValueError')

    try:
        ingest_textbook_text(text_path, expected_sha256='0' * 64)
    except ValueError as exc:
        assert 'SHA-256 mismatch' in str(exc)
    else:
        raise AssertionError('wrong source hash must raise ValueError')
    shutil.rmtree(tmp_path)
