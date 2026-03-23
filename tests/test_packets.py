from scibowl.ingest.packets import parse_packet_text


def test_parse_packet_text_skips_visual_bonus() -> None:
    text = """
TOSS-UP
1) BIOLOGY Short Answer What organelle contains chlorophyll?
ANSWER: Chloroplast

Visual Bonus
2) PHYSICS Short Answer Identify the graph shown.
ANSWER: Harmonic oscillator

BONUS
3) CHEMISTRY Multiple Choice Which gas is noble? W) Oxygen X) Argon Y) Nitrogen Z) Hydrogen
ANSWER: X) Argon
"""

    rows = parse_packet_text(text, source_id="packet_test")

    assert len(rows) == 2
    assert all("visual" not in row.question_text.lower() for row in rows)


def test_parse_packet_text_handles_inline_esbot_style() -> None:
    text = """
Tossup1) Math - Multiple Choice: Which of the following is not possible?
W) One X) Two Y) Three Z) Four
ANSWER: Z) Four
Bonus1) Physics - Short Answer: What quantity is the area under a force-distance graph?
ANSWER: Work
"""

    rows = parse_packet_text(text, source_id="packet_test")

    assert len(rows) == 2
    assert rows[0].category.value == "math"
    assert rows[0].question_text == "Which of the following is not possible?"
    assert [choice.model_dump() for choice in rows[0].choices] == [
        {"label": "W", "text": "One"},
        {"label": "X", "text": "Two"},
        {"label": "Y", "text": "Three"},
        {"label": "Z", "text": "Four"},
    ]
    assert rows[1].answer_text == "ANSWER: Work"


def test_parse_packet_text_handles_collapsed_format_tokens() -> None:
    text = """
TOSS-UP
1. ChemistryMultipleChoice Which indicator changes color near an endpoint?
W) Buffer X) Indicator Y) Acid Z) Base
ANSWER: X) Indicator
"""

    rows = parse_packet_text(text, source_id="packet_test")

    assert len(rows) == 1
    assert rows[0].answer_mode.value == "multiple_choice"


def test_parse_packet_text_cleans_answer_line_artifacts() -> None:
    text = """
Tossup 1) Biology - Short Answer What organelle contains chlorophyll?
ANSWER: Chloroplast [RG] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bonus 2) Chemistry - Multiple Choice Which gas is noble? W) Oxygen X) Argon Y) Nitrogen Z) Hydrogen
ANSWER: X) Argon MIT Science Bowl Invitational Round 1 Page 2
"""

    rows = parse_packet_text(text, source_id="packet_test")

    assert len(rows) == 2
    assert rows[0].answer_text == "ANSWER: Chloroplast"
    assert rows[1].answer_text == "ANSWER: X) Argon"


def test_parse_packet_text_extracts_multiline_choices() -> None:
    text = """
BONUS
1) CHEMISTRY Multiple Choice Which gas is noble?

W) Oxygen
X) Argon
Y) Nitrogen
Z) Hydrogen

ANSWER: X
"""

    rows = parse_packet_text(text, source_id="packet_test")

    assert len(rows) == 1
    assert rows[0].question_text == "Which gas is noble?"
    assert [choice.model_dump() for choice in rows[0].choices] == [
        {"label": "W", "text": "Oxygen"},
        {"label": "X", "text": "Argon"},
        {"label": "Y", "text": "Nitrogen"},
        {"label": "Z", "text": "Hydrogen"},
    ]
    assert rows[0].answer_text == "ANSWER: X) Argon"


def test_parse_packet_text_ignores_x_axis_in_mc_stem() -> None:
    text = """
TOSS-UP
1) Physics Multiple Choice A wire lies on the x-axis. Which of the following directions matches the magnetic field at a point above it?
W) Positive x
X) Positive z
Y) Negative z
Z) Positive y
ANSWER: X
"""

    rows = parse_packet_text(text, source_id="packet_test")

    assert len(rows) == 1
    assert rows[0].question_text == "A wire lies on the x-axis. Which of the following directions matches the magnetic field at a point above it?"
    assert [choice.model_dump() for choice in rows[0].choices] == [
        {"label": "W", "text": "Positive x"},
        {"label": "X", "text": "Positive z"},
        {"label": "Y", "text": "Negative z"},
        {"label": "Z", "text": "Positive y"},
    ]


def test_parse_packet_text_ignores_terminal_z_in_mc_stem() -> None:
    text = """
BONUS
1) Physics Multiple Choice A transmission line has impedance Z. Which of the following best describes what happens at a discontinuity?
W) It stays unchanged
X) Some energy is reflected
Y) It disappears
Z) It doubles in speed
ANSWER: X
"""

    rows = parse_packet_text(text, source_id="packet_test")

    assert len(rows) == 1
    assert rows[0].question_text == "A transmission line has impedance Z. Which of the following best describes what happens at a discontinuity?"
    assert [choice.model_dump() for choice in rows[0].choices] == [
        {"label": "W", "text": "It stays unchanged"},
        {"label": "X", "text": "Some energy is reflected"},
        {"label": "Y", "text": "It disappears"},
        {"label": "Z", "text": "It doubles in speed"},
    ]
    assert rows[0].answer_text == "ANSWER: X) Some energy is reflected"


def test_parse_packet_text_downgrades_broken_mc_to_short_answer() -> None:
    text = """
BONUS
1) Physics Multiple Choice How many seconds are in two minutes?
ANSWER: 120
"""

    rows = parse_packet_text(text, source_id="packet_test")

    assert len(rows) == 1
    assert rows[0].answer_mode.value == "short_answer"
    assert rows[0].choices == []


def test_parse_packet_text_handles_writer_name_between_type_and_number() -> None:
    text = """
Bonus
18) Earth and Space - Short Answer: Although Venus has a very low orbital tilt, it shows periodic variations in atmospheric absorption due to variations in its energy budget. To the nearest year, what is the measured period of these atmospheric oscillations?
ANSWER: 11 years

Tossup - Colin
19) Chemistry - Short Answer: What quantity, defined as half the product of the concentration and the square of the charge of an ion, is commonly used to define activity coefficients for strong electrolytes in solution?
ANSWER: Ionic strength
"""

    rows = parse_packet_text(text, source_id="packet_test")

    assert len(rows) == 2
    assert rows[0].answer_text == "ANSWER: 11 years"
    assert rows[1].question_text.startswith("What quantity, defined as half the product")
    assert rows[1].answer_text == "ANSWER: Ionic strength"
