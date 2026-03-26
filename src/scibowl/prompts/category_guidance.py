from __future__ import annotations

from scibowl.schema.common import Category


_CATEGORY_GUIDANCE: dict[Category, tuple[str, ...]] = {
    Category.BIOLOGY: (
        "Favor mechanistic, causal, comparative, and functional reasoning over single-fact recall.",
        "Use clue phrases that steer students toward the answer, such as contrasts, causal links, process comparisons, and condition-result language.",
        "Good clue structures include patterns like 'in contrast to', 'while X does Y, this does Z', 'because X', and other scientifically meaningful comparisons.",
        "Front-load harder mechanistic or structural clues, then move toward more obvious functional or definitional clues later in the question.",
        "State the actual interrogative target early enough that students are not forced to wait until the last few words to know what concept is being requested.",
        "Prefer prompts about pathways, structures, regulation, evolution, physiology, genetics, or experiment-based interpretation rather than isolated vocabulary checks.",
    ),
    Category.CHEMISTRY: (
        "Favor conceptual reasoning about structure, mechanism, equilibrium, energetics, periodic behavior, and experimental consequences over isolated fact recall.",
        "Use clue phrases that set up comparisons, predictions, or causal explanations, especially when contrasting related species, processes, or conditions.",
        "Front-load harder structural or mechanistic clues, then move toward more obvious observable or naming clues later in the question.",
        "State the actual interrogative target early enough that students can identify what property, species, or process they are meant to determine.",
        "Prefer prompts that ask students to infer outcomes, explain behavior, compare cases, or connect molecular structure to macroscopic behavior.",
    ),
    Category.PHYSICS: (
        "Favor conceptual reasoning about physical principles, limiting cases, system behavior, and interpretation of relationships over plug-and-chug recall.",
        "Use clue phrases that compare regimes, identify causes, or connect equations to physical meaning.",
        "Front-load harder principle-based or model-based clues, then move toward more obvious observational or formula-level clues later in the question.",
        "State the actual interrogative target early enough that students can tell which quantity, principle, or phenomenon they are solving for.",
        "Prefer prompts that ask for explanation, prediction, comparison, or qualitative interpretation before raw computation.",
    ),
    Category.EARTH_SPACE: (
        "Favor conceptual reasoning about processes, systems, observations, and cause-effect relationships over isolated catalog facts.",
        "Use clue phrases that compare environments, explain observations, connect mechanisms to outcomes, or distinguish similar Earth or space phenomena.",
        "Front-load harder process-based, observational, or systems-level clues, then move toward more obvious descriptive clues later in the question.",
        "State the actual interrogative target early enough that students can tell whether the question is asking for a process, object, pattern, or interpretation.",
        "Prefer prompts about interpreting data, comparing scenarios, explaining geologic or astronomical behavior, or linking evidence to conclusions.",
    ),
    Category.MATH: (
        "Favor multi-step conceptual setup, recognition of structure, and elegant reasoning over direct formula lookup or routine arithmetic.",
        "Use clue phrases that highlight constraints, contrasts, invariants, or equivalent reformulations.",
        "Front-load harder structural clues, then move toward more obvious computational or special-case clues later in the question.",
        "State the actual interrogative target early enough that students can identify what object, quantity, or property they are meant to determine.",
        "Prefer prompts that reward insight, reformulation, pattern recognition, or theorem application rather than raw speed.",
    ),
    Category.ENERGY: (
        "Favor conceptual reasoning about systems, tradeoffs, modeling assumptions, and applied science over isolated current-events recall.",
        "Use clue phrases that compare scenarios, explain constraints, or connect physical principles to technological behavior.",
        "Front-load harder system-level or mechanism-level clues, then move toward more obvious application clues later in the question.",
        "State the actual interrogative target early enough that students can tell which process, quantity, or technology feature is being asked about.",
        "Prefer prompts that require scientific interpretation or engineering reasoning rather than trivia about organizations, policies, or headlines.",
    ),
}


def category_prompt_guidance(category: Category) -> list[str]:
    return list(_CATEGORY_GUIDANCE.get(category, ()))
