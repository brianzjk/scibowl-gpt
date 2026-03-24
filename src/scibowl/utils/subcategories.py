from __future__ import annotations

DEFAULT_RANDOM_SUBCATEGORY_POOLS: dict[str, tuple[str, ...]] = {
    "math": (
        "Algebra",
        "Calculus",
        "Combinatorics",
        "Geometry",
        "Number Theory",
    ),
    "earth_space": (
        "Cosmology",
        "Hydrology",
        "Meteorology",
        "Observation",
        "Rocks and Minerals",
        "Solar System",
        "Stars",
        "Tectonics",
    ),
}

SUBCATEGORY_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "Cosmology": ("Cosmo",),
    "Hydrology": ("Hydro",),
    "Meteorology": ("Meteo",),
    "Rocks and Minerals": ("R&M", "rocks", "minerals"),
    "Combinatorics": ("Combo",),
    "Geometry": ("Geo",),
    "Number Theory": ("NT",),
}


def expand_subcategory_phrases(value: str) -> list[str]:
    clean_value = value.strip()
    if not clean_value:
        return []

    phrases = {clean_value}
    canonical_lookup = {key.casefold(): key for key in SUBCATEGORY_QUERY_ALIASES}
    alias_lookup = {
        alias.casefold(): canonical
        for canonical, aliases in SUBCATEGORY_QUERY_ALIASES.items()
        for alias in aliases
    }

    canonical = canonical_lookup.get(clean_value.casefold()) or alias_lookup.get(clean_value.casefold())
    if canonical is not None:
        phrases.add(canonical)
        phrases.update(SUBCATEGORY_QUERY_ALIASES.get(canonical, ()))

    return sorted(phrases)
