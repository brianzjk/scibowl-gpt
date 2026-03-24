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

EARTH_SPACE_ASTRO_SUBCATEGORIES: frozenset[str] = frozenset(
    {
        "Cosmology",
        "Observation",
        "Solar System",
        "Stars",
    }
)

EARTH_SPACE_EARTH_SUBCATEGORIES: frozenset[str] = frozenset(
    {
        "Hydrology",
        "Meteorology",
        "Rocks and Minerals",
        "Tectonics",
    }
)

SUBCATEGORY_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "Cosmology": ("Cosmo",),
    "Hydrology": ("Hydro",),
    "Meteorology": ("Meteo",),
    "Rocks and Minerals": ("R&M", "rocks", "minerals"),
    "Combinatorics": ("Combo",),
    "Geometry": ("Geo",),
    "Number Theory": ("NT",),
}

SUBCATEGORY_TOPIC_HINTS: dict[str, tuple[str, ...]] = {
    "Cosmology": (
        "cosmic microwave background",
        "expansion of the universe",
        "dark matter",
        "dark energy",
        "inflation",
        "hubble law",
    ),
    "Hydrology": (
        "water cycle",
        "groundwater",
        "aquifer",
        "water table",
        "runoff",
        "infiltration",
        "watershed",
        "streamflow",
        "porosity",
        "permeability",
    ),
    "Meteorology": (
        "atmosphere",
        "air mass",
        "front",
        "pressure",
        "humidity",
        "clouds",
        "storms",
        "precipitation",
        "jet stream",
        "weather",
        "climate",
    ),
    "Observation": (
        "spectra",
        "telescopes",
        "light curve",
        "parallax",
        "redshift",
        "detectors",
    ),
    "Rocks and Minerals": (
        "igneous",
        "sedimentary",
        "metamorphic",
        "hardness",
        "cleavage",
        "luster",
        "mineral properties",
    ),
    "Solar System": (
        "planets",
        "moons",
        "asteroids",
        "comets",
        "orbits",
        "planetary atmospheres",
    ),
    "Stars": (
        "stellar evolution",
        "stellar spectra",
        "luminosity",
        "fusion",
        "main sequence",
        "red giant",
    ),
    "Tectonics": (
        "plate boundaries",
        "subduction",
        "rift",
        "transform fault",
        "earthquake",
        "volcanism",
        "seafloor spreading",
    ),
}

_CANONICAL_LOOKUP = {key.casefold(): key for key in SUBCATEGORY_QUERY_ALIASES}
_ALIAS_LOOKUP = {
    alias.casefold(): canonical
    for canonical, aliases in SUBCATEGORY_QUERY_ALIASES.items()
    for alias in aliases
}


def canonicalize_subcategory(value: str) -> str:
    clean_value = value.strip()
    if not clean_value:
        return ""
    return _CANONICAL_LOOKUP.get(clean_value.casefold()) or _ALIAS_LOOKUP.get(clean_value.casefold()) or clean_value


def expand_subcategory_phrases(value: str) -> list[str]:
    clean_value = value.strip()
    if not clean_value:
        return []

    phrases = {clean_value}
    canonical = canonicalize_subcategory(clean_value)
    if canonical is not None:
        phrases.add(canonical)
        phrases.update(SUBCATEGORY_QUERY_ALIASES.get(canonical, ()))
        phrases.update(SUBCATEGORY_TOPIC_HINTS.get(canonical, ()))

    return sorted(phrases)


def subcategory_guidance_terms(value: str) -> list[str]:
    canonical = canonicalize_subcategory(value)
    if not canonical:
        return []
    return list(SUBCATEGORY_TOPIC_HINTS.get(canonical, ()))
