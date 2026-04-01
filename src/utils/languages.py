"""
Language enumeration and metadata for multilingual analysis.

This module defines the canonical set of 13 languages used throughout
the Tiny Aya Under The Hood project. Each language carries structured
metadata — ISO 639-1 code, FLORES-200 dataset code, writing script,
language family, and resource level — enabling consistent cross-lingual
analysis across all modules and notebooks.

The module also provides pre-computed convenience groupings:
    - ``LANGUAGE_FAMILIES``: languages grouped by genetic family
    - ``SCRIPT_GROUPS``: languages grouped by writing system
    - ``RESOURCE_GROUPS``: languages grouped by data availability

Usage::

    from src.utils.languages import Language, LANGUAGE_FAMILIES

    # Access individual language metadata
    hindi = Language.HINDI
    print(hindi.flores_code)   # "hin_Deva"
    print(hindi.family)        # "Indo-European"

    # Iterate over all Indo-European languages
    for lang in LANGUAGE_FAMILIES["Indo-European"]:
        print(lang.lang_name)

References:
    - FLORES-200: https://github.com/facebookresearch/flores
    - Tiny Aya: https://huggingface.co/collections/CohereLabs/tiny-aya
"""


from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Language metadata container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LanguageInfo:
    """Immutable metadata container for a single language.

    This dataclass stores all the static information needed to work
    with a language across the project's analysis pipelines: data
    loading (via ``flores_code``), linguistic grouping (via ``script``
    and ``family``), and resource-aware analysis (via ``resource_level``).

    Attributes:
        name: Human-readable lowercase language name (e.g., "hindi").
            Used as the canonical key in dictionaries and file names.
        iso_code: ISO 639-1 two-letter code (e.g., "hi").
            Used for compact display in plots and tables.
        flores_code: FLORES-200 language code (e.g., "hin_Deva").
            Used to load the correct column from the FLORES dataset.
        script: Writing system name (e.g., "Devanagari", "Latin").
            Used for script-based CKA decomposition analysis.
        family: Genetic language family (e.g., "Indo-European").
            Used for language family clustering analysis.
        resource_level: Data availability tier — one of "high",
            "mid", or "low". Used for equity-aware analysis.
    """

    name: str
    iso_code: str
    flores_code: str
    script: str
    family: str
    resource_level: str


# ---------------------------------------------------------------------------
# Language enumeration
# ---------------------------------------------------------------------------

class Language(Enum):
    """Enumeration of the 13 supported languages for analysis.

    Each member's value is a ``LanguageInfo`` instance containing all
    metadata. Convenience properties provide direct attribute access
    without going through ``.value``.

    The language set covers:
        - 5 language families: Indo-European, Dravidian, Niger-Congo,
          Afro-Asiatic, Turkic
        - 6 writing scripts: Latin, Devanagari, Bengali, Tamil, Arabic,
          Ge'ez (Ethiopic)
        - 3 resource tiers: high (en, ar, de, fr, es), mid (hi, bn, ta,
          tr, fa), low (sw, am, yo)

    This diversity is deliberately chosen to test whether Tiny Aya's
    cross-lingual alignment is truly semantic (family/script-independent)
    or influenced by surface-level similarities.
    """

    HINDI = LanguageInfo(
        name="hindi",
        iso_code="hi",
        flores_code="hin_Deva",
        script="Devanagari",
        family="Indo-European",
        resource_level="mid",
    )
    ENGLISH = LanguageInfo(
        name="english",
        iso_code="en",
        flores_code="eng_Latn",
        script="Latin",
        family="Indo-European",
        resource_level="high",
    )
    BENGALI = LanguageInfo(
        name="bengali",
        iso_code="bn",
        flores_code="ben_Beng",
        script="Bengali",
        family="Indo-European",
        resource_level="mid",
    )
    TAMIL = LanguageInfo(
        name="tamil",
        iso_code="ta",
        flores_code="tam_Taml",
        script="Tamil",
        family="Dravidian",
        resource_level="mid",
    )
    SWAHILI = LanguageInfo(
        name="swahili",
        iso_code="sw",
        flores_code="swh_Latn",
        script="Latin",
        family="Niger-Congo",
        resource_level="low",
    )
    AMHARIC = LanguageInfo(
        name="amharic",
        iso_code="am",
        flores_code="amh_Ethi",
        script="Ge'ez",
        family="Afro-Asiatic",
        resource_level="low",
    )
    YORUBA = LanguageInfo(
        name="yoruba",
        iso_code="yo",
        flores_code="yor_Latn",
        script="Latin",
        family="Niger-Congo",
        resource_level="low",
    )
    ARABIC = LanguageInfo(
        name="arabic",
        iso_code="ar",
        flores_code="arb_Arab",
        script="Arabic",
        family="Afro-Asiatic",
        resource_level="high",
    )
    TURKISH = LanguageInfo(
        name="turkish",
        iso_code="tr",
        flores_code="tur_Latn",
        script="Latin",
        family="Turkic",
        resource_level="mid",
    )
    PERSIAN = LanguageInfo(
        name="persian",
        iso_code="fa",
        flores_code="pes_Arab",
        script="Arabic",
        family="Indo-European",
        resource_level="mid",
    )
    GERMAN = LanguageInfo(
        name="german",
        iso_code="de",
        flores_code="deu_Latn",
        script="Latin",
        family="Indo-European",
        resource_level="high",
    )
    FRENCH = LanguageInfo(
        name="french",
        iso_code="fr",
        flores_code="fra_Latn",
        script="Latin",
        family="Indo-European",
        resource_level="high",
    )
    SPANISH = LanguageInfo(
        name="spanish",
        iso_code="es",
        flores_code="spa_Latn",
        script="Latin",
        family="Indo-European",
        resource_level="high",
    )

    # -----------------------------------------------------------------
    # Convenience properties for direct attribute access
    # -----------------------------------------------------------------

    @property
    def info(self) -> LanguageInfo:
        """Return the full ``LanguageInfo`` metadata object."""
        return self.value

    @property
    def lang_name(self) -> str:
        """Return the lowercase language name (e.g., 'hindi')."""
        return self.value.name

    @property
    def iso_code(self) -> str:
        """Return the ISO 639-1 two-letter code (e.g., 'hi')."""
        return self.value.iso_code

    @property
    def flores_code(self) -> str:
        """Return the FLORES-200 language code (e.g., 'hin_Deva')."""
        return self.value.flores_code

    @property
    def script(self) -> str:
        """Return the writing script name (e.g., 'Devanagari')."""
        return self.value.script

    @property
    def family(self) -> str:
        """Return the language family name (e.g., 'Indo-European')."""
        return self.value.family

    @property
    def resource_level(self) -> str:
        """Return the resource availability tier ('high'/'mid'/'low')."""
        return self.value.resource_level


# ---------------------------------------------------------------------------
# Pre-computed convenience groupings
# ---------------------------------------------------------------------------

#: Languages grouped by genetic family.
#: Keys: "Indo-European", "Dravidian", "Niger-Congo", "Afro-Asiatic", "Turkic"
LANGUAGE_FAMILIES: dict[str, list[Language]] = {}

#: Languages grouped by writing script.
#: Keys: "Latin", "Devanagari", "Bengali", "Tamil", "Arabic", "Ge'ez"
SCRIPT_GROUPS: dict[str, list[Language]] = {}

#: Languages grouped by data resource availability.
#: Keys: "high", "mid", "low"
RESOURCE_GROUPS: dict[str, list[Language]] = {}

# Build the groupings by iterating over all enum members once.
for _lang in Language:
    LANGUAGE_FAMILIES.setdefault(_lang.family, []).append(_lang)
    SCRIPT_GROUPS.setdefault(_lang.script, []).append(_lang)
    RESOURCE_GROUPS.setdefault(_lang.resource_level, []).append(_lang)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_all_flores_codes() -> dict[str, str]:
    """Return a mapping of language name to FLORES-200 dataset code.

    Returns:
        Dictionary with lowercase language names as keys and FLORES-200
        codes as values. Example: ``{"hindi": "hin_Deva", ...}``.
    """
    return {lang.lang_name: lang.flores_code for lang in Language}


def get_language_by_iso(iso_code: str) -> Language | None:
    """Look up a ``Language`` enum member by its ISO 639-1 code.

    Args:
        iso_code: Two-letter ISO 639-1 code (e.g., "hi", "en").

    Returns:
        The matching ``Language`` member, or ``None`` if no match is found.

    Example::

        >>> get_language_by_iso("hi")
        <Language.HINDI: LanguageInfo(name='hindi', ...)>
        >>> get_language_by_iso("xx") is None
        True
    """
    for lang in Language:
        if lang.iso_code == iso_code:
            return lang
    return None


def get_language_by_name(name: str) -> Language | None:
    """Look up a ``Language`` enum member by its lowercase name.

    Args:
        name: Lowercase language name (e.g., "hindi", "english").

    Returns:
        The matching ``Language`` member, or ``None`` if no match is found.
    """
    name_lower = name.lower()
    for lang in Language:
        if lang.lang_name == name_lower:
            return lang
    return None
