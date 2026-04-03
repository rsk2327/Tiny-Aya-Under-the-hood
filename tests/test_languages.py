"""
Tests for the language registry (src.utils.languages).

Validates that all Language enum members have complete and consistent
metadata, that convenience groupings are correctly built, and that
lookup functions work as expected.
"""

import pytest

from src.utils.languages import (
    LANGUAGE_FAMILIES,
    RESOURCE_GROUPS,
    SCRIPT_GROUPS,
    Language,
    LanguageInfo,
    get_all_flores_codes,
    get_language_by_iso,
    get_language_by_name,
)


class TestLanguageEnum:
    """Tests for the Language enum and its metadata."""

    def test_language_count(self) -> None:
        """Verify we have exactly 13 languages."""
        assert len(Language) == 13

    def test_all_languages_have_complete_metadata(self) -> None:
        """Every Language member must have all LanguageInfo fields filled."""
        for lang in Language:
            info = lang.info
            assert isinstance(info, LanguageInfo)
            assert info.name, f"{lang.name} has empty name"
            assert info.iso_code, f"{lang.name} has empty iso_code"
            assert info.flores_code, f"{lang.name} has empty flores_code"
            assert info.script, f"{lang.name} has empty script"
            assert info.family, f"{lang.name} has empty family"
            assert info.resource_level in ("high", "mid", "low"), (
                f"{lang.name} has invalid resource_level: {info.resource_level}"
            )

    def test_iso_codes_are_unique(self) -> None:
        """No two languages should share an ISO code."""
        iso_codes = [lang.iso_code for lang in Language]
        assert len(iso_codes) == len(set(iso_codes))

    def test_flores_codes_are_unique(self) -> None:
        """No two languages should share a FLORES code."""
        flores_codes = [lang.flores_code for lang in Language]
        assert len(flores_codes) == len(set(flores_codes))

    def test_property_accessors(self) -> None:
        """Convenience properties should return correct values."""
        hindi = Language.HINDI
        assert hindi.lang_name == "hindi"
        assert hindi.iso_code == "hi"
        assert hindi.flores_code == "hin_Deva"
        assert hindi.script == "Devanagari"
        assert hindi.family == "Indo-European"
        assert hindi.resource_level == "mid"

    def test_language_info_is_frozen(self) -> None:
        """LanguageInfo should be immutable (frozen dataclass)."""
        info = Language.ENGLISH.info
        with pytest.raises(AttributeError):
            info.name = "modified"  # type: ignore[misc]


class TestConvenienceGroupings:
    """Tests for LANGUAGE_FAMILIES, SCRIPT_GROUPS, RESOURCE_GROUPS."""

    def test_all_languages_in_families(self) -> None:
        """Every language must appear in exactly one family group."""
        all_langs_in_families = [
            lang for langs in LANGUAGE_FAMILIES.values() for lang in langs
        ]
        assert set(all_langs_in_families) == set(Language)

    def test_all_languages_in_scripts(self) -> None:
        """Every language must appear in exactly one script group."""
        all_langs_in_scripts = [
            lang for langs in SCRIPT_GROUPS.values() for lang in langs
        ]
        assert set(all_langs_in_scripts) == set(Language)

    def test_all_languages_in_resources(self) -> None:
        """Every language must appear in exactly one resource group."""
        all_langs_in_resources = [
            lang for langs in RESOURCE_GROUPS.values() for lang in langs
        ]
        assert set(all_langs_in_resources) == set(Language)

    def test_expected_families(self) -> None:
        """Verify the expected language families are present."""
        expected = {"Indo-European", "Dravidian", "Niger-Congo",
                    "Afro-Asiatic", "Turkic"}
        assert set(LANGUAGE_FAMILIES.keys()) == expected

    def test_expected_scripts(self) -> None:
        """Verify the expected scripts are present."""
        expected = {"Latin", "Devanagari", "Bengali", "Tamil",
                    "Arabic", "Ge'ez"}
        assert set(SCRIPT_GROUPS.keys()) == expected


class TestLookupFunctions:
    """Tests for lookup helper functions."""

    def test_get_all_flores_codes_returns_all(self) -> None:
        """Should return a dict with 13 entries."""
        codes = get_all_flores_codes()
        assert len(codes) == 13
        assert codes["english"] == "eng_Latn"
        assert codes["hindi"] == "hin_Deva"

    def test_get_language_by_iso_found(self) -> None:
        """Should find a language by its ISO code."""
        result = get_language_by_iso("hi")
        assert result == Language.HINDI

    def test_get_language_by_iso_not_found(self) -> None:
        """Should return None for an unknown ISO code."""
        result = get_language_by_iso("xx")
        assert result is None

    def test_get_language_by_name_found(self) -> None:
        """Should find a language by its lowercase name."""
        result = get_language_by_name("arabic")
        assert result == Language.ARABIC

    def test_get_language_by_name_case_insensitive(self) -> None:
        """Should handle mixed case input."""
        result = get_language_by_name("English")
        assert result == Language.ENGLISH

    def test_get_language_by_name_not_found(self) -> None:
        """Should return None for an unknown name."""
        result = get_language_by_name("klingon")
        assert result is None
