"""
Language enumeration for multilingual analysis.
"""

from enum import Enum


class Language(Enum):
    """Enumeration of supported languages for analysis."""

    HINDI = "hindi"
    ENGLISH = "english"
    BENGALI = 'bengali'
    TAMIL = 'tamil'
    SWAHILI = 'swahili'
    AMHARIC = 'amharic'
    YORUBA = 'yoruba'
    ARABIC = 'arabic'
    TURKISH = 'turkish'
    PERSIAN = 'persian'
    GERMAN = 'german'
    FRENCH = 'french'
    SPANISH = 'spanish'


