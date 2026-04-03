"""Data loading, translation, and dataset generation utilities.

Sub-modules
-----------
flores_loader
    Load the FLORES+ parallel corpus from HuggingFace (1,012 aligned
    sentences across 13 languages).  Requires ``HF_TOKEN``.
translate_data_openai
    Translate CSV sentences to multiple languages via OpenAI's
    structured-output API (GPT-4.1).
linguistic_variation
    Generate and review controlled linguistic variation sentence pairs
    (lexical, syntactic, semantic) via the Cohere API.
    Contributed by @danielmargento.
"""
