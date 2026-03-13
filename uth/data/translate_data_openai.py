"""
Translation pipeline for processing CSV files with OpenAI.

This module provides a complete pipeline for translating sentences from CSV files
using OpenAI's structured output API.
"""

import csv
import json
import os
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel, Field

from uth.utils import Language


# ============================================================================
# Pydantic Models
# ============================================================================

class TranslationItem(BaseModel):
    """Single translation result with sentence ID."""

    sentence_id: str = Field(..., description="Original sentence ID from input")
    translated_text: str = Field(..., description="Translated text in target language")


class TranslationBatch(BaseModel):
    """Batch of translation results."""

    translations: List[TranslationItem] = Field(
        ...,
        description="List of translated sentences with their IDs"
    )


# ============================================================================
# Prompt Templates
# ============================================================================

TRANSLATION_SYSTEM_PROMPT = """You are a professional translator. Your task is to translate sentences from the source language to the target language while preserving meaning, tone, and context.

For each sentence provided, you must:
1. Translate it accurately to the target language
2. Maintain the original meaning and nuance
3. Return the translation along with the sentence ID

Be precise and consistent in your translations."""


def create_translation_prompt(sentences: list[dict], target_language: str) -> str:
    """
    Create a translation prompt for a batch of sentences.

    Args:
        sentences: List of dicts with 'sentence_id' and 'text' keys
        target_language: Target language for translation

    Returns:
        Formatted prompt string
    """
    sentences_text = "\n".join([
        f"ID: {sent['sentence_id']}\nText: {sent['text']}"
        for sent in sentences
    ])

    prompt = f"""Translate the following sentences to {target_language}.

For each sentence, return the sentence ID and the translated text.

Sentences to translate:
{sentences_text}

Please provide the translations in the exact format specified, maintaining the sentence IDs."""

    return prompt


# ============================================================================
# Translation Pipeline
# ============================================================================

class TranslationPipeline:
    """
    Pipeline for translating sentences from CSV using OpenAI's structured output.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4,1",
        batch_size: int = 10,
    ):
        """
        Initialize the translation pipeline.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use for translation
            batch_size: Number of sentences to process in a single API call
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.batch_size = batch_size

    def read_csv(self, file_path: str | Path) -> List[dict]:
        """
        Read sentences from CSV file.

        Args:
            file_path: Path to CSV file with 'sentence_id' and 'text' columns

        Returns:
            List of dicts with sentence_id and text
        """
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate required columns
            if 'text' not in reader.fieldnames:
                raise ValueError("CSV must contain 'text' column")

            # Use sentence_id if available, otherwise create one
            has_id = 'sentence_id' in reader.fieldnames

            for idx, row in enumerate(reader):
                sentence_id = row.get('sentence_id', str(idx)) if has_id else str(idx)
                sentences.append({
                    'sentence_id': sentence_id,
                    'text': row['text']
                })

        return sentences

    def translate_batch(
        self,
        sentences: List[dict],
        target_language: str | Language,
    ) -> List[TranslationItem]:
        """
        Translate a batch of sentences using OpenAI structured output.

        Args:
            sentences: List of dicts with 'sentence_id' and 'text'
            target_language: Target language (string or Language enum)

        Returns:
            List of TranslationItem objects
        """
        # Convert Language enum to string if needed
        if isinstance(target_language, Language):
            target_language = target_language.value

        # Create prompt
        prompt = create_translation_prompt(sentences, target_language)

        # Call OpenAI with structured output
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format=TranslationBatch,
        )

        # Extract parsed response
        result = completion.choices[0].message.parsed
        return result.translations

    def translate_file(
        self,
        input_file: str | Path,
        target_languages: List[Language],
        output_file: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        return_as_json: bool = False,
    ) -> List[dict]:
        """
        Translate all sentences in a CSV file to multiple languages.

        Args:
            input_file: Path to input CSV file
            target_languages: List of target languages for translation
            output_file: Optional path to save results as CSV or JSON
            output_dir: Optional directory to save output with auto-generated filename
            return_as_json: If True, save as JSON instead of CSV

        Returns:
            List of all translation results with metadata
        """
        # Read input sentences
        print(f"Reading sentences from {input_file}...")
        sentences = self.read_csv(input_file)
        print(f"Found {len(sentences)} sentences")

        # Create a mapping of sentence_id to original text for later reference
        original_texts = {sent['sentence_id']: sent['text'] for sent in sentences}

        # Collect all translations across all languages
        all_results = []

        # Loop through each target language
        for target_language in target_languages:
            print(f"\n{'='*60}")
            print(f"Translating to {target_language.value}...")
            print(f"{'='*60}")

            # Process in batches for this language
            language_translations = []
            num_batches = (len(sentences) + self.batch_size - 1) // self.batch_size

            for i in tqdm(
                range(0, len(sentences), self.batch_size),
                desc=f"Translating to {target_language.value}"
            ):
                batch = sentences[i:i + self.batch_size]
                translations = self.translate_batch(batch, target_language)
                language_translations.extend(translations)

            print(f"Completed {len(language_translations)} translations for {target_language.value}")

            # Convert translations to final format with all metadata
            for translation in language_translations:
                all_results.append({
                    'sentence_alignment_id': translation.sentence_id,
                    'text': translation.translated_text,
                    'language_ID': target_language.value,
                    'original_text': original_texts[translation.sentence_id]
                })

        print(f"\n{'='*60}")
        print(f"Total translations: {len(all_results)} across {len(target_languages)} languages")
        print(f"{'='*60}")

        # Determine output file path
        final_output_file = None
        if output_dir:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "json" if return_as_json else "csv"
            filename = f"translated_dataset_{timestamp}.{extension}"

            # Create output directory if it doesn't exist
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            final_output_file = output_dir_path / filename
        elif output_file:
            final_output_file = output_file

        # Save to file if output path determined
        if final_output_file:
            if return_as_json:
                self.save_to_json(all_results, final_output_file)
            else:
                self.save_to_csv(all_results, final_output_file)

        return all_results

    def save_to_csv(
        self,
        results: List[dict],
        output_file: str | Path,
    ):
        """
        Save translation results to CSV file.

        Args:
            results: List of dicts with translation data
            output_file: Path to output CSV file
        """
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['text', 'sentence_alignment_id', 'language_ID', 'original_text']
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved {len(results)} translations to {output_file}")

    def save_to_json(
        self,
        results: List[dict],
        output_file: str | Path,
    ):
        """
        Save translation results to JSON file.

        Args:
            results: List of dicts with translation data
            output_file: Path to output JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(results)} translations to {output_file}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the translation pipeline.
    """

    # Initialize pipeline
    pipeline = TranslationPipeline(
        model="gpt-4.1",
        batch_size=10,
    )

    # Define target languages
    target_languages = [
        Language.HINDI,
        Language.BENGALI,
        Language.TAMIL,
        Language.ARABIC,
    ]

    input_file = "/Users/roshansk/Documents/GitHub/Tiny-Aya-Under-the-hood/uth/data/test_data.csv"

    # Example 1: Save with explicit output file (CSV)
    results_csv = pipeline.translate_file(
        input_file=input_file,
        target_languages=target_languages,
        output_file="/Users/roshansk/Documents/GitHub/Tiny-Aya-Under-the-hood/uth/data/multilingual_translations.csv",
        return_as_json=False,
    )


    print(f"\nTotal results: {len(results_csv)}")
