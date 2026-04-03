"""Translation pipeline for processing CSV files with OpenAI.

This module provides a complete pipeline for translating sentences
from CSV files to multiple target languages using OpenAI's
structured-output API (``response_format`` with Pydantic models).

The pipeline:
    1. Reads sentences from a CSV file (``sentence_id``, ``text``).
    2. Batches them into groups of ``batch_size``.
    3. Sends each batch to the OpenAI Chat Completions API with a
       ``TranslationBatch`` Pydantic schema as the response format.
    4. Collects results and writes to CSV or JSON.

Authentication:
    Requires ``OPENAI_API_KEY`` in the environment or passed
    directly to :class:`TranslationPipeline`.

Usage::

    from src.data.translate_data_openai import TranslationPipeline
    from src.utils import Language

    pipeline = TranslationPipeline(model="gpt-4.1", batch_size=10)
    results = pipeline.translate_file(
        input_file="data/test_data.csv",
        target_languages=[Language.HINDI, Language.ARABIC],
        output_file="data/translations.csv",
    )
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.utils import Language

# ============================================================================
# Pydantic Models
# ============================================================================


class TranslationItem(BaseModel):
    """A single translated sentence returned by the OpenAI API.

    Attributes
    ----------
    sentence_id : str
        The original sentence identifier from the input CSV, used to
        join translations back to source text.
    translated_text : str
        The translated text in the target language.
    """

    sentence_id: str = Field(..., description="Original sentence ID from input")
    translated_text: str = Field(..., description="Translated text in target language")


class TranslationBatch(BaseModel):
    """Container for a batch of translation results.

    Used as the ``response_format`` in the OpenAI structured-output
    call so the API returns type-safe, parseable JSON.

    Attributes
    ----------
    translations : list[TranslationItem]
        Ordered list of translated sentences, one per input sentence.
    """

    translations: list[TranslationItem] = Field(
        ...,
        description="List of translated sentences with their IDs",
    )


# ============================================================================
# Prompt Templates
# ============================================================================

#: System prompt instructing the model to behave as a professional
#: translator.  Sent as the ``system`` message in every API call.
TRANSLATION_SYSTEM_PROMPT = """You are a professional translator. Your task is to translate sentences from the source language to the target language while preserving meaning, tone, and context.

For each sentence provided, you must:
1. Translate it accurately to the target language
2. Maintain the original meaning and nuance
3. Return the translation along with the sentence ID

Be precise and consistent in your translations."""


def create_translation_prompt(sentences: list[dict], target_language: str) -> str:
    """Build the user-message prompt for a translation batch.

    Parameters
    ----------
    sentences : list[dict]
        Each dict must contain ``"sentence_id"`` and ``"text"`` keys.
    target_language : str
        Human-readable target language name (e.g. ``"hindi"``).

    Returns
    -------
    str
        Formatted prompt ready to send as the ``user`` message.
    """
    sentences_text = "\n".join(
        [f"ID: {sent['sentence_id']}\nText: {sent['text']}" for sent in sentences]
    )

    return f"""Translate the following sentences to {target_language}.

For each sentence, return the sentence ID and the translated text.

Sentences to translate:
{sentences_text}

Please provide the translations in the exact format specified, maintaining the sentence IDs."""


# ============================================================================
# Translation Pipeline
# ============================================================================


class TranslationPipeline:
    """End-to-end pipeline for CSV translation via OpenAI structured output.

    The pipeline reads an input CSV, translates every sentence to each
    requested target language in batches, and writes the results to
    CSV or JSON.

    Attributes
    ----------
    client : openai.OpenAI
        Authenticated OpenAI client instance.
    model : str
        Model identifier (e.g. ``"gpt-4.1"``).
    batch_size : int
        Number of sentences per API call.

    Examples
    --------
    >>> pipeline = TranslationPipeline(model="gpt-4.1", batch_size=10)
    >>> results = pipeline.translate_file(
    ...     "data/test_data.csv",
    ...     target_languages=[Language.HINDI],
    ...     output_file="data/out.csv",
    ... )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4,1",
        batch_size: int = 10,
    ) -> None:
        """Initialise the pipeline.

        Parameters
        ----------
        api_key : str or None
            OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when *None*.
        model : str
            OpenAI model identifier.
        batch_size : int
            Sentences per API call.

        Raises
        ------
        openai.AuthenticationError
            If no valid API key can be resolved.
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.batch_size = batch_size

    def read_csv(self, file_path: str | Path) -> list[dict]:
        """Read sentences from a CSV file.

        Parameters
        ----------
        file_path : str or Path
            Path to a CSV with at least a ``text`` column.  An optional
            ``sentence_id`` column is used as-is; otherwise sequential
            integer IDs are generated.

        Returns
        -------
        list[dict]
            Each dict has ``"sentence_id"`` (str) and ``"text"`` (str).

        Raises
        ------
        ValueError
            If the CSV does not contain a ``text`` column.
        FileNotFoundError
            If *file_path* does not exist.
        """
        sentences = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if "text" not in reader.fieldnames:
                raise ValueError("CSV must contain 'text' column")

            has_id = "sentence_id" in reader.fieldnames

            for idx, row in enumerate(reader):
                sentence_id = row.get("sentence_id", str(idx)) if has_id else str(idx)
                sentences.append({"sentence_id": sentence_id, "text": row["text"]})

        return sentences

    def translate_batch(
        self,
        sentences: list[dict],
        target_language: str | Language,
    ) -> list[TranslationItem]:
        """Translate a single batch of sentences via the OpenAI API.

        Uses the ``beta.chat.completions.parse`` endpoint with
        :class:`TranslationBatch` as the ``response_format`` so the
        response is guaranteed to conform to our Pydantic schema.

        Parameters
        ----------
        sentences : list[dict]
            Dicts with ``"sentence_id"`` and ``"text"`` keys.
        target_language : str or Language
            Target language name or ``Language`` enum member.

        Returns
        -------
        list[TranslationItem]
            One item per input sentence.

        Raises
        ------
        openai.APIError
            On any upstream API failure.
        """
        if isinstance(target_language, Language):
            target_language = target_language.value

        prompt = create_translation_prompt(sentences, target_language)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=TranslationBatch,
        )

        result = completion.choices[0].message.parsed
        return result.translations

    def translate_file(
        self,
        input_file: str | Path,
        target_languages: list[Language],
        output_file: str | Path | None = None,
        output_dir: str | Path | None = None,
        return_as_json: bool = False,
    ) -> list[dict]:
        """Translate every sentence in a CSV to multiple languages.

        Parameters
        ----------
        input_file : str or Path
            Input CSV (must have a ``text`` column).
        target_languages : list[Language]
            Languages to translate into.
        output_file : str, Path, or None
            Explicit output path.  Mutually exclusive with *output_dir*.
        output_dir : str, Path, or None
            Directory for auto-named output (timestamped filename).
        return_as_json : bool
            Write JSON instead of CSV when saving.

        Returns
        -------
        list[dict]
            Each dict has ``sentence_alignment_id``, ``text``,
            ``language_ID``, and ``original_text``.

        Raises
        ------
        ValueError
            If the input CSV is malformed (missing ``text`` column).
        openai.APIError
            On any upstream API failure during batch translation.
        """
        print(f"Reading sentences from {input_file}...")
        sentences = self.read_csv(input_file)
        print(f"Found {len(sentences)} sentences")

        original_texts = {sent["sentence_id"]: sent["text"] for sent in sentences}
        all_results: list[dict] = []

        for target_language in target_languages:
            print(f"\n{'=' * 60}")
            print(f"Translating to {target_language.value}...")
            print(f"{'=' * 60}")

            language_translations: list[TranslationItem] = []

            for i in tqdm(
                range(0, len(sentences), self.batch_size),
                desc=f"Translating to {target_language.value}",
            ):
                batch = sentences[i : i + self.batch_size]
                translations = self.translate_batch(batch, target_language)
                language_translations.extend(translations)

            print(
                f"Completed {len(language_translations)} translations "
                f"for {target_language.value}"
            )

            for translation in language_translations:
                all_results.append(
                    {
                        "sentence_alignment_id": translation.sentence_id,
                        "text": translation.translated_text,
                        "language_ID": target_language.value,
                        "original_text": original_texts[translation.sentence_id],
                    }
                )

        print(f"\n{'=' * 60}")
        print(f"Total translations: {len(all_results)} across {len(target_languages)} languages")
        print(f"{'=' * 60}")

        # Determine output path
        final_output_file = None
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "json" if return_as_json else "csv"
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            final_output_file = output_dir_path / f"translated_dataset_{timestamp}.{extension}"
        elif output_file:
            final_output_file = output_file

        if final_output_file:
            if return_as_json:
                self.save_to_json(all_results, final_output_file)
            else:
                self.save_to_csv(all_results, final_output_file)

        return all_results

    def save_to_csv(self, results: list[dict], output_file: str | Path) -> None:
        """Write translation results to a CSV file.

        Parameters
        ----------
        results : list[dict]
            Rows to write (``text``, ``sentence_alignment_id``,
            ``language_ID``, ``original_text``).
        output_file : str or Path
            Destination path (created or overwritten).
        """
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["text", "sentence_alignment_id", "language_ID", "original_text"],
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved {len(results)} translations to {output_file}")

    def save_to_json(self, results: list[dict], output_file: str | Path) -> None:
        """Write translation results to a JSON file.

        Parameters
        ----------
        results : list[dict]
            Rows to write.
        output_file : str or Path
            Destination path (created or overwritten).
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(results)} translations to {output_file}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Quick smoke-test: translate 10 sentences to 4 languages.
    pipeline = TranslationPipeline(model="gpt-4.1", batch_size=10)

    target_languages = [
        Language.HINDI,
        Language.BENGALI,
        Language.TAMIL,
        Language.ARABIC,
    ]

    results_csv = pipeline.translate_file(
        input_file="data/test_data.csv",
        target_languages=target_languages,
        output_file="data/multilingual_translations.csv",
        return_as_json=False,
    )

    print(f"\nTotal results: {len(results_csv)}")
