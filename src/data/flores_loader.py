"""
FLORES+ parallel corpus loader for cross-lingual alignment analysis.

This module loads semantically aligned sentences from the FLORES+
benchmark dataset (``openlanguagedata/flores_plus`` on HuggingFace),
providing a clean interface for extracting parallel text across the
13 target languages defined in ``src.utils.languages``.

FLORES+ is the actively maintained successor to FLORES-200, managed
by the Open Language Data Initiative (OLDI). Its ``devtest`` split
contains 1,012 sentences that are professionally translated across
228+ language varieties, making it ideal for controlled cross-lingual
representation analysis where semantic equivalence is guaranteed.

Key design decisions:
    - We use the ``devtest`` split (1,012 sentences) rather than
      ``dev`` (997 sentences) because ``devtest`` is the standard
      evaluation split and has slightly more data.
    - Sentences are returned as a dictionary keyed by language name
      (matching ``Language.lang_name``), ensuring compatibility with
      all downstream analysis modules.
    - An optional ``max_sentences`` parameter supports development
      and debugging with smaller subsets.
    - Authentication is handled via ``HF_TOKEN`` from a ``.env``
      file in the project root. The ``python-dotenv`` library loads
      this automatically so users only need to set it once.

Authentication setup:
    FLORES+ is a gated dataset. To access it:
    1. Create a HuggingFace account at https://huggingface.co/join
    2. Accept the dataset terms at
       https://huggingface.co/datasets/openlanguagedata/flores_plus
    3. Create an access token at https://huggingface.co/settings/tokens
    4. Add ``HF_TOKEN=hf_your_token_here`` to a ``.env`` file in
       the project root directory.

Usage::

    from src.data.flores_loader import load_flores_parallel_corpus
    from src.utils.languages import Language

    # Load all 1,012 parallel sentences for all 13 languages
    corpus = load_flores_parallel_corpus()
    print(len(corpus["english"]))  # 1012

    # Load a subset for quick testing
    corpus_small = load_flores_parallel_corpus(max_sentences=100)

    # Load only specific languages
    corpus_subset = load_flores_parallel_corpus(
        languages=[Language.ENGLISH, Language.HINDI, Language.ARABIC]
    )

References:
    - FLORES+ dataset: https://huggingface.co/datasets/openlanguagedata/flores_plus
    - NLLB paper: https://doi.org/10.1038/s41586-024-07335-x
    - Tiny Aya: https://huggingface.co/collections/CohereLabs/tiny-aya
"""


import logging
import os
from pathlib import Path

from src.utils.languages import Language

# Module-level logger for diagnostic output.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The HuggingFace dataset identifier for FLORES+.
_FLORES_DATASET_ID = "openlanguagedata/flores_plus"

# The evaluation split we use for analysis.
_FLORES_SPLIT = "devtest"


# ---------------------------------------------------------------------------
# Environment / auth helpers
# ---------------------------------------------------------------------------

def _ensure_hf_auth() -> str | None:
    """Load the HuggingFace token from the environment.

    Attempts to load ``HF_TOKEN`` from:
    1. The process environment (already set).
    2. A ``.env`` file in the project root (via ``python-dotenv``).

    If a token is found, it is also set in the ``huggingface_hub``
    login so that ``datasets.load_dataset`` can use it transparently.

    Returns:
        The HuggingFace token string, or ``None`` if not found.
    """
    # Try loading from .env file if not already in the environment.
    try:
        from dotenv import load_dotenv

        # Walk up from this file to find the project root .env.
        # This file lives at src/data/flores_loader.py, so the
        # project root is two directories up.
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug("Loaded .env from %s", env_path)
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env loading.")

    token = os.environ.get("HF_TOKEN")

    if token:
        # Set the token for huggingface_hub so datasets can use it.
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.debug("HuggingFace authentication configured.")
        except ImportError:
            # If huggingface_hub is not available, datasets will
            # fall back to the HF_TOKEN env var directly.
            pass
    else:
        logger.warning(
            "HF_TOKEN not found in environment or .env file. "
            "FLORES+ is a gated dataset and requires authentication. "
            "See the module docstring for setup instructions."
        )

    return token


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_flores_parallel_corpus(
    languages: list[Language] | None = None,
    max_sentences: int | None = None,
    split: str = _FLORES_SPLIT,
    cache_dir: str | None = None,
) -> dict[str, list[str]]:
    """Load parallel sentences from the FLORES+ dataset.

    Downloads (or loads from cache) the FLORES+ dataset from
    HuggingFace Hub, then extracts aligned sentences for each
    requested language. All returned lists are guaranteed to have
    the same length and sentence alignment -- ``corpus["english"][i]``
    is the translation of ``corpus["hindi"][i]`` for all ``i``.

    FLORES+ stores each language as a separate config (e.g.,
    ``"eng_Latn"``, ``"hin_Deva"``). This function loads each config
    separately, extracts the ``"text"`` column, and aligns them by
    row index (which corresponds to the sentence ``"id"`` field).

    Args:
        languages: List of ``Language`` enum members to load. If
            ``None``, loads all 13 languages defined in the registry.
        max_sentences: Maximum number of sentences to load per
            language. If ``None``, loads the full split (1,012
            sentences for ``devtest``). Useful for development
            and debugging.
        split: Dataset split to load. Defaults to ``"devtest"`` (the
            standard evaluation split with 1,012 sentences).
        cache_dir: Optional directory for caching the downloaded
            dataset. If ``None``, uses the HuggingFace default
            cache directory (``~/.cache/huggingface/datasets``).

    Returns:
        Dictionary mapping lowercase language names to lists of
        sentences. All lists have identical length and are aligned
        by index (i.e., same-index entries are translations of
        each other).

    Raises:
        ImportError: If the ``datasets`` library is not installed.
        ValueError: If a requested language's FLORES code is not
            found as a valid config in the dataset.
        RuntimeError: If the dataset download or loading fails
            (e.g., missing authentication, network error).

    Example::

        >>> corpus = load_flores_parallel_corpus(max_sentences=5)
        >>> len(corpus["english"])
        5
        >>> len(corpus["hindi"])
        5
    """
    # --- Lazy import to avoid hard dependency at module load time ---
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required for FLORES+ loading. "
            "Install it with: uv pip install datasets"
        ) from exc

    # --- Ensure HuggingFace authentication is configured ---
    _ensure_hf_auth()

    # --- Default to all 13 languages if none specified ---
    if languages is None:
        languages = list(Language)

    if not languages:
        raise ValueError("At least one language must be specified.")

    logger.info(
        "Loading FLORES+ '%s' split for %d languages...",
        split,
        len(languages),
    )

    # ------------------------------------------------------------------
    # Load each language's config separately and extract sentences
    # ------------------------------------------------------------------
    corpus: dict[str, list[str]] = {}

    for lang in languages:
        flores_code = lang.flores_code

        logger.info(
            "  Loading %s (%s)...",
            lang.lang_name,
            flores_code,
        )

        try:
            # FLORES+ uses per-language configs (e.g., "eng_Latn").
            # Each config contains columns: id, text, iso_639_3, etc.
            dataset = load_dataset(
                _FLORES_DATASET_ID,
                name=flores_code,
                split=split,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FLORES+ config '{flores_code}' for "
                f"language '{lang.lang_name}'. This may be caused by:\n"
                f"  1. Missing HF_TOKEN in .env (FLORES+ is gated).\n"
                f"  2. You haven't accepted the dataset terms at:\n"
                f"     https://huggingface.co/datasets/{_FLORES_DATASET_ID}\n"
                f"  3. Invalid FLORES code '{flores_code}'.\n"
                f"  4. Network connectivity issue.\n"
                f"Original error: {exc}"
            ) from exc

        # Validate that the 'text' column exists.
        if "text" not in dataset.column_names:
            raise ValueError(
                f"FLORES+ config '{flores_code}' does not contain a "
                f"'text' column. Available columns: "
                f"{dataset.column_names}"
            )

        # Extract the text column as a plain Python list.
        sentences = dataset["text"]

        # Apply the sentence limit if requested.
        if max_sentences is not None:
            sentences = sentences[:max_sentences]

        corpus[lang.lang_name] = sentences

        logger.debug(
            "    Loaded %d sentences for %s.",
            len(sentences),
            lang.lang_name,
        )

    # ------------------------------------------------------------------
    # Validate alignment: all languages must have the same count
    # ------------------------------------------------------------------
    sentence_counts = {name: len(sents) for name, sents in corpus.items()}
    unique_counts = set(sentence_counts.values())

    if len(unique_counts) != 1:
        raise RuntimeError(
            f"Sentence count mismatch across languages -- parallel "
            f"alignment is broken. Counts: {sentence_counts}"
        )

    num_sentences = unique_counts.pop()
    logger.info(
        "Successfully loaded %d parallel sentences for %d languages.",
        num_sentences,
        len(corpus),
    )

    return corpus


def get_corpus_statistics(corpus: dict[str, list[str]]) -> dict[str, dict]:
    """Compute basic statistics for a parallel corpus.

    Useful for exploratory data analysis and sanity checking
    the loaded corpus before running expensive model inference.

    Args:
        corpus: Dictionary mapping language names to sentence lists,
            as returned by ``load_flores_parallel_corpus``.

    Returns:
        Dictionary mapping language names to stat dictionaries, each
        containing:
            - ``num_sentences``: Number of sentences.
            - ``avg_char_length``: Mean character count per sentence.
            - ``min_char_length``: Shortest sentence character count.
            - ``max_char_length``: Longest sentence character count.
            - ``avg_word_count``: Mean whitespace-tokenized word count.

    Example::

        >>> stats = get_corpus_statistics(corpus)
        >>> stats["english"]["avg_word_count"]
        18.5
    """
    statistics: dict[str, dict] = {}

    for lang_name, sentences in corpus.items():
        if not sentences:
            statistics[lang_name] = {
                "num_sentences": 0,
                "avg_char_length": 0.0,
                "min_char_length": 0,
                "max_char_length": 0,
                "avg_word_count": 0.0,
            }
            continue

        char_lengths = [len(s) for s in sentences]
        word_counts = [len(s.split()) for s in sentences]

        statistics[lang_name] = {
            "num_sentences": len(sentences),
            "avg_char_length": sum(char_lengths) / len(char_lengths),
            "min_char_length": min(char_lengths),
            "max_char_length": max(char_lengths),
            "avg_word_count": sum(word_counts) / len(word_counts),
        }

    return statistics
