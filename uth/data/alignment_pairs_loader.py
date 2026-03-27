"""
Loader for the alignment pairs dataset.

Parses alignment_pairs.json into AlignmentPair objects ready for
TypeAlignmentAnalyzer. The dataset is English-only (3,000 pairs,
1,000 per type: lexical, syntactic, semantic).

Usage::

    pairs = load_alignment_pairs()
    pairs = load_alignment_pairs(max_per_type=100)  # cap for quick runs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from type_alignment.analyzer import AlignmentPair

_DATA_PATH = Path(__file__).parent / "linguistic_variation" / "linguistic_variation.json"

_CONTRAST_LABELS = {
    "lexical": "synonym substitution: same sentence, one word swapped for a synonym",
    "syntactic": "structural variation: same meaning, different grammatical construction",
    "semantic": "paraphrase: same concept expressed with different words",
}


def load_alignment_pairs(
    max_per_type: Optional[int] = None,
    data_path: Optional[Path] = None,
) -> List[AlignmentPair]:
    """
    Load alignment_pairs.json as a list of AlignmentPair objects.

    Args:
        max_per_type: If set, cap the number of pairs loaded per type.
            Useful for quick smoke tests. None loads all pairs.
        data_path: Override the default data file path. Defaults to
            uth/data/linguistic_variation/linguistic_variation.json.

    Returns:
        List of AlignmentPair objects, source_lang and target_lang both "en".

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If a record is missing required fields or has an
            unknown pair type.
    """
    path = Path(data_path) if data_path else _DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Alignment pairs dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    counts: dict[str, int] = {"lexical": 0, "syntactic": 0, "semantic": 0}
    pairs: List[AlignmentPair] = []

    for record in records:
        pair_type = record.get("type", "")
        if pair_type not in counts:
            raise ValueError(
                f"Unknown pair type '{pair_type}' in record {record.get('pair_id')}. "
                f"Expected one of: {list(counts)}"
            )

        if max_per_type is not None and counts[pair_type] >= max_per_type:
            continue

        for field in ("pair_id", "sentence_1", "sentence_2"):
            if field not in record:
                raise ValueError(f"Record missing required field '{field}': {record}")

        pairs.append(
            AlignmentPair(
                source=record["sentence_1"],
                target=record["sentence_2"],
                source_lang="en",
                target_lang="en",
                pair_type=pair_type,
                pair_id=int(record["pair_id"]),
                linguistic_contrast=_CONTRAST_LABELS[pair_type],
            )
        )
        counts[pair_type] += 1

    return pairs
