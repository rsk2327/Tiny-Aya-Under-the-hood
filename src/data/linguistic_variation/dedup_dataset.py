"""Deduplicate the linguistic variation dataset.

Removes two kinds of duplicates from ``linguistic_variation.json``:

1. **Exact sentence duplicates** -- pairs whose ``sentence_1`` has
   already been seen (case-insensitive).
2. **Reused word swaps** (lexical type only) -- pairs that swap the
   same ordered word pair as an earlier entry (e.g. two pairs both
   swapping *bought* / *purchased*).

The cleaned dataset is written back to the same JSON file.

Usage::

    python -m src.data.linguistic_variation.dedup_dataset
"""

import json
from pathlib import Path


def extract_swap(s1: str, s2: str) -> tuple[str, str] | None:
    """Return the single differing word pair between two sentences.

    Both sentences are split on whitespace and compared token by
    token (case-insensitive, punctuation stripped).  If exactly one
    position differs, the two differing words are returned as a
    sorted tuple.

    Parameters
    ----------
    s1 : str
        First sentence.
    s2 : str
        Second sentence.

    Returns
    -------
    tuple[str, str] or None
        Sorted ``(word_a, word_b)`` when exactly one word differs,
        or *None* when the sentences differ in length or in more
        (or fewer) than one position.
    """
    w1 = s1.strip().split()
    w2 = s2.strip().split()
    if len(w1) != len(w2):
        return None
    # Strip trailing punctuation before comparing so that
    # "store." and "store" are treated as identical.
    diffs = [
        (a.lower().strip(".,!?;:"), b.lower().strip(".,!?;:"))
        for a, b in zip(w1, w2)
        if a.lower() != b.lower()
    ]
    return tuple(sorted(diffs[0])) if len(diffs) == 1 else None


def dedup(data_path: str = "linguistic_variation.json") -> None:
    """Remove duplicate pairs from the linguistic variation dataset.

    Reads *data_path*, groups pairs by variation type, applies the
    deduplication rules described in the module docstring, and
    overwrites the file with the cleaned result.

    Parameters
    ----------
    data_path : str
        Path to the JSON file containing the generated pairs.
        Defaults to ``linguistic_variation.json`` (relative to CWD).
    """
    path = Path(data_path)
    with open(path) as f:
        pairs = json.load(f)

    by_type: dict[str, list[dict]] = {}
    for p in pairs:
        by_type.setdefault(p["type"], []).append(p)

    cleaned: list[dict] = []
    for vtype, type_pairs in by_type.items():
        before = len(type_pairs)
        seen_s1: set[str] = set()
        seen_swaps: set[tuple[str, str]] = set()
        kept: list[dict] = []

        for p in type_pairs:
            s1_key = p["sentence_1"].strip().lower()

            # Drop exact sentence duplicates.
            if s1_key in seen_s1:
                continue
            seen_s1.add(s1_key)

            # For lexical pairs, also drop reused word swaps.
            if vtype == "lexical":
                swap = extract_swap(p["sentence_1"], p["sentence_2"])
                if swap is None or swap in seen_swaps:
                    continue
                seen_swaps.add(swap)

            kept.append(p)

        removed = before - len(kept)
        print(f"{vtype}: {before} → {len(kept)} ({removed} removed)")
        cleaned.extend(kept)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(cleaned)} pairs to {path}")


if __name__ == "__main__":
    dedup()
