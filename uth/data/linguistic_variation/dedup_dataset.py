import json
from pathlib import Path


def extract_swap(s1: str, s2: str):
    """Return the single swapped word pair, or None if diff != exactly 1 word."""
    w1 = s1.strip().split()
    w2 = s2.strip().split()
    if len(w1) != len(w2):
        return None
    diffs = [(a.lower().strip(".,!?;:"), b.lower().strip(".,!?;:"))
             for a, b in zip(w1, w2) if a.lower() != b.lower()]
    return tuple(sorted(diffs[0])) if len(diffs) == 1 else None


def dedup(data_path: str = "linguistic_variation.json"):
    path = Path(data_path)
    with open(path) as f:
        pairs = json.load(f)

    by_type = {}
    for p in pairs:
        by_type.setdefault(p["type"], []).append(p)

    cleaned = []
    for vtype, type_pairs in by_type.items():
        before = len(type_pairs)
        seen_s1 = set()
        seen_swaps = set()
        kept = []

        for p in type_pairs:
            s1_key = p["sentence_1"].strip().lower()

            # Drop exact sentence duplicates
            if s1_key in seen_s1:
                continue
            seen_s1.add(s1_key)

            # For lexical: drop pairs that reuse the exact same swap pair
            if vtype == "lexical":
                swap = extract_swap(p["sentence_1"], p["sentence_2"])
                if swap is None or swap in seen_swaps:
                    continue
                seen_swaps.add(swap)  # swap is already a sorted tuple, e.g. ("bought", "purchased")

            kept.append(p)

        removed = before - len(kept)
        print(f"{vtype}: {before} → {len(kept)} ({removed} removed)")
        cleaned.extend(kept)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(cleaned)} pairs to {path}")


if __name__ == "__main__":
    dedup()
