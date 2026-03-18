import json
import os
from pathlib import Path
from typing import List, Optional

import cohere
from pydantic import BaseModel


LEXICAL_CRITERIA = """A valid lexical pair must:
1. Differ in exactly ONE word — every other token must be identical.
2. The swapped words must be true synonyms or near-synonyms in context.
3. The swap must not change register (formal vs informal).
4. Both sentences must be grammatical and natural."""

SYNTACTIC_CRITERIA = """A valid syntactic pair must:
1. Describe the same event or state — truth-conditionally equivalent.
2. Differ meaningfully in grammatical structure (not just word order).
3. Represent a recognized grammatical alternation (active/passive, cleft, dative alternation, etc.).
4. Both sentences must be grammatical and natural."""

SEMANTIC_CRITERIA = """A valid semantic pair must:
1. Differ substantially in both vocabulary AND syntactic form (not just a synonym swap or passive).
2. Express the same meaning unambiguously — the paraphrase relationship must be clear.
3. Cover a recognized semantic phenomenon (negation paraphrase, entailment, scalar implicature, etc.).
4. Both sentences must be grammatical and natural."""

CRITERIA_BY_TYPE = {
    "lexical": LEXICAL_CRITERIA,
    "syntactic": SYNTACTIC_CRITERIA,
    "semantic": SEMANTIC_CRITERIA,
}


class PairVerdict(BaseModel):
    pair_id: str
    valid: bool
    reason: str


class BatchVerdict(BaseModel):
    verdicts: List[PairVerdict]


class LinguisticVariationReviewer:

    def __init__(self, api_key: Optional[str] = None, model: str = "command-a-03-2025"):
        self.client = cohere.ClientV2(api_key=api_key or os.getenv("COHERE_API_KEY"))
        self.model = model

    def review_batch(self, variation_type: str, pairs: List[dict]) -> List[PairVerdict]:
        criteria = CRITERIA_BY_TYPE[variation_type]
        pairs_text = "\n".join(
            f'[{p["pair_id"]}] s1: "{p["sentence_1"]}" | s2: "{p["sentence_2"]}"'
            for p in pairs
        )

        system_prompt = f"""You are a rigorous computational linguistics reviewer. Your job is to evaluate sentence pairs for a {variation_type} variation dataset.

CRITERIA FOR A VALID {variation_type.upper()} PAIR:
{criteria}

For each pair, return:
- valid: true if it fully satisfies ALL criteria, false if it violates any
- reason: one sentence explaining the verdict (be specific about which criterion fails if invalid)"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review these {variation_type} pairs:\n\n{pairs_text}"},
            ],
            response_format={
                "type": "json_object",
                "json_schema": BatchVerdict.model_json_schema(),
            },
        )
        data = json.loads(response.message.content[0].text)
        return BatchVerdict(**data).verdicts

    def review(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        batch_size: int = 25,
    ) -> dict:
        with open(input_path, "r", encoding="utf-8") as f:
            all_pairs = json.load(f)

        # Group by type
        by_type: dict[str, List[dict]] = {}
        for pair in all_pairs:
            by_type.setdefault(pair["type"], []).append(pair)

        # Duplicate detection across the full dataset (by normalised sentence_1)
        print("\n--- Checking for duplicates ---")
        seen_s1: dict[str, str] = {}  # normalised s1 -> first pair_id
        duplicate_ids: set[str] = set()
        for pair in all_pairs:
            key = pair["sentence_1"].strip().lower()
            if key in seen_s1:
                duplicate_ids.add(pair["pair_id"])
                duplicate_ids.add(seen_s1[key])
            else:
                seen_s1[key] = pair["pair_id"]
        print(f"  Found {len(duplicate_ids)} pairs involved in duplicates")

        all_verdicts: List[dict] = []
        summary: dict[str, dict] = {}

        for variation_type, pairs in by_type.items():
            print(f"\n--- Reviewing {variation_type.upper()} ({len(pairs)} pairs) ---")
            type_verdicts = []

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i: i + batch_size]
                verdicts = self.review_batch(variation_type, batch)
                type_verdicts.extend(verdicts)

                valid_count = sum(1 for v in verdicts if v.valid)
                print(f"  Batch {i // batch_size + 1}: {valid_count}/{len(verdicts)} valid")

            duplicate_count = sum(1 for p in pairs if p["pair_id"] in duplicate_ids)
            valid_total = sum(1 for v in type_verdicts if v.valid and v.pair_id not in duplicate_ids)
            summary[variation_type] = {
                "total": len(pairs),
                "valid": valid_total,
                "invalid": len(pairs) - valid_total,
                "duplicates": duplicate_count,
                "pass_rate": round(valid_total / len(pairs) * 100, 1),
            }

            for v in type_verdicts:
                is_dup = v.pair_id in duplicate_ids
                all_verdicts.append({
                    "pair_id": v.pair_id,
                    "type": variation_type,
                    "valid": v.valid and not is_dup,
                    "duplicate": is_dup,
                    "reason": "duplicate sentence_1" if is_dup else v.reason,
                })

        print("\n=== SUMMARY ===")
        for t, s in summary.items():
            print(f"  {t}: {s['valid']}/{s['total']} valid ({s['pass_rate']}%) | {s['duplicates']} duplicates")

        results = {"summary": summary, "verdicts": all_verdicts}

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nSaved review results to {path}")

        return results


if __name__ == "__main__":
    reviewer = LinguisticVariationReviewer(model="command-a-03-2025")

    reviewer.review(
        input_path="linguistic_variation.json",
        output_path="linguistic_variation_review.json",
        batch_size=25,
    )
