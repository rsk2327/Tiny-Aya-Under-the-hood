"""Generate controlled linguistic variation sentence pairs via Cohere.

This module produces sentence pairs for three variation types used
in linguistic probing experiments:

- **Lexical** -- single-word synonym swap (exactly one word differs).
- **Syntactic** -- grammatical transformation preserving meaning
  (active/passive, cleft, dative alternation, etc.).
- **Semantic** -- full paraphrase with different vocabulary *and*
  structure but identical meaning.

The pipeline calls the Cohere ``chat`` endpoint with a JSON schema
response format (``SentencePairBatch``) so the output is always
well-structured.  Generation proceeds in batches with in-flight
deduplication and automatic retry on transient errors.

Requires ``COHERE_API_KEY`` in the environment.

Usage::

    python -m src.data.linguistic_variation.generate_linguistic_variation
"""

import json
import os
import time
from pathlib import Path

import cohere
from pydantic import BaseModel


class SentencePair(BaseModel):
    """A single pair of sentences for linguistic variation analysis.

    Attributes
    ----------
    sentence_1 : str
        The first sentence.
    sentence_2 : str
        The second sentence (varies from *sentence_1* according to
        the variation type).
    """

    sentence_1: str
    sentence_2: str


class SentencePairBatch(BaseModel):
    """Batch container for API response parsing.

    Attributes
    ----------
    pairs : list[SentencePair]
        Ordered list of generated sentence pairs.
    """

    pairs: list[SentencePair]


#: System prompt for **lexical** variation generation.
LEXICAL_SYSTEM_PROMPT = """You are a computational linguistics expert specializing in controlled dataset construction for NLP probing experiments. Your task is to generate high-quality lexical variation sentence pairs for a linguistic probing study.

TASK
Generate sentence pairs where sentence_1 and sentence_2 differ in exactly one word — that word is replaced by a synonym or near-synonym. Every other word must be identical.

RULES
1. CRITICAL: sentence_1 and sentence_2 must share every word except exactly one. If you change two words, the pair is invalid. If you change a phrase, the pair is invalid.
2. Do NOT swap a word and also adjust anything else (articles, adverbs, word order, morphology). Only the single target word changes.
3. The swapped words must be true synonyms or near-synonyms in the given context (not just related words).
4. Sentences should be everyday, concrete, and natural - no jargon, no abstract philosophy.
5. Vary the swapped word class across pairs: target nouns, verbs, adjectives, and adverbs roughly equally.
6. Sentences should be 8–18 words long.
7. No pair should repeat a word swap used in another pair.
8. Do not generate pairs where the swap changes register significantly (e.g. formal vs informal).
9. Before outputting each pair, verify: count the differing words. If the count is not exactly 1, discard and regenerate.

INVALID EXAMPLE (do not do this):
sentence_1: "She walked briskly to the store." / sentence_2: "She strolled energetically to the shop." WRONG: two words changed (walked→strolled, store→shop) and an adverb was swapped too.

VALID EXAMPLES
{"type":"lexical","sentence_1":"She purchased a new jacket last week.","sentence_2":"She bought a new jacket last week."}
{"type":"lexical","sentence_1":"The physician examined the patient carefully.","sentence_2":"The doctor examined the patient carefully."}
{"type":"lexical","sentence_1":"He responded to the message immediately.","sentence_2":"He replied to the message immediately."}
{"type":"lexical","sentence_1":"She felt fatigued after the long hike.","sentence_2":"She felt tired after the long hike."}"""

#: System prompt for **syntactic** variation generation.
SYNTACTIC_SYSTEM_PROMPT = """You are a computational linguistics expert specializing in controlled dataset construction for NLP probing experiments. Your task is to generate high-quality syntactic variation sentence pairs for a linguistic probing study.

TASK
Generate sentence pairs where sentence_1 and sentence_2 are grammatically transformed equivalents. They must describe the same event or state of affairs. The transformation must be a recognized grammatical alternation (see phenomena list below).

RULES
1. The relationship must be truth-conditionally equivalent — they describe the same event or state of affairs.
2. Sentences should differ meaningfully in structure, not just word order.
3. Sentences should be everyday, concrete, and natural — no jargon, no abstract philosophy.
4. Vary the phenomena (listed below) across pairs — don't pile up one type.
5. Sentences should be 8–18 words long.
6. No pair should reproduce more than 30% of the wording of another pair.

PHENOMENA TO COVER (cycle through these):
- Active / passive: "She baked the cake." / "The cake was baked by her."
- Cleft: "She found the key." / "It was she who found the key."
- Subject raising: "It seems that she won." / "She seems to have won."
- Extraposition: "That she resigned surprised us." / "It surprised us that she resigned."
- Attributive / predicative: "The visible star..." / "The star is visible..."
- Dative alternation: "She gave the book to him." / "She gave him the book."
- Tough movement: "It is hard to please her." / "She is hard to please."
- Temporal clause reordering: "After eating, she left." / "She left after eating."
- Genitive alternation: "The book of the student." / "The student's book."

EXAMPLES
{"type":"syntactic","sentence_1":"The manager approved the new plan yesterday.","sentence_2":"The new plan was approved by the manager yesterday."}
{"type":"syntactic","sentence_1":"She gave the report to her colleague.","sentence_2":"She gave her colleague the report."}
{"type":"syntactic","sentence_1":"After finishing her coffee, she left the office.","sentence_2":"She left the office after finishing her coffee."}
{"type":"syntactic","sentence_1":"It surprised everyone that he refused the offer.","sentence_2":"That he refused the offer surprised everyone."}"""

#: System prompt for **semantic** variation generation.
SEMANTIC_SYSTEM_PROMPT = """You are a computational linguistics expert specializing in controlled dataset construction for NLP probing experiments. Your task is to generate high-quality semantic variation sentence pairs for a linguistic probing study.

TASK
Generate sentence pairs where sentence_1 and sentence_2 are semantically related. They must describe the same event or state, but differ substantially in both vocabulary and syntactic form.

RULES
1. The sentences must differ meaningfully in both vocabulary and structure — not just a synonym swap (that's lexical) or a passive/active transformation (that's syntactic).
2. The relationship must be clear and unambiguous — avoid pairs where the paraphrase relationship could be debatable or context-dependent.
3. Sentences should be everyday, concrete, and natural — no jargon, no abstract philosophy.
4. Sentences should be 8–18 words long.
5. Be conservative: if the equivalence could be disputed, discard the pair.

PHENOMENA TO COVER (cycle through these):
- Negation paraphrase: "The meeting was canceled." / "The meeting did not take place."
- Entailment: "The car stopped." / "The car is no longer moving."
- Scalar implicature: "He ate some of the cake." / "He did not eat all of the cake."
- Existential paraphrase: "There are no seats left." / "All seats are taken."
- Resultative: "She locked the door." / "The door is now locked."
- Aspectual paraphrase: "He has finished the report." / "The report is done."

EXAMPLES
{"type":"semantic","sentence_1":"The meeting was canceled.","sentence_2":"The meeting did not take place."}
{"type":"semantic","sentence_1":"All of the guests arrived on time.","sentence_2":"None of the guests were late."}
{"type":"semantic","sentence_1":"She has finished all her work.","sentence_2":"No work remains for her to do."}
{"type":"semantic","sentence_1":"The store ran out of milk.","sentence_2":"There was no milk left at the store."}"""

#: Maps variation type name to its system prompt.
VARIATION_SYSTEM_PROMPTS = {
    "lexical": LEXICAL_SYSTEM_PROMPT,
    "syntactic": SYNTACTIC_SYSTEM_PROMPT,
    "semantic": SEMANTIC_SYSTEM_PROMPT,
}

#: Canonical ordering of variation types.
VARIATION_TYPES = ["lexical", "syntactic", "semantic"]


class LinguisticVariationPipeline:
    """Generate linguistic variation sentence pairs via the Cohere API.

    Attributes
    ----------
    client : cohere.ClientV2
        Authenticated Cohere client.
    model : str
        Cohere model identifier (e.g. ``"command-a-03-2025"``).
    batch_size : int
        Number of pairs requested per API call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "command-a-03-2025",
        batch_size: int = 50,
    ):
        """Initialise the pipeline.

        Parameters
        ----------
        api_key : str or None
            Cohere API key.  Falls back to ``COHERE_API_KEY`` env var.
        model : str
            Cohere model identifier.
        batch_size : int
            Pairs per API call.
        """
        self.client = cohere.ClientV2(api_key=api_key or os.getenv("COHERE_API_KEY"))
        self.model = model
        self.batch_size = batch_size

    def generate_batch(
        self,
        variation_type: str,
        batch_size: int,
        recent_pairs: list[dict],
        used_swap_words: set | None = None,
    ) -> list[SentencePair]:
        """Request one batch of sentence pairs from the Cohere API.

        Parameters
        ----------
        variation_type : str
            One of ``"lexical"``, ``"syntactic"``, ``"semantic"``.
        batch_size : int
            Number of pairs to request.
        recent_pairs : list[dict]
            Recently generated pairs (last 4 shown to the model to
            discourage repetition).
        used_swap_words : set or None
            For lexical type, the set of already-used swap tuples so
            the model avoids repeating them.

        Returns
        -------
        list[SentencePair]
            Parsed sentence pairs from the API response.
        """
        # Inject the last 4 generated pairs so the model avoids repeating them
        recency_note = ""
        if recent_pairs:
            examples = "\n".join(
                f'  "{p.sentence_1}" / "{p.sentence_2}"'
                for p in recent_pairs[-4:]
            )
            recency_note = f"\n\nAvoid generating pairs similar to these recently generated ones:\n{examples}"

        # For lexical, tell the model which exact swap pairs are already used
        used_words_note = ""
        if variation_type == "lexical" and used_swap_words:
            pairs_list = ", ".join(f"{a}/{b}" for a, b in sorted(used_swap_words))
            used_words_note = (
                f"\n\nDo NOT reuse any of these already-used synonym swaps — "
                f"choose entirely different word pairs:\n{pairs_list}"
            )

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": VARIATION_SYSTEM_PROMPTS[variation_type]},
                {"role": "user", "content": (
                    f"Generate exactly {batch_size} {variation_type} variation sentence pairs. "
                    f"Ensure variety in topics, sentence length, and phenomena types across the batch."
                    + recency_note
                    + used_words_note
                )},
            ],
            response_format={
                "type": "json_object",
                "json_schema": SentencePairBatch.model_json_schema(),
            },
        )
        data = json.loads(response.message.content[0].text)
        return SentencePairBatch(**data).pairs

    def generate(
        self,
        pairs_per_type: int = 1000,
        output_path: str | Path | None = None,
        interactive: bool = True,
        variation_types: list[str] | None = None,
    ) -> list[dict]:
        """Run the full generation loop for all requested variation types.

        Generates *pairs_per_type* pairs for each variation type,
        deduplicating in-flight and saving after every batch so
        progress is never lost.  In interactive mode the user can
        stop a single type or the entire run.

        Parameters
        ----------
        pairs_per_type : int
            Target number of pairs per variation type.
        output_path : str, Path, or None
            JSON file to write results to (created or appended).
        interactive : bool
            If *True*, prompt the user after every batch.
        variation_types : list[str] or None
            Subset of types to generate.  Defaults to all three.

        Returns
        -------
        list[dict]
            All generated pairs (including any pre-existing ones
            loaded from *output_path*).
        """
        # Load existing results if the file already has data
        all_results = []
        if output_path and Path(output_path).exists():
            with open(output_path, encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing pairs from {output_path}")

        pair_counter = max((int(r["pair_id"]) for r in all_results), default=0) + 1

        for variation_type in (variation_types or VARIATION_TYPES):
            print(f"\n--- {variation_type.upper()} ({pairs_per_type} pairs) ---")
            generated = []
            seen_sentences = set()
            stopped_early = False

            # Load used swap pairs for lexical to avoid repeating swaps
            used_swap_words: set = set()
            if variation_type == "lexical" and output_path:
                used_words_path = Path(output_path).parent / "lexical_used_words.json"
                if used_words_path.exists():
                    with open(used_words_path) as f:
                        data = json.load(f)
                    # Support both old format (word->id) and new format (a/b->id)
                    used_swap_words = {tuple(k.split("/")) for k in data.keys() if "/" in k}
                    print(f"  Loaded {len(used_swap_words)} used swap pairs")

            # Also track seen swap pairs for lexical dedup during generation
            from dedup_dataset import extract_swap as _extract_swap
            seen_swaps: set = set(used_swap_words)  # pre-seed with already-used pairs

            stalled_batches = 0
            while len(generated) < pairs_per_type:
                remaining = pairs_per_type - len(generated)
                # Always request a full batch so we have more candidates to dedup from
                batch_size = self.batch_size

                # Retry on transient network errors
                for attempt in range(3):
                    try:
                        raw_batch = self.generate_batch(variation_type, batch_size, generated)
                        break
                    except Exception as e:
                        if attempt == 2:
                            print(f"  API error after 3 attempts, saving and stopping: {e}")
                            self._save(all_results, output_path)
                            return all_results
                        print(f"  API error (attempt {attempt + 1}/3), retrying in 5s: {e}")
                        time.sleep(5)

                # Dedup: drop sentence duplicates and (for lexical) swap duplicates
                new_pairs = []
                for pair in raw_batch:
                    key = pair.sentence_1.strip().lower()
                    if key in seen_sentences:
                        continue
                    if variation_type == "lexical":
                        swap = _extract_swap(pair.sentence_1, pair.sentence_2)
                        if swap is None or swap in seen_swaps:
                            continue
                        seen_swaps.add(swap)
                    seen_sentences.add(key)
                    new_pairs.append(pair)
                    if len(generated) + len(new_pairs) >= pairs_per_type:
                        break

                generated.extend(new_pairs)
                duplicates_dropped = len(raw_batch) - len(new_pairs)

                # Save after every batch so progress is never lost
                for pair in new_pairs:
                    all_results.append({
                        "pair_id": str(pair_counter),
                        "type": variation_type,
                        "sentence_1": pair.sentence_1,
                        "sentence_2": pair.sentence_2,
                    })
                    pair_counter += 1
                self._save(all_results, output_path)

                if len(new_pairs) == 0:
                    stalled_batches += 1
                else:
                    stalled_batches = 0

                if stalled_batches >= 15:
                    print(f"  Stalled — stopping {variation_type} at {len(generated)} pairs.")
                    break

                if interactive:
                    print(f"\nBatch ({len(new_pairs)} new, {duplicates_dropped} duplicates dropped):")
                    for i, pair in enumerate(new_pairs, 1):
                        print(f"  {i}. [{variation_type}]")
                        print(f"     s1: {pair.sentence_1}")
                        print(f"     s2: {pair.sentence_2}")
                    print(f"\n  Progress: {len(generated)}/{pairs_per_type}")
                    choice = input("  [Enter] continue  |  [q] stop this type  |  [Q] stop everything: ").strip().lower()
                    if choice == "q":
                        print(f"  Stopping {variation_type} early with {len(generated)} pairs.")
                        stopped_early = True
                        break

                    elif choice == "qq" or choice == "Q":
                        print("  Stopping generation entirely.")
                        return all_results
                else:
                    print(f"  {variation_type}: {len(generated)}/{pairs_per_type} (+{len(new_pairs)}, -{duplicates_dropped} dupes)")

            print(f"  Finished {variation_type}: {len(generated)} pairs saved.")

        return all_results

    def _save(self, results: list[dict], output_path: str | Path | None):
        """Persist *results* to *output_path* as JSON (no-op if path is None)."""
        if not output_path:
            return
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} pairs to {path}")


if __name__ == "__main__":
    pipeline = LinguisticVariationPipeline(model="command-a-03-2025", batch_size=50)

    results = pipeline.generate(
        pairs_per_type=1000,
        output_path="src/data/linguistic_variation/linguistic_variation.json",
        interactive=False,
    )
