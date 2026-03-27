"""
Linguistic Alignment Analyzer.

Measures how cosine similarity between typed sentence pairs evolves
across transformer layers. This reveals which layers geometrically
cluster sentences that share meaning (semantic), structure (syntactic),
or surface tokens (lexical).

This is NOT linear probing. No classifier is trained. The analysis is
purely geometric: at layer L, are semantically equivalent sentences
closer together in the representation space than structurally equivalent
or lexically overlapping ones?

Core workflow::

    pairs = [
        AlignmentPair(
            source="The meeting was cancelled.",
            target="The meeting did not take place.",
            source_lang="en", target_lang="en",
            pair_type="semantic",
            pair_id=0,
            linguistic_contrast="paraphrase: same meaning, no shared content words",
        ),
        ...
    ]

    inferencer = MultilingualInference("CohereLabs/tiny-aya-global")
    alignment = TypeAlignmentAnalyzer.from_pairs(pairs, inferencer)
    alignment.save("./outputs/alignment_en/")
    df = alignment.to_dataframe()  # (num_layers, 4): layer, semantic, syntactic, lexical

Or, if you already have an InferenceResult::

    result = inferencer.extract(sentences, metadata)
    analyzer = TypeAlignmentAnalyzer(result)
    alignment = analyzer.compute()

Pooling note (addressing peer review):
    Mean pooling averages all token representations and does not explicitly
    encode word order. However, transformer hidden states are already
    contextualised — positional information is baked in. This means even
    mean-pooled vectors carry implicit order information, but the effect may
    be weaker for syntactic analysis where word order is the key signal.

    To compare pooling strategies, pass ExtractionConfig(pooling="last")
    instead of the default "mean". The "last" strategy uses the final
    non-padding token, which in decoder-only models accumulates context
    from the full sequence via causal attention::

        config_mean = ExtractionConfig(pooling="mean")
        config_last = ExtractionConfig(pooling="last")

        alignment_mean = TypeAlignmentAnalyzer.from_pairs(
            pairs, inferencer, config_mean
        )
        alignment_last = TypeAlignmentAnalyzer.from_pairs(
            pairs, inferencer, config_last
        )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import h5py
import numpy as np

from type_alignment.inference import (
    ExtractionConfig,
    InferenceResult,
    MultilingualInference,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D float vectors.

    Returns 0.0 if either vector has near-zero norm (degenerate case).
    The epsilon (1e-9) matches the inline logic in the root inference files.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# AlignmentPair — input specification
# ---------------------------------------------------------------------------

@dataclass
class AlignmentPair:
    """
    A single typed probing pair for linguistic alignment analysis.

    Encodes one (source, target) comparison with full provenance —
    what languages are involved, what linguistic dimension this pair
    isolates, and a human-readable description of the contrast.

    Attributes:
        source: The source sentence text.
        target: The target sentence text.
        source_lang: BCP-47 language code for the source (e.g. "en").
        target_lang: BCP-47 language code for the target (e.g. "en").
        pair_type: The linguistic dimension being probed.
            "semantic"  — same meaning, different words / language.
            "syntactic" — same grammatical structure, different content.
            "lexical"   — high word overlap, different meaning.
        pair_id: Integer identifier. Must be unique within a pair_type
            when used in the same analysis run.
        linguistic_contrast: Human-readable description of what this
            pair isolates. Stored in metadata and in saved JSON for
            reproducibility.
    """

    source: str
    target: str
    source_lang: str
    target_lang: str
    pair_type: Literal["semantic", "syntactic", "lexical"]
    pair_id: int
    linguistic_contrast: str


# ---------------------------------------------------------------------------
# AlignmentCurve — result for one linguistic dimension
# ---------------------------------------------------------------------------

@dataclass
class AlignmentCurve:
    """
    Aggregated cosine similarity scores for one pair type across all layers.

    Attributes:
        pair_type: Which linguistic dimension these scores represent.
        scores_per_layer: Mean cosine similarity across all pairs at
            each layer. Shape: (num_layers,). This is the primary curve
            used for plotting.
        per_pair_scores: Individual pair cosine similarities.
            Shape: (num_pairs, num_layers). Row i is the similarity
            curve for pair_ids[i].
        pair_ids: The pair_id values corresponding to each row of
            per_pair_scores, in the same order.
        peak_layer: Architectural layer index at which scores_per_layer
            is maximised. If sparse extraction was used (config.layers
            not None), this is the original architectural index, not a
            positional index into the extracted set.
        peak_score: The value of scores_per_layer at peak_layer.
    """

    pair_type: str
    scores_per_layer: np.ndarray    # shape (num_layers,)
    per_pair_scores: np.ndarray     # shape (num_pairs, num_layers)
    pair_ids: List[int]
    peak_layer: int
    peak_score: float


def _empty_layer_scores(pair_type: str, num_layers: int) -> AlignmentCurve:
    """Return a NaN-filled AlignmentCurve for a missing pair type."""
    return AlignmentCurve(
        pair_type=pair_type,
        scores_per_layer=np.full(num_layers, np.nan),
        per_pair_scores=np.empty((0, num_layers)),
        pair_ids=[],
        peak_layer=-1,
        peak_score=float("nan"),
    )


# ---------------------------------------------------------------------------
# Private utility — used by AlignmentResult.compute()
# ---------------------------------------------------------------------------

def _find_crossover(a: np.ndarray, b: np.ndarray) -> Optional[int]:
    """Return first index where a[i] > b[i], or None if it never occurs."""
    for i in range(len(a)):
        if not np.isnan(a[i]) and not np.isnan(b[i]) and a[i] > b[i]:
            return i
    return None


# ---------------------------------------------------------------------------
# AlignmentResult — full structured output
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """
    Complete structured output of a linguistic alignment analysis run.

    Attributes:
        semantic: Alignment scores for semantic (paraphrase / translation) pairs.
        syntactic: Alignment scores for syntactic (structure-matched) pairs.
        lexical: Alignment scores for lexical (word-overlap) pairs.
        num_layers: Total number of extracted layers.
        model_name: HuggingFace model ID that produced the embeddings.
        crossover_semantic_over_lexical: First layer index where semantic
            similarity exceeds lexical similarity, or None if it never does.
            Marks where the model transitions from surface-token matching to
            meaning-based alignment.
        crossover_semantic_over_syntactic: First layer where semantic
            similarity exceeds syntactic similarity, or None.
    """

    semantic: AlignmentCurve
    syntactic: AlignmentCurve
    lexical: AlignmentCurve
    num_layers: int
    model_name: str
    crossover_semantic_over_lexical: Optional[int]
    crossover_semantic_over_syntactic: Optional[int]

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save to disk.

        Creates a directory at `path` containing:
            scores.h5       — numpy arrays (HDF5, gzip compressed)
            alignment.json  — scalar fields, crossovers, peak info

        HDF5 layout::

            /semantic/scores_per_layer     (num_layers,)
            /semantic/per_pair_scores      (num_pairs, num_layers)
            /syntactic/...
            /lexical/...

        JSON layout::

            {
              "model_name": "...",
              "num_layers": 4,
              "crossover_semantic_over_lexical": 2,
              "crossover_semantic_over_syntactic": null,
              "semantic":  {"pair_type": "semantic",
                            "pair_ids": [...],
                            "peak_layer": 3, "peak_score": 0.91},
              ...
            }
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # --- arrays ---
        with h5py.File(out / "scores.h5", "w") as f:
            for scores in (self.semantic, self.syntactic, self.lexical):
                grp = f.create_group(scores.pair_type)
                grp.create_dataset(
                    "scores_per_layer",
                    data=scores.scores_per_layer,
                    compression="gzip",
                )
                grp.create_dataset(
                    "per_pair_scores",
                    data=scores.per_pair_scores,
                    compression="gzip",
                )

        # --- scalars ---
        meta: Dict = {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "crossover_semantic_over_lexical": self.crossover_semantic_over_lexical,
            "crossover_semantic_over_syntactic": self.crossover_semantic_over_syntactic,
        }
        for scores in (self.semantic, self.syntactic, self.lexical):
            meta[scores.pair_type] = {
                "pair_type": scores.pair_type,
                "pair_ids": scores.pair_ids,
                "peak_layer": scores.peak_layer,
                "peak_score": (
                    float(scores.peak_score)
                    if not np.isnan(scores.peak_score)
                    else None
                ),
            }

        with open(out / "alignment.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AlignmentResult":
        """Load a previously saved AlignmentResult."""
        p = Path(path)

        with open(p / "alignment.json", encoding="utf-8") as f:
            meta = json.load(f)

        scores_map: Dict[str, AlignmentCurve] = {}
        with h5py.File(p / "scores.h5", "r") as f:
            for pt in ("semantic", "syntactic", "lexical"):
                grp = f[pt]
                spl = grp["scores_per_layer"][:]
                pps = grp["per_pair_scores"][:]
                m = meta[pt]
                peak_score = m["peak_score"] if m["peak_score"] is not None else float("nan")
                scores_map[pt] = AlignmentCurve(
                    pair_type=pt,
                    scores_per_layer=spl,
                    per_pair_scores=pps,
                    pair_ids=m["pair_ids"],
                    peak_layer=m["peak_layer"],
                    peak_score=peak_score,
                )

        return cls(
            semantic=scores_map["semantic"],
            syntactic=scores_map["syntactic"],
            lexical=scores_map["lexical"],
            num_layers=meta["num_layers"],
            model_name=meta["model_name"],
            crossover_semantic_over_lexical=meta["crossover_semantic_over_lexical"],
            crossover_semantic_over_syntactic=meta["crossover_semantic_over_syntactic"],
        )

    # ----------------------------------------------------------------
    # DataFrame export
    # ----------------------------------------------------------------

    def to_dataframe(self):
        """
        Return a tidy pandas DataFrame for downstream plotting.

        Columns: layer (int), semantic, syntactic, lexical (float).
        Missing pair types appear as NaN columns.

        Returns:
            DataFrame of shape (num_layers, 4).
        """
        import pandas as pd  # inline import — not required for core usage

        return pd.DataFrame(
            {
                "layer": list(range(self.num_layers)),
                "semantic": self.semantic.scores_per_layer,
                "syntactic": self.syntactic.scores_per_layer,
                "lexical": self.lexical.scores_per_layer,
            }
        )

    # ----------------------------------------------------------------
    # Layer specialization
    # ----------------------------------------------------------------

    def dominant_type_per_layer(self) -> List[str]:
        """
        Return the dominant linguistic signal at each layer.

        For each layer, returns whichever of "semantic", "syntactic",
        "lexical" has the highest mean cosine similarity score at that
        layer. Returns "unknown" if all three scores are NaN (e.g. a
        pair type was missing from the InferenceResult).

        This is the "Layer Specialization" analysis described in the
        pipeline spec:

            dominant[layer] = argmax(
                lexical_scores[layer],
                syntactic_scores[layer],
                semantic_scores[layer],
            )

        Returns:
            List of length num_layers, e.g.:
                ["lexical", "lexical", "syntactic", "semantic"]
        """
        dominant = []
        for i in range(self.num_layers):
            candidates = {
                "semantic": float(self.semantic.scores_per_layer[i]),
                "syntactic": float(self.syntactic.scores_per_layer[i]),
                "lexical": float(self.lexical.scores_per_layer[i]),
            }
            valid = {k: v for k, v in candidates.items() if not np.isnan(v)}
            if not valid:
                dominant.append("unknown")
            else:
                dominant.append(max(valid, key=lambda k: valid[k]))
        return dominant

    # ----------------------------------------------------------------
    # Info
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        model_short = (
            self.model_name.split("/")[-1]
            if "/" in self.model_name
            else self.model_name
        )
        sem_peak = f"layer {self.semantic.peak_layer} ({self.semantic.peak_score:.3f})"
        syn_peak = f"layer {self.syntactic.peak_layer} ({self.syntactic.peak_score:.3f})"
        lex_peak = f"layer {self.lexical.peak_layer} ({self.lexical.peak_score:.3f})"
        return (
            f"AlignmentResult(\n"
            f"  model='{model_short}', layers={self.num_layers}\n"
            f"  semantic_peak={sem_peak}\n"
            f"  syntactic_peak={syn_peak}\n"
            f"  lexical_peak={lex_peak}\n"
            f"  crossover_sem_over_lex={self.crossover_semantic_over_lexical}\n"
            f"  crossover_sem_over_syn={self.crossover_semantic_over_syntactic}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# TypeAlignmentAnalyzer — main class
# ---------------------------------------------------------------------------

class TypeAlignmentAnalyzer:
    """
    Computes per-layer cosine similarity profiles for semantic, syntactic,
    and lexical sentence pairs from an already-extracted InferenceResult.

    This formalises the inline analysis loop from the original exploration
    into a reusable, typed class.

    The core question answered:
        At which layers does the model geometrically cluster sentences by
        meaning (semantic), structure (syntactic), or surface form (lexical)?
        At which layer does meaning-encoding overtake surface-form matching?

    Usage::

        result = inferencer.extract(sentences, metadata, config)
        analyzer = TypeAlignmentAnalyzer(result)
        alignment = analyzer.compute()
        alignment.save("./outputs/alignment_en/")

    Or via the convenience classmethod::

        alignment = TypeAlignmentAnalyzer.from_pairs(
            pairs, inferencer
        )
    """

    def __init__(self, inference_result: InferenceResult) -> None:
        """
        Args:
            inference_result: Must contain sentences with at least one of
                pair_type in {"semantic", "syntactic", "lexical"} and both
                pair_role "source" and "target" per pair_id.

        Raises:
            ValueError: If inference_result has no metadata entries.
        """
        if not inference_result.metadata:
            raise ValueError("InferenceResult has no metadata entries.")
        self._result = inference_result

    # ----------------------------------------------------------------
    # compute — core algorithm
    # ----------------------------------------------------------------

    def compute(self) -> AlignmentResult:
        """
        Compute per-layer alignment scores for all three pair types.

        Algorithm per pair type:
            1. Filter InferenceResult to rows of this pair_type.
            2. If no rows → emit warning, use NaN AlignmentCurve.
            3. For each layer_idx:
               For each pair_id:
                 - Look up source and target rows by pair_role in metadata
                   (never assume positional order).
                 - Compute cosine similarity between their layer embeddings.
            4. Mean over pairs → scores_per_layer curve.
            5. peak_layer = architectural index at argmax of the mean curve.

        Crossover detection after all three types are computed.

        Returns:
            AlignmentResult with fully populated scores and crossover layers.
        """
        result = self._result
        num_layers = result.num_layers
        extracted_layers: List[int] = result.config.get(
            "layers", list(range(num_layers))
        )

        scores_by_type: Dict[str, AlignmentCurve] = {}

        for pair_type in ("semantic", "syntactic", "lexical"):
            subset = result.filter(pair_type=pair_type)

            if subset.num_sentences == 0:
                logger.warning(
                    "No sentences with pair_type='%s' found in InferenceResult. "
                    "AlignmentCurve for this type will be NaN.",
                    pair_type,
                )
                scores_by_type[pair_type] = _empty_layer_scores(pair_type, num_layers)
                continue

            pair_ids = sorted(set(m["pair_id"] for m in subset.metadata))
            per_pair = np.zeros((len(pair_ids), num_layers))

            for layer_pos in range(num_layers):
                layer_idx = extracted_layers[layer_pos]
                emb_layer = subset.get_layer(layer_idx)  # (num_sentences, hidden_dim)

                for i, pid in enumerate(pair_ids):
                    pair_sub = subset.filter(pair_id=pid)

                    if pair_sub.num_sentences < 2:
                        raise ValueError(
                            f"pair_type='{pair_type}', pair_id={pid}: expected source "
                            f"and target rows but found {pair_sub.num_sentences} sentence(s). "
                            f"Check your metadata for missing pair_role fields."
                        )

                    # Look up source/target by pair_role — never assume row order.
                    src_indices = [
                        j for j, m in enumerate(pair_sub.metadata)
                        if m.get("pair_role") == "source"
                    ]
                    tgt_indices = [
                        j for j, m in enumerate(pair_sub.metadata)
                        if m.get("pair_role") == "target"
                    ]

                    if not src_indices or not tgt_indices:
                        raise ValueError(
                            f"pair_type='{pair_type}', pair_id={pid}: could not find "
                            f"both pair_role='source' and pair_role='target' in metadata. "
                            f"Found roles: {[m.get('pair_role') for m in pair_sub.metadata]}"
                        )

                    # Get the embeddings for this pair at this layer.
                    # pair_sub.get_layer returns (num_sentences_in_pair, hidden_dim).
                    pair_emb = pair_sub.get_layer(layer_idx)
                    per_pair[i, layer_pos] = _cosine_similarity(
                        pair_emb[src_indices[0]], pair_emb[tgt_indices[0]]
                    )

            scores_per_layer = per_pair.mean(axis=0)
            peak_pos = int(np.argmax(scores_per_layer))
            peak_layer = extracted_layers[peak_pos]
            peak_score = float(scores_per_layer[peak_pos])

            scores_by_type[pair_type] = AlignmentCurve(
                pair_type=pair_type,
                scores_per_layer=scores_per_layer,
                per_pair_scores=per_pair,
                pair_ids=pair_ids,
                peak_layer=peak_layer,
                peak_score=peak_score,
            )

        semantic = scores_by_type["semantic"]
        syntactic = scores_by_type["syntactic"]
        lexical = scores_by_type["lexical"]

        crossover_sem_lex = _find_crossover(
            semantic.scores_per_layer, lexical.scores_per_layer
        )
        crossover_sem_syn = _find_crossover(
            semantic.scores_per_layer, syntactic.scores_per_layer
        )

        return AlignmentResult(
            semantic=semantic,
            syntactic=syntactic,
            lexical=lexical,
            num_layers=num_layers,
            model_name=result.model_name,
            crossover_semantic_over_lexical=crossover_sem_lex,
            crossover_semantic_over_syntactic=crossover_sem_syn,
        )

    # ----------------------------------------------------------------
    # from_pairs — full pipeline convenience classmethod
    # ----------------------------------------------------------------

    def compute_retrieval_mrr(
        self, pair_type: str = "lexical"
    ) -> np.ndarray:
        """
        Compute Mean Reciprocal Rank (MRR) across all layers for a pair type.

        For each layer, builds source and target embedding matrices from the
        aligned pairs and calls compute_mrr() from retrieval_metrics. This
        tests whether the model's representations can functionally retrieve
        the correct counterpart by nearest-neighbor search.

        Interpretation:
            MRR = 1.0 → every source embedding retrieves its exact target
                        as nearest neighbor (perfect functional alignment).
            MRR ≈ 1/N → retrieval is essentially random (no alignment).

        The "Alignment Transition Layer" from the spec can be read off as
        the point where MRR begins rising sharply for semantic pairs.

        Args:
            pair_type: Which pair type to evaluate. Defaults to "lexical"
                since cross-lingual word translations are the canonical
                retrieval test (dog ↔ perro). Use "semantic" for paraphrase
                retrieval.

        Returns:
            Array of shape (num_layers,) with MRR score per layer.

        Raises:
            ValueError: If no sentences of the given pair_type are present.
        """
        from type_alignment.retrieval_metrics import compute_mrr  # noqa: PLC0415

        result = self._result
        subset = result.filter(pair_type=pair_type)

        if subset.num_sentences == 0:
            raise ValueError(
                f"No sentences with pair_type='{pair_type}' in InferenceResult."
            )

        pair_ids = sorted(set(m["pair_id"] for m in subset.metadata))
        num_layers = result.num_layers
        extracted_layers: List[int] = result.config.get(
            "layers", list(range(num_layers))
        )

        mrr_per_layer = np.zeros(num_layers)

        for layer_pos in range(num_layers):
            layer_idx = extracted_layers[layer_pos]

            src_embs = []
            tgt_embs = []

            for pid in pair_ids:
                pair_sub = subset.filter(pair_id=pid)
                pair_emb = pair_sub.get_layer(layer_idx)

                src_indices = [
                    j for j, m in enumerate(pair_sub.metadata)
                    if m.get("pair_role") == "source"
                ]
                tgt_indices = [
                    j for j, m in enumerate(pair_sub.metadata)
                    if m.get("pair_role") == "target"
                ]

                if not src_indices or not tgt_indices:
                    raise ValueError(
                        f"pair_type='{pair_type}', pair_id={pid}: missing "
                        f"source or target row."
                    )

                src_embs.append(pair_emb[src_indices[0]])
                tgt_embs.append(pair_emb[tgt_indices[0]])

            src_matrix = np.stack(src_embs, axis=0)   # (N, hidden_dim)
            tgt_matrix = np.stack(tgt_embs, axis=0)   # (N, hidden_dim)
            mrr_per_layer[layer_pos] = compute_mrr(src_matrix, tgt_matrix)

        return mrr_per_layer

    def compute_cross_language_consistency(
        self,
        pair_type: str = "semantic",
    ) -> CrossLanguageConsistencyResult:
        """
        Compute cross-language consistency for a given pair type.

        Groups pairs by (source_lang, target_lang) combination and computes
        mean cosine similarity per language pair per layer. The variance
        across language pairs at each layer reveals whether that layer is
        language-agnostic (low variance) or language-specific (high variance).

        Expected pattern:
            Early layers  — high variance (language-specific representations)
            Later layers  — low variance  (language-agnostic representations)

        Requires the InferenceResult to contain pairs from at least two
        distinct language combinations, e.g. en↔es AND en↔hi pairs of
        the same pair_type.

        Args:
            pair_type: Which pair type to analyze. Defaults to "semantic"
                since paraphrase/translation pairs most directly test
                language-agnostic meaning representations.

        Returns:
            CrossLanguageConsistencyResult with variance_per_layer as the
            primary signal.

        Raises:
            ValueError: If fewer than 2 language combinations are present
                for the given pair_type (variance is undefined).
        """
        result = self._result
        subset = result.filter(pair_type=pair_type)

        if subset.num_sentences == 0:
            raise ValueError(
                f"No sentences with pair_type='{pair_type}' in InferenceResult."
            )

        # Build proper (source_lang, target_lang) per pair_id.
        pair_ids = sorted(set(m["pair_id"] for m in subset.metadata))
        lang_pair_map: dict = {}
        for pid in pair_ids:
            pair_sub = subset.filter(pair_id=pid)
            src_meta = next(
                (m for m in pair_sub.metadata if m.get("pair_role") == "source"), None
            )
            tgt_meta = next(
                (m for m in pair_sub.metadata if m.get("pair_role") == "target"), None
            )
            if src_meta and tgt_meta:
                lp = (src_meta.get("lang", ""), tgt_meta.get("lang", ""))
                lang_pair_map[pid] = lp

        # Group pair_ids by language pair.
        groups: dict = {}
        for pid, lp in lang_pair_map.items():
            groups.setdefault(lp, []).append(pid)

        unique_lang_pairs = sorted(groups.keys())

        if len(unique_lang_pairs) < 2:
            raise ValueError(
                f"compute_cross_language_consistency requires at least 2 distinct "
                f"language pairs for pair_type='{pair_type}'. "
                f"Found: {unique_lang_pairs}. "
                f"Add pairs from multiple language combinations to the dataset."
            )

        num_layers = result.num_layers
        extracted_layers: List[int] = result.config.get(
            "layers", list(range(num_layers))
        )

        # Compute mean similarity per language pair per layer.
        sim_matrix = np.zeros((len(unique_lang_pairs), num_layers))

        for lp_idx, lp in enumerate(unique_lang_pairs):
            pids = groups[lp]

            for layer_pos in range(num_layers):
                layer_idx = extracted_layers[layer_pos]
                sims = []

                for pid in pids:
                    pair_sub = subset.filter(pair_id=pid)
                    pair_emb = pair_sub.get_layer(layer_idx)

                    src_indices = [
                        j for j, m in enumerate(pair_sub.metadata)
                        if m.get("pair_role") == "source"
                    ]
                    tgt_indices = [
                        j for j, m in enumerate(pair_sub.metadata)
                        if m.get("pair_role") == "target"
                    ]

                    if src_indices and tgt_indices:
                        sims.append(
                            _cosine_similarity(
                                pair_emb[src_indices[0]], pair_emb[tgt_indices[0]]
                            )
                        )

                sim_matrix[lp_idx, layer_pos] = float(np.mean(sims)) if sims else np.nan

        # Variance across language pairs at each layer.
        variance_per_layer = np.nanvar(sim_matrix, axis=0)

        return CrossLanguageConsistencyResult(
            pair_type=pair_type,
            language_pairs=unique_lang_pairs,
            similarity_per_lang_pair=sim_matrix,
            variance_per_layer=variance_per_layer,
            num_layers=num_layers,
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: List[AlignmentPair],
        inferencer: MultilingualInference,
        config: Optional[ExtractionConfig] = None,
    ) -> AlignmentResult:
        """
        Run the full pipeline: AlignmentPair list → AlignmentResult.

        Converts the typed pairs into the flat sentences + metadata format
        that MultilingualInference.extract() requires, runs extraction, then
        calls compute().

        Args:
            pairs: List of AlignmentPair objects. pair_ids must be unique
                within each pair_type (cross-type reuse is fine).
            inferencer: An already-loaded MultilingualInference instance.
            config: ExtractionConfig. Defaults to mean pooling, all layers,
                batch_size=8.

        Returns:
            AlignmentResult from compute().

        Raises:
            ValueError: If pairs is empty.
            ValueError: If any pair_id is duplicated within a pair_type.
        """
        if not pairs:
            raise ValueError("pairs must not be empty.")

        # Validate uniqueness of pair_id within each pair_type.
        seen: Dict[str, set] = {"semantic": set(), "syntactic": set(), "lexical": set()}
        for p in pairs:
            if p.pair_id in seen[p.pair_type]:
                raise ValueError(
                    f"Duplicate pair_id={p.pair_id} within pair_type='{p.pair_type}'."
                )
            seen[p.pair_type].add(p.pair_id)

        sentences: List[str] = []
        metadata: List[Dict] = []

        for p in pairs:
            sentences.append(p.source)
            metadata.append(
                {
                    "lang": p.source_lang,
                    "pair_id": p.pair_id,
                    "pair_type": p.pair_type,
                    "pair_role": "source",
                    "linguistic_contrast": p.linguistic_contrast,
                }
            )
            sentences.append(p.target)
            metadata.append(
                {
                    "lang": p.target_lang,
                    "pair_id": p.pair_id,
                    "pair_type": p.pair_type,
                    "pair_role": "target",
                    "linguistic_contrast": p.linguistic_contrast,
                }
            )

        result = inferencer.extract(sentences, metadata, config)
        return cls(result).compute()


# ---------------------------------------------------------------------------
# CrossLanguageConsistencyResult — result for cross-language variance analysis
# ---------------------------------------------------------------------------

@dataclass
class CrossLanguageConsistencyResult:
    """
    Result of cross-language consistency analysis for one pair type.

    Measures whether the model's similarity scores are stable across
    different language pairs at each layer. Low variance = the layer
    treats all language pairs equally (language-agnostic). High variance
    = the layer is sensitive to which specific languages are compared
    (language-specific).

    Attributes:
        pair_type: Which linguistic dimension was analyzed.
        language_pairs: The (source_lang, target_lang) combinations
            found in the data, e.g. [("en", "es"), ("en", "hi")].
        similarity_per_lang_pair: Mean cosine similarity per language
            pair per layer. Shape: (num_lang_pairs, num_layers).
            Row i corresponds to language_pairs[i].
        variance_per_layer: Variance of similarity scores across
            language pairs at each layer. Shape: (num_layers,).
            This is the primary signal: high = language-specific,
            low = language-agnostic.
        num_layers: Number of extracted layers.
    """

    pair_type: str
    language_pairs: List[tuple]
    similarity_per_lang_pair: np.ndarray   # (num_lang_pairs, num_layers)
    variance_per_layer: np.ndarray         # (num_layers,)
    num_layers: int


# ---------------------------------------------------------------------------
# Runnable smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    from uth.data.alignment_pairs_loader import load_alignment_pairs

    MODEL_NAME = "CohereLabs/tiny-aya-global"
    SAVE_PATH = "./outputs/alignment_en"

    # Cap pairs per type for a quick run; set to None to use all 1000.
    MAX_PER_TYPE: Optional[int] = None

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ---- Load Dataset 2 ----
    #
    # 3000 English pairs (1000 per type):
    #   lexical   — synonym substitution ("purchased" ↔ "bought")
    #   syntactic — structural variation (active ↔ passive, cleft, etc.)
    #   semantic  — paraphrases (same concept, different expression)
    #
    # NOTE: dataset is English-only.
    # Cross-language consistency and MRR analyses require multilingual data
    # (parallel sentences across en/es/hi/ar). Those are marked TODO below
    # and will be enabled once Dataset 3 is available (WIP).

    pairs: List[AlignmentPair] = load_alignment_pairs(max_per_type=MAX_PER_TYPE)
    print(f"Loaded {len(pairs)} pairs "
          f"({sum(1 for p in pairs if p.pair_type == 'lexical')} lexical, "
          f"{sum(1 for p in pairs if p.pair_type == 'syntactic')} syntactic, "
          f"{sum(1 for p in pairs if p.pair_type == 'semantic')} semantic)")

    # ---- Load model and run pipeline ----
    print(f"\nLoading {MODEL_NAME}...")
    inferencer = MultilingualInference(
        model_name=MODEL_NAME,
        device=device,
        dtype=torch.float16,
    )
    print(f"\n{inferencer}\n")

    config = ExtractionConfig(pooling="mean", batch_size=4)

    # Build flat sentences + metadata manually rather than calling
    # from_pairs() — so the single InferenceResult can be reused
    # for both alignment scoring (analyzer.compute()) and MRR
    # (analyzer.compute_retrieval_mrr()), avoiding a second forward pass.
    sentences: List[str] = []
    metadata: List[Dict] = []
    for p in pairs:
        sentences.append(p.source)
        metadata.append({"lang": p.source_lang, "pair_id": p.pair_id,
                          "pair_type": p.pair_type, "pair_role": "source",
                          "linguistic_contrast": p.linguistic_contrast})
        sentences.append(p.target)
        metadata.append({"lang": p.target_lang, "pair_id": p.pair_id,
                          "pair_type": p.pair_type, "pair_role": "target",
                          "linguistic_contrast": p.linguistic_contrast})

    result = inferencer.extract(sentences, metadata, config)
    analyzer = TypeAlignmentAnalyzer(result)
    alignment = analyzer.compute()

    # ---- Print similarity curves ----
    print(f"\n{alignment}\n")

    dominant = alignment.dominant_type_per_layer()
    print(f"{'Layer':>5}  {'Semantic':>10}  {'Syntactic':>10}  {'Lexical':>10}  {'Dominant':>10}")
    print("-" * 56)
    for layer_idx in range(alignment.num_layers):
        sem = alignment.semantic.scores_per_layer[layer_idx]
        syn = alignment.syntactic.scores_per_layer[layer_idx]
        lex = alignment.lexical.scores_per_layer[layer_idx]
        dom = dominant[layer_idx]
        print(f"{layer_idx:>5}  {sem:>10.4f}  {syn:>10.4f}  {lex:>10.4f}  {dom:>10}")

    # ---- Pooling comparison: mean vs. last-token ----
    #
    # Open question: mean pooling ignores token order — syntactic pairs
    # (active/passive) share the same tokens so they could appear spuriously
    # similar. Counter-argument: hidden states are already contextualized.
    # We verify empirically: if mean and last-token curves diverge significantly
    # on syntactic pairs, positional information is being washed out by pooling.
    #
    # One extra forward pass with pooling="last"; everything else is reused.
    print("\n" + "=" * 64)
    print("POOLING COMPARISON: mean vs. last-token hidden state")
    print("=" * 64)

    config_last = ExtractionConfig(pooling="last", batch_size=4)
    result_last = inferencer.extract(sentences, metadata, config_last)
    alignment_last = TypeAlignmentAnalyzer(result_last).compute()

    print(f"\n{'Layer':>5}  "
          f"{'Sem(mean)':>10}  {'Sem(last)':>10}  {'Sem(Δ)':>8}  "
          f"{'Syn(mean)':>10}  {'Syn(last)':>10}  {'Syn(Δ)':>8}  "
          f"{'Lex(mean)':>10}  {'Lex(last)':>10}  {'Lex(Δ)':>8}")
    print("-" * 102)

    for i in range(alignment.num_layers):
        sem_m = alignment.semantic.scores_per_layer[i]
        sem_l = alignment_last.semantic.scores_per_layer[i]
        syn_m = alignment.syntactic.scores_per_layer[i]
        syn_l = alignment_last.syntactic.scores_per_layer[i]
        lex_m = alignment.lexical.scores_per_layer[i]
        lex_l = alignment_last.lexical.scores_per_layer[i]
        print(
            f"{i:>5}  "
            f"{sem_m:>10.4f}  {sem_l:>10.4f}  {sem_l - sem_m:>+8.4f}  "
            f"{syn_m:>10.4f}  {syn_l:>10.4f}  {syn_l - syn_m:>+8.4f}  "
            f"{lex_m:>10.4f}  {lex_l:>10.4f}  {lex_l - lex_m:>+8.4f}"
        )

    # Summary: max absolute difference per pair type across all layers.
    # If Δ > 0.05 for syntactic pairs the two methods diverge meaningfully.
    for label, sc_m, sc_l in [
        ("semantic",  alignment.semantic.scores_per_layer,  alignment_last.semantic.scores_per_layer),
        ("syntactic", alignment.syntactic.scores_per_layer, alignment_last.syntactic.scores_per_layer),
        ("lexical",   alignment.lexical.scores_per_layer,   alignment_last.lexical.scores_per_layer),
    ]:
        max_diff = float(np.max(np.abs(sc_l - sc_m)))
        verdict = "significant — consider last-token" if max_diff > 0.05 else "within tolerance — mean pooling OK"
        print(f"  {label:>10}: max |Δ| = {max_diff:.4f}  →  {verdict}")

    # ---- MRR validation (translation retrieval) ----
    # Requires cross-lingual parallel pairs (same sentence in en + es/hi/ar).
    # TODO: enable once multilingual dataset is available (Dataset 3, WIP).
    # mrr = analyzer.compute_retrieval_mrr(pair_type="lexical")
    mrr = None

    # ---- Save and verify round-trip ----
    alignment.save(SAVE_PATH)
    print(f"\nSaved to {SAVE_PATH}/")

    loaded = AlignmentResult.load(SAVE_PATH)
    assert np.allclose(
        loaded.semantic.scores_per_layer,
        alignment.semantic.scores_per_layer,
        equal_nan=True,
    )
    assert loaded.crossover_semantic_over_lexical == alignment.crossover_semantic_over_lexical
    print("Save/load verified ✓")

    # ---- DataFrame ----
    df = alignment.to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(df.to_string(index=False))

    # ---- Plots ----
    from type_alignment.visualization import (
        plot_alignment_curves,
        plot_dominant_alignment_layers,
        plot_alignment_gaps,
        plot_retrieval_mrr_curve,
    )

    # Mean pooling: zoom y-axis to [0.75, 1.0] — values are compressed near 1.
    plot_alignment_curves(
        alignment,
        title="Linguistic Alignment Curves — mean pooling",
        y_min=0.75,
        save_path=f"{SAVE_PATH}/alignment_curves_mean.png",
    )
    plot_alignment_curves(
        alignment_last,
        title="Linguistic Alignment Curves — last-token pooling",
        save_path=f"{SAVE_PATH}/alignment_curves_last.png",
    )
    # Overlaid lines with dominant-region shading.
    plot_dominant_alignment_layers(
        alignment_last,
        save_path=f"{SAVE_PATH}/dominant_alignment_layers.png",
    )
    # Gap curves: directly shows whether semantic catches up to lexical/syntactic.
    plot_alignment_gaps(
        alignment_last,
        save_path=f"{SAVE_PATH}/alignment_gaps.png",
    )
    # TODO: uncomment once multilingual dataset is available.
    # plot_retrieval_mrr_curve(
    #     mrr,
    #     baseline_label=len([p for p in pairs if p.pair_type == "lexical"]),
    #     save_path=f"{SAVE_PATH}/retrieval_mrr_curve.png",
    # )
    print(f"\nPlots saved to {SAVE_PATH}/")
